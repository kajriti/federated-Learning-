"""
main.py — COMPLETE VERSION
--------------------------
WHAT IS COMPLETE NOW:
1. Manual FL training loop (no Ray/Flower simulation engine needed)
2. All 4 algorithms: FedAvg, DP-FedAvg, DP-SCAFFOLD, ULDP-AVG
3. Category 1 attack metrics: membership inference AUC + gradient inversion MSE
4. All mandatory plots from the instruction doc
5. Results table with all required columns
6. IID vs Non-IID comparison
7. Privacy-utility tradeoff curve

HOW TO RUN:
    # Quick test (5 rounds, all algorithms):
    python main.py --quick_test

    # Single experiment:
    python main.py --config configs/dp_scaffold_mnist.yaml

    # Specific algorithm:
    python main.py --algorithm dp_scaffold --dataset mnist --num_clients 10 --num_rounds 50

    # Full experiment matrix (many hours):
    python main.py --run_all
"""

import os
import sys
import argparse
import yaml
import numpy as np
import torch
import random
import pandas as pd
from datetime import datetime

# ============================================================
# CRITICAL: Set seeds BEFORE anything else
# ============================================================
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed(SEED)

# ============================================================
# DEFAULT CONFIG
# ============================================================
DEFAULT_CONFIG = {
    "algorithm": "fedavg",
    "dataset": "mnist",
    "num_clients": 10,
    "alpha": 0.5,
    "num_rounds": 20,
    "client_fraction": 0.5,
    "local_epochs": 5,
    "batch_size": 32,
    "learning_rate": 0.01,
    "momentum": 0.9,
    "global_lr": 1.0,
    "model_type": "default",
    "data_sampling_ratio": 0.2,
    "user_sampling_rate": 1.0,
    "weighting_strategy": "uniform",
    "local_epochs_per_user": 1,
    "data_dir": "./data",
    "results_dir": "./results",
    "run_attacks": True,       # NEW: whether to run privacy attacks after training
    "dp": {
        "noise_multiplier": 1.0,
        "clipping_norm": 1.0,
        "delta": 1e-5,
    },
}


def load_config(config_path: str) -> dict:
    import copy
    config = copy.deepcopy(DEFAULT_CONFIG)
    if config_path and os.path.exists(config_path):
        with open(config_path, "r") as f:
            user_config = yaml.safe_load(f)
        if "dp" in user_config:
            config["dp"].update(user_config.pop("dp"))
        config.update(user_config)
        print(f"  Loaded config: {config_path}")
    return config


# ============================================================
# MANUAL FL ROUND (no Ray, no multiprocessing)
# ============================================================

def train_one_round_manually(global_params, algorithm, config, selected_clients, c_global=None):
    """
    Run one FL round by calling client.fit() directly.
    No Ray, no multiprocessing — works everywhere including Windows.
    """
    from client import FedAvgClient, DPFedAvgClient, DPScaffoldClient, ULDPAvgClient

    algorithm_map = {
        "fedavg":      FedAvgClient,
        "dp_fedavg":   DPFedAvgClient,
        "dp_scaffold": DPScaffoldClient,
        "uldp_avg":    ULDPAvgClient,
    }
    ClientClass = algorithm_map[algorithm]

    all_updates = []
    all_c_updates = []
    all_metrics = []

    for cid in selected_clients:
        client = ClientClass(client_id=cid, config=config)

        # DP-SCAFFOLD: send model params + global control variate c together
        if algorithm == "dp_scaffold" and c_global is not None:
            params_to_send = global_params + c_global
        else:
            params_to_send = global_params

        updated_params, num_samples, metrics = client.fit(
            parameters=params_to_send,
            config={"server_round": 0}
        )

        if algorithm == "dp_scaffold":
            n = len(global_params)
            delta_y = updated_params[:n]
            delta_c = updated_params[n:]
            all_updates.append((num_samples, delta_y))
            all_c_updates.append(delta_c)
        else:
            all_updates.append((num_samples, updated_params))

        all_metrics.append(metrics)

    # ---- SERVER AGGREGATION ----
    total_weight = sum(w for w, _ in all_updates)
    n_params = len(global_params)

    if algorithm == "dp_scaffold":
        # x_t = x_{t-1} + eta_g * weighted_avg(delta_y)  [Algorithm 1, Step 21a]
        new_params = []
        for j in range(n_params):
            layer = sum((w / total_weight) * dy[j] for w, dy in all_updates)
            new_params.append(global_params[j] + config["global_lr"] * layer)

        # c_t = c_{t-1} + l * avg(delta_c)  [Algorithm 1, Step 21b]
        l = config["client_fraction"]
        new_c = []
        for j in range(n_params):
            layer_c = sum(dc[j] for dc in all_c_updates) / len(all_c_updates)
            new_c.append(c_global[j] + l * layer_c)

        return new_params, new_c, all_metrics

    else:
        # Standard FedAvg: weighted average of model parameters
        new_params = []
        for j in range(n_params):
            layer = sum((w / total_weight) * params[j] for w, params in all_updates)
            new_params.append(layer)
        return new_params, None, all_metrics


def evaluate_global_model(params, config):
    """Evaluate global model on held-out test set."""
    from model import get_model, set_parameters
    from data import get_test_dataloader

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = get_model(config["dataset"], config.get("model_type", "default")).to(device)
    set_parameters(model, params)

    loader = get_test_dataloader(config["dataset"], data_dir=config.get("data_dir", "./data"))
    criterion = torch.nn.CrossEntropyLoss()
    model.eval()
    total_loss, correct, total = 0.0, 0, 0

    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            out = model(x)
            total_loss += criterion(out, y).item() * len(y)
            correct += (out.argmax(1) == y).sum().item()
            total += len(y)

    loss = total_loss / total if total > 0 else 0.0
    acc  = correct / total if total > 0 else 0.0
    return loss, acc


# ============================================================
# MAIN EXPERIMENT RUNNER
# ============================================================

def run_experiment(config: dict) -> dict:
    from model import get_model, get_parameters, get_model_size_mb
    from utils import ResultsLogger, ALGORITHM_LABELS

    algorithm        = config["algorithm"]
    dataset          = config["dataset"]
    num_clients      = config["num_clients"]
    alpha            = config["alpha"]
    num_rounds       = config["num_rounds"]
    results_dir      = config.get("results_dir", "./results")
    client_fraction  = config.get("client_fraction", 0.5)
    clients_per_round = max(2, int(num_clients * client_fraction))

    alpha_str      = str(alpha).replace(".", "p")
    exp_name       = f"{algorithm}_{dataset}_n{num_clients}_a{alpha_str}"
    timestamp      = datetime.now().strftime("%m%d_%H%M")
    exp_name_full  = f"{exp_name}_{timestamp}"

    print(f"\n{'='*60}")
    print(f"  Experiment : {exp_name}")
    print(f"  Algorithm  : {ALGORITHM_LABELS.get(algorithm, algorithm)}")
    print(f"  Dataset    : {dataset.upper()} | Clients: {num_clients} | Alpha: {alpha}")
    print(f"  Rounds     : {num_rounds} | Local epochs: {config['local_epochs']}")
    if algorithm in ("dp_scaffold", "dp_fedavg", "uldp_avg"):
        dp = config.get("dp", {})
        print(f"  DP Config  : sigma={dp.get('noise_multiplier',1.0)}, "
              f"C={dp.get('clipping_norm',1.0)}, delta={dp.get('delta',1e-5)}")
    print(f"{'='*60}")

    os.makedirs(results_dir, exist_ok=True)
    logger = ResultsLogger(exp_name_full, results_dir)

    # Initialize global model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    init_model   = get_model(dataset, config.get("model_type", "default"))
    global_params = get_parameters(init_model)

    model_mb = get_model_size_mb(init_model)
    logger.set_model_size(model_mb, clients_per_round)

    # DP-SCAFFOLD also tracks global control variate (starts at zero)
    c_global = None
    if algorithm == "dp_scaffold":
        c_global = [np.zeros_like(p) for p in global_params]

    round_accuracies = []
    round_losses     = []
    round_epsilons   = []
    convergence_round = None

    # ---- TRAINING LOOP ----
    for rnd in range(1, num_rounds + 1):

        # Sample clients for this round
        rng      = np.random.RandomState(SEED + rnd)
        selected = rng.choice(num_clients, size=clients_per_round, replace=False).tolist()

        try:
            global_params, c_global, metrics_list = train_one_round_manually(
                global_params, algorithm, config, selected, c_global
            )
        except Exception as e:
            print(f"\n  ERROR in round {rnd}: {e}")
            import traceback
            traceback.print_exc()
            break

        # Evaluate on global test set (required every round per instruction doc)
        loss, acc = evaluate_global_model(global_params, config)

        epsilons     = [m.get("epsilon", 0.0) for m in metrics_list]
        avg_eps      = float(np.mean(epsilons)) if epsilons else 0.0
        train_losses = [m.get("train_loss", 0.0) for m in metrics_list]
        avg_tl       = float(np.mean(train_losses)) if train_losses else 0.0

        round_accuracies.append(acc)
        round_losses.append(loss)
        round_epsilons.append(avg_eps)

        if convergence_round is None and acc >= 0.80:
            convergence_round = rnd
            print(f"  *** Converged at round {rnd} (>=80%) ***")

        logger.log_round(
            round_num=rnd,
            algorithm=algorithm,
            dataset=dataset,
            num_clients=num_clients,
            alpha=alpha,
            global_accuracy=acc,
            global_loss=loss,
            epsilon=avg_eps,
            train_loss=avg_tl,
        )

        if rnd % 5 == 0 or rnd == 1 or rnd == num_rounds:
            print(f"  Round {rnd:3d}/{num_rounds} | "
                  f"Acc: {acc:.4f} | Loss: {loss:.4f} | Eps: {avg_eps:.4f}")

    # ============================================================
    # POST-TRAINING: Category 1 Privacy Attacks
    # ============================================================
    attack_results = {}
    if config.get("run_attacks", True):
        try:
            from attack_metrics import run_all_privacy_attacks
            from data import get_client_dataloader, get_test_dataloader
            from model import get_model, set_parameters

            print(f"\n  Running Category 1 privacy attacks on final model...")
            attack_model = get_model(dataset, config.get("model_type", "default")).to(device)
            set_parameters(attack_model, global_params)

            train_loader = get_client_dataloader(
                client_id=0, dataset_name=dataset, num_clients=num_clients,
                alpha=alpha, batch_size=32, data_dir=config.get("data_dir", "./data")
            )
            test_loader = get_test_dataloader(dataset, data_dir=config.get("data_dir", "./data"))

            attack_results = run_all_privacy_attacks(
                model=attack_model,
                train_loader=train_loader,
                test_loader=test_loader,
                device=device,
                run_gradient_inversion=True,
                run_membership_inference=True,
            )

            print(f"\n  === ATTACK RESULTS ===")
            for k, v in attack_results.items():
                print(f"    {k}: {v:.4f}")

        except ImportError:
            print("  (attack_metrics.py not found — skipping privacy attacks)")
        except Exception as e:
            print(f"  Attack evaluation failed: {e}")

    csv_path = logger.save_csv()
    summary  = logger.get_summary()

    # Add attack results to summary
    summary.update({
        "Attack Success Rate": f"{attack_results.get('attack_success_rate', 'N/A')}",
        "MI AUC":              f"{attack_results.get('membership_inference_auc', 'N/A'):.4f}"
                               if "membership_inference_auc" in attack_results else "N/A",
        "Reconstruction MSE":  f"{attack_results.get('reconstruction_mse', 'N/A'):.6f}"
                               if "reconstruction_mse" in attack_results else "N/A",
    })

    if round_accuracies:
        print(f"\n  Final accuracy   : {round_accuracies[-1]:.4f}")
    print(f"  Convergence round: {convergence_round if convergence_round else 'not reached'}")

    return {
        "config":            config,
        "summary":           summary,
        "csv_path":          csv_path,
        "exp_name":          exp_name_full,
        "round_accuracies":  round_accuracies,
        "round_losses":      round_losses,
        "round_epsilons":    round_epsilons,
        "attack_results":    attack_results,
    }


# ============================================================
# IID vs NON-IID COMPARISON (mandatory per instruction doc)
# ============================================================

def run_iid_vs_noniid_comparison(algorithm: str = "dp_scaffold", dataset: str = "mnist"):
    """
    Run one algorithm across multiple alpha values.
    Produces the IID vs Non-IID mandatory comparison plot.

    instruction doc: "IID vs. Non-IID comparison plot"
    """
    import copy
    from utils import plot_heterogeneity_comparison

    alphas      = [0.01, 0.1, 0.5, 1.0, "IID"]
    all_results = {}
    results_dir = "./results"

    print(f"\n  Running IID vs Non-IID comparison: {algorithm} on {dataset}")

    for alpha in alphas:
        config = copy.deepcopy(DEFAULT_CONFIG)
        config.update({
            "algorithm":  algorithm,
            "dataset":    dataset,
            "num_clients": 10,
            "alpha":      alpha,
            "num_rounds": 50,
            "run_attacks": False,  # skip attacks for comparison runs
        })

        result = run_experiment(config)
        rounds = list(range(1, len(result["round_accuracies"]) + 1))

        all_results[str(alpha)] = {
            algorithm: pd.DataFrame({
                "round":           rounds,
                "global_accuracy": result["round_accuracies"],
                "global_loss":     result["round_losses"],
            })
        }

    figures_dir = os.path.join(results_dir, "figures")
    os.makedirs(figures_dir, exist_ok=True)

    plot_heterogeneity_comparison(
        all_results,
        algorithm=algorithm,
        save_path=os.path.join(figures_dir, f"iid_vs_noniid_{algorithm}_{dataset}.png")
    )
    print(f"  IID vs Non-IID plot saved!")


# ============================================================
# FULL EXPERIMENT MATRIX
# ============================================================

def run_all_experiments(quick: bool = False):
    import copy
    from utils import generate_results_table, plot_accuracy_vs_rounds, \
                      plot_loss_vs_rounds, plot_baseline_vs_methods, \
                      plot_epsilon_vs_accuracy

    results_dir = "./results"
    all_results  = {}
    all_summaries = []

    if quick:
        # Quick test: 5 rounds, all 4 algorithms, MNIST, 10 clients, alpha=0.5
        experiments = [
            {"algorithm": "fedavg",      "num_rounds": 5,  "run_attacks": False},
            {"algorithm": "dp_fedavg",   "num_rounds": 5,  "run_attacks": False},
            {"algorithm": "dp_scaffold", "num_rounds": 5,  "run_attacks": False},
            {"algorithm": "uldp_avg",    "num_rounds": 5,  "run_attacks": False},
        ]
        for e in experiments:
            e.setdefault("dataset",     "mnist")
            e.setdefault("num_clients", 10)
            e.setdefault("alpha",       0.5)
    else:
        # Full matrix per instruction doc
        algorithms = ["fedavg", "dp_fedavg", "dp_scaffold", "uldp_avg"]
        datasets   = ["mnist", "fmnist", "cifar10"]
        n_clients  = [10, 50, 100]
        alphas     = [0.01, 0.1, 0.5, 1.0, "IID"]
        experiments = [
            {
                "algorithm":  a,
                "dataset":    d,
                "num_clients": n,
                "alpha":      al,
                "num_rounds": 200,
                "run_attacks": (al == 0.5 and n == 10),  # attacks only for main config
            }
            for a in algorithms
            for d in datasets
            for n in n_clients
            for al in alphas
        ]

    print(f"\n  Total experiments: {len(experiments)}\n")

    for i, exp_override in enumerate(experiments):
        config = copy.deepcopy(DEFAULT_CONFIG)
        config.update(exp_override)
        print(f"[{i+1}/{len(experiments)}] "
              f"{config['algorithm']} | {config['dataset']} | "
              f"n={config['num_clients']} | α={config['alpha']}")
        try:
            result = run_experiment(config)
            key = (f"{config['algorithm']}_{config['dataset']}"
                   f"_n{config['num_clients']}_a{config['alpha']}")
            all_results[key] = result
            if result.get("summary"):
                all_summaries.append(result["summary"])
        except Exception as e:
            print(f"  SKIPPED: {e}")
            import traceback; traceback.print_exc()

    # Generate all required plots
    _generate_all_plots(all_results, results_dir)
    generate_results_table(all_summaries,
                           save_path=os.path.join(results_dir, "results_table.csv"))
    print(f"\n  Done! All results in: {results_dir}/")


def _generate_all_plots(all_results, results_dir):
    """Generate all mandatory plots from instruction doc."""
    from utils import (plot_accuracy_vs_rounds, plot_loss_vs_rounds,
                       plot_baseline_vs_methods, plot_epsilon_vs_accuracy)

    figures_dir = os.path.join(results_dir, "figures")
    os.makedirs(figures_dir, exist_ok=True)

    # Build per-algorithm DataFrames (use first result for each algorithm)
    algo_dfs = {}
    for key, result in all_results.items():
        algo = result["config"]["algorithm"]
        if algo not in algo_dfs and result.get("round_accuracies"):
            rounds = list(range(1, len(result["round_accuracies"]) + 1))
            algo_dfs[algo] = pd.DataFrame({
                "round":           rounds,
                "global_accuracy": result["round_accuracies"],
                "global_loss":     result["round_losses"],
                "epsilon":         result.get("round_epsilons", [0.0] * len(rounds)),
            })

    if not algo_dfs:
        print("  Nothing to plot.")
        return

    # Figure 1: Accuracy vs rounds (MANDATORY)
    plot_accuracy_vs_rounds(
        algo_dfs,
        title="Figure 1: Global Test Accuracy vs. Communication Rounds",
        save_path=os.path.join(figures_dir, "figure1_accuracy_vs_rounds.png")
    )

    # Figure 2: Loss vs rounds (MANDATORY)
    plot_loss_vs_rounds(
        algo_dfs,
        title="Figure 2: Global Test Loss vs. Communication Rounds",
        save_path=os.path.join(figures_dir, "figure2_loss_vs_rounds.png")
    )

    # Figure 3: Baseline vs methods bar chart (MANDATORY)
    plot_baseline_vs_methods(
        algo_dfs,
        save_path=os.path.join(figures_dir, "figure3_baseline_vs_methods.png")
    )

    # Figure 4: Privacy-utility tradeoff (Cat. 1 specific)
    dp_only = {k: v for k, v in algo_dfs.items() if k != "fedavg"}
    if dp_only:
        plot_epsilon_vs_accuracy(
            dp_only,
            save_path=os.path.join(figures_dir, "figure4_privacy_utility_tradeoff.png")
        )

    print(f"  All figures saved to: {figures_dir}/")


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="FL Experiment Runner — Privacy & Inference (Cat. 1)")
    parser.add_argument("--config",           type=str,   default=None,
                        help="Path to YAML config file")
    parser.add_argument("--algorithm",        type=str,   default=None,
                        choices=["fedavg", "dp_fedavg", "dp_scaffold", "uldp_avg"])
    parser.add_argument("--dataset",          type=str,   default=None,
                        choices=["mnist", "fmnist", "cifar10"])
    parser.add_argument("--num_clients",      type=int,   default=None)
    parser.add_argument("--alpha",            type=str,   default=None,
                        help="Dirichlet alpha: 0.01, 0.1, 0.5, 1.0, or IID")
    parser.add_argument("--num_rounds",       type=int,   default=None)
    parser.add_argument("--noise_multiplier", type=float, default=None,
                        help="DP noise sigma (higher = more private, lower accuracy)")
    parser.add_argument("--no_attacks",       action="store_true",
                        help="Skip privacy attack evaluation (faster)")
    parser.add_argument("--quick_test",       action="store_true",
                        help="Quick test: 5 rounds, all 4 algorithms")
    parser.add_argument("--run_all",          action="store_true",
                        help="Full experiment matrix (many hours)")
    parser.add_argument("--iid_comparison",   action="store_true",
                        help="IID vs Non-IID comparison for dp_scaffold on MNIST")
    args = parser.parse_args()

    if args.run_all:
        run_all_experiments(quick=False)

    elif args.quick_test:
        run_all_experiments(quick=True)

    elif args.iid_comparison:
        algo = args.algorithm or "dp_scaffold"
        ds   = args.dataset or "mnist"
        run_iid_vs_noniid_comparison(algorithm=algo, dataset=ds)

    else:
        import copy
        config = load_config(args.config)
        if args.algorithm:        config["algorithm"]              = args.algorithm
        if args.dataset:          config["dataset"]                = args.dataset
        if args.num_clients:      config["num_clients"]            = args.num_clients
        if args.num_rounds:       config["num_rounds"]             = args.num_rounds
        if args.noise_multiplier: config["dp"]["noise_multiplier"] = args.noise_multiplier
        if args.no_attacks:       config["run_attacks"]            = False
        if args.alpha:
            config["alpha"] = float(args.alpha) if args.alpha != "IID" else "IID"

        result = run_experiment(config)

        # Plot single experiment result
        if result.get("round_accuracies"):
            from utils import plot_accuracy_vs_rounds, ALGORITHM_LABELS
            rounds = list(range(1, len(result["round_accuracies"]) + 1))
            df = pd.DataFrame({
                "round":           rounds,
                "global_accuracy": result["round_accuracies"],
                "global_loss":     result["round_losses"],
                "epsilon":         result.get("round_epsilons", [0.0] * len(rounds)),
            })
            algo        = config["algorithm"]
            figures_dir = os.path.join(config.get("results_dir", "./results"), "figures")
            os.makedirs(figures_dir, exist_ok=True)
            plot_accuracy_vs_rounds(
                {algo: df},
                title=f"{ALGORITHM_LABELS.get(algo, algo)} — accuracy vs rounds",
                save_path=os.path.join(figures_dir, f"{result['exp_name']}_accuracy.png")
            )
            print(f"\n  Plot saved to: {figures_dir}/")
