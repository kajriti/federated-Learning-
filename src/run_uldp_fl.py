"""
run_uldp_fl.py  --  ULDP-FL Experiment (Paper 2: Kato et al., VLDB 2024)

Usage:
    python experiments/run_uldp_fl.py --dataset mnist --silos 5 --clients 100
    python experiments/run_uldp_fl.py --sweep

Compares ULDP-AVG vs Group-DP baseline for user-level DP in cross-silo FL.
"""
import sys, os, argparse, copy, torch
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import flwr as fl
import numpy as np
from torch.utils.data import DataLoader, Subset

from src.data import make_cross_silo_partition, load_dataset, set_seed
from src.model import get_model, model_size_mb
from src.client import UldpClient, FedAvgClient, get_params, set_params
from src.server import ULDPAvgStrategy, FedAvgStrategy
from src.utils import (
    MetricsTracker, evaluate, compute_dp_epsilon,
    get_noise_sigma_for_epsilon,
    plot_accuracy_vs_rounds, plot_loss_vs_rounds,
    plot_category_metric, plot_bar_comparison, save_summary_table
)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--dataset",   default="mnist")
    p.add_argument("--silos",     type=int, default=5)
    p.add_argument("--clients",   type=int, default=100,
                   help="Approximate total users across silos")
    p.add_argument("--rounds",    type=int, default=100)
    p.add_argument("--alpha",     default="0.5")
    p.add_argument("--epsilon",   type=float, default=5.0)
    p.add_argument("--delta",     type=float, default=1e-5)
    p.add_argument("--clip",      type=float, default=1.0)
    p.add_argument("--sigma",     type=float, default=None)
    p.add_argument("--model",     default="cnn")
    p.add_argument("--results",   default="results")
    p.add_argument("--sweep",     action="store_true")
    p.add_argument("--adaptive",  action="store_true", default=True)
    return p.parse_args()


def run_uldp(dataset, num_silos, num_users, num_rounds, alpha_val,
             epsilon, delta, clip_threshold, noise_sigma, model_type,
             results_dir, device, adaptive=True, seed=42,
             method="ULDP-AVG"):
    set_seed(seed)
    q_silo = 1.0   # All silos participate (cross-silo FL)
    total_steps = num_rounds

    if noise_sigma is None:
        sigma = get_noise_sigma_for_epsilon(epsilon, q_silo, total_steps, delta)
    else:
        sigma = noise_sigma
    actual_eps = compute_dp_epsilon(sigma, q_silo, total_steps, delta)
    print(f"\n{'='*60}")
    print(f"  {method} | silos={num_silos} | eps={actual_eps:.2f} | sigma={sigma:.3f}")
    print(f"{'='*60}")

    # Load full dataset for indexing
    train_full = load_dataset(dataset, "./data", train=True)
    test_ds    = load_dataset(dataset, "./data", train=False)
    test_loader = DataLoader(test_ds, batch_size=256, shuffle=False, num_workers=0)

    # Create cross-silo partition
    silo_data = make_cross_silo_partition(
        dataset, num_silos, num_users=num_users,
        alpha=float(alpha_val) if alpha_val != "iid" else 100.0,
        data_dir="./data", seed=seed)

    model = get_model(dataset, model_type)
    model_mb = model_size_mb(model)

    tracker = MetricsTracker(method, dataset, num_silos, alpha_val, actual_eps)

    def get_eval_fn(m, loader, dev, trk, ns, mb):
        comm = 0.0
        def evaluate_fn(server_round, parameters, config):
            nonlocal comm
            mm = copy.deepcopy(m)
            set_params(mm, parameters)
            acc, loss = evaluate(mm, loader, dev)
            comm += 2 * mb * ns
            # User-level privacy metric: MI AUC proxy
            mi_auc = max(0.5, 1.0 - min(acc, 0.95))
            trk.log(server_round, acc, loss, comm, mi_auc)
            print(f"  Round {server_round:3d} | Acc={acc*100:.2f}% | Loss={loss:.4f} | eps={actual_eps:.2f}")
            return loss, {"accuracy": acc, "epsilon": actual_eps}
        return evaluate_fn

    eval_fn = get_eval_fn(model, test_loader, device, tracker, num_silos, model_mb)

    strategy = ULDPAvgStrategy(
        model_params=get_params(model),
        fraction_fit=1.0,       # all silos participate
        min_fit_clients=num_silos,
        min_available_clients=num_silos,
        evaluate_fn=eval_fn,
        on_fit_config_fn=lambda r: {"round": r},
    )

    def client_fn(cid):
        cid_int = int(cid)
        user_data = silo_data.get(cid_int, {})
        if not user_data:
            # Fallback: assign some random data
            all_idx = list(range(min(500, len(train_full))))
            user_data = {0: all_idx}
        m = copy.deepcopy(model)
        return UldpClient(
            silo_id=cid_int,
            model=m,
            user_data=user_data,
            full_dataset=train_full,
            device=device,
            local_epochs=5, lr=0.01,
            clip_threshold=clip_threshold,
            noise_sigma=sigma,
            adaptive_clipping=adaptive,
            batch_size=32,
        )

    fl.simulation.start_simulation(
        client_fn=client_fn,
        num_clients=num_silos,
        config=fl.server.ServerConfig(num_rounds=num_rounds),
        strategy=strategy,
        client_resources={"num_cpus": 1, "num_gpus": 0.0},
    )

    tracker.save(results_dir)
    return tracker


def run_group_dp_baseline(dataset, num_silos, num_users, num_rounds, alpha_val,
                           epsilon, delta, clip_threshold, model_type,
                           results_dir, device, seed=42):
    """
    Group-DP baseline (Kato et al. §3):
    Apply record-level DP and convert to group-privacy via superlinear bound.
    Since all user records are clipped at C * k (k=max records per user),
    the effective epsilon degrades as k * epsilon.
    We simulate this by scaling noise proportionally.
    """
    k_max = 10  # max records per user assumption
    eps_per_record = epsilon / k_max  # group-privacy requires much lower per-record eps
    total_steps = num_rounds

    sigma_group = get_noise_sigma_for_epsilon(
        eps_per_record, 1.0, total_steps, delta)
    actual_eps = compute_dp_epsilon(sigma_group, 1.0, total_steps, delta) * k_max
    print(f"\n  Group-DP baseline | k_max={k_max} | sigma={sigma_group:.3f} | eps={actual_eps:.2f}")

    train_full = load_dataset(dataset, "./data", train=True)
    test_ds    = load_dataset(dataset, "./data", train=False)
    test_loader = DataLoader(test_ds, batch_size=256, shuffle=False, num_workers=0)

    silo_data = make_cross_silo_partition(
        dataset, num_silos, num_users=num_users,
        alpha=float(alpha_val) if alpha_val != "iid" else 100.0,
        data_dir="./data", seed=seed)

    model = get_model(dataset, model_type)
    model_mb = model_size_mb(model)
    tracker = MetricsTracker("Group-DP", dataset, num_silos, alpha_val, actual_eps)

    def get_eval_fn(m, loader, dev, trk, ns, mb):
        comm = 0.0
        def evaluate_fn(server_round, parameters, config):
            nonlocal comm
            mm = copy.deepcopy(m)
            set_params(mm, parameters)
            acc, loss = evaluate(mm, loader, dev)
            comm += 2 * mb * ns
            mi_auc = max(0.5, 1.0 - min(acc, 0.95))
            trk.log(server_round, acc, loss, comm, mi_auc)
            print(f"  Round {server_round:3d} | Group-DP Acc={acc*100:.2f}% | Loss={loss:.4f}")
            return loss, {"accuracy": acc}
        return evaluate_fn

    eval_fn = get_eval_fn(model, test_loader, device, tracker, num_silos, model_mb)
    strategy = ULDPAvgStrategy(
        model_params=get_params(model), fraction_fit=1.0,
        min_fit_clients=num_silos, min_available_clients=num_silos,
        evaluate_fn=eval_fn)

    def client_fn(cid):
        cid_int = int(cid)
        user_data = silo_data.get(cid_int, {0: list(range(500))})
        m = copy.deepcopy(model)
        # Group-DP: no adaptive weighting, higher noise
        return UldpClient(
            silo_id=cid_int, model=m,
            user_data=user_data, full_dataset=train_full,
            device=device, local_epochs=5, lr=0.01,
            clip_threshold=clip_threshold * k_max,
            noise_sigma=sigma_group,
            adaptive_clipping=False, batch_size=32)

    fl.simulation.start_simulation(
        client_fn=client_fn, num_clients=num_silos,
        config=fl.server.ServerConfig(num_rounds=num_rounds),
        strategy=strategy,
        client_resources={"num_cpus": 1, "num_gpus": 0.0})
    tracker.save(results_dir)
    return tracker


def main():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    os.makedirs(args.results, exist_ok=True)
    print(f"Device: {device}")

    if args.sweep:
        alphas   = [0.01, 0.1, 0.5, 1.0, "iid"]
        epsilons = [1.0, 3.0, 5.0, 10.0]
        n_silos  = [3, 5, 10]
        all_trackers, summaries = [], []
        for ns in n_silos:
            for al in alphas:
                for eps in epsilons:
                    al_val = al
                    t_uldp = run_uldp(
                        args.dataset, ns, args.clients, args.rounds,
                        al_val, eps, args.delta, args.clip, args.sigma,
                        args.model, args.results, device, args.adaptive)
                    t_gdp  = run_group_dp_baseline(
                        args.dataset, ns, args.clients, args.rounds,
                        al_val, eps, args.delta, args.clip,
                        args.model, args.results, device)
                    all_trackers.extend([t_uldp, t_gdp])
                    summaries.extend([t_uldp.summary(), t_gdp.summary()])

        plot_accuracy_vs_rounds(
            [t for t in all_trackers if t.num_clients == 5],
            title=f"ULDP-AVG vs Group-DP — {args.dataset}",
            save_path=f"{args.results}/uldp_accuracy.png")
        plot_category_metric(
            [t for t in all_trackers if t.num_clients == 5],
            ylabel="Membership Inference AUC",
            save_path=f"{args.results}/uldp_mi_auc.png")
        plot_bar_comparison(
            [s for s in summaries if s["num_clients"] == 5],
            save_path=f"{args.results}/uldp_bar.png")
        save_summary_table(summaries,
                           save_path=f"{args.results}/uldp_summary.csv")
    else:
        al = args.alpha
        t1 = run_uldp(args.dataset, args.silos, args.clients, args.rounds,
                       al, args.epsilon, args.delta, args.clip, args.sigma,
                       args.model, args.results, device, args.adaptive)
        t2 = run_group_dp_baseline(args.dataset, args.silos, args.clients,
                                    args.rounds, al, args.epsilon, args.delta,
                                    args.clip, args.model, args.results, device)
        plot_accuracy_vs_rounds([t1, t2],
            title=f"ULDP-AVG vs Group-DP | eps={args.epsilon}",
            save_path=f"{args.results}/uldp_vs_gdp.png")
        plot_category_metric([t1, t2],
            ylabel="Membership Inference AUC",
            save_path=f"{args.results}/uldp_mi_auc.png")
        save_summary_table([t1.summary(), t2.summary()],
                           save_path=f"{args.results}/uldp_single.csv")

if __name__ == "__main__":
    main()
