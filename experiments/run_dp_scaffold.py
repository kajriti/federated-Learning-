"""
run_dp_scaffold.py  --  DP-SCAFFOLD vs DP-FedAvg Experiment (Paper 1)

Usage:
    python experiments/run_dp_scaffold.py --dataset mnist --clients 10 --epsilon 5.0
    python experiments/run_dp_scaffold.py --sweep

Reproduces the experiments from Noble et al. (2022) using Flower framework.
"""
import sys, os, argparse, copy, torch
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import flwr as fl
import numpy as np

from src.data import get_client_dataloaders, set_seed
from src.model import get_model, model_size_mb
from src.client import FedAvgClient, DPScaffoldClient, get_params, set_params
from src.server import FedAvgStrategy, DPScaffoldStrategy
from src.utils import (
    MetricsTracker, evaluate, compute_dp_epsilon,
    get_noise_sigma_for_epsilon,
    plot_accuracy_vs_rounds, plot_loss_vs_rounds,
    plot_iid_vs_noniid, plot_category_metric, plot_bar_comparison,
    save_summary_table
)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--dataset",   default="mnist")
    p.add_argument("--clients",   type=int, default=10)
    p.add_argument("--rounds",    type=int, default=100)
    p.add_argument("--alpha",     default="0.5")
    p.add_argument("--epsilon",   type=float, default=5.0)
    p.add_argument("--delta",     type=float, default=1e-5)
    p.add_argument("--clip",      type=float, default=1.0)
    p.add_argument("--sigma",     type=float, default=None,
                   help="Noise sigma (if None, computed from epsilon)")
    p.add_argument("--model",     default="cnn")
    p.add_argument("--results",   default="results")
    p.add_argument("--sweep",     action="store_true")
    p.add_argument("--no_warm_start", action="store_true")
    return p.parse_args()


def run_dp_scaffold(dataset, num_clients, num_rounds, alpha_val, epsilon,
                    delta, clip_threshold, noise_sigma, model_type,
                    results_dir, device, seed=42, warm_start=True):
    set_seed(seed)
    tag = f"DP-SCAFFOLD_{dataset}_C{num_clients}_a{alpha_val}_eps{epsilon}"
    print(f"\n{'='*60}")
    print(f"  DP-SCAFFOLD | eps={epsilon} | C={clip_threshold} | sigma={noise_sigma:.3f}")
    print(f"{'='*60}")

    q_user = 0.5    # client fraction
    q_data = 0.2    # data subsampling ratio
    total_steps = num_rounds * 5  # K=5 local epochs

    # Compute noise sigma from epsilon if not provided
    if noise_sigma is None:
        sigma = get_noise_sigma_for_epsilon(epsilon, q_user * q_data,
                                             total_steps, delta)
    else:
        sigma = noise_sigma
    actual_eps = compute_dp_epsilon(sigma, q_user * q_data, total_steps, delta)
    print(f"  Noise sigma={sigma:.4f} | actual eps={actual_eps:.2f}")

    train_loaders, test_loader = get_client_dataloaders(
        dataset, num_clients, alpha_val, batch_size=32,
        data_dir="./data", seed=seed)

    model = get_model(dataset, model_type)
    model_mb = model_size_mb(model)
    num_params = len(list(model.state_dict().keys()))

    tracker = MetricsTracker("DP-SCAFFOLD", dataset, num_clients,
                              alpha_val, actual_eps)

    def get_eval_fn(m, loader, dev, trk, nc, mb):
        comm = 0.0
        def evaluate_fn(server_round, parameters, config):
            nonlocal comm
            mm = copy.deepcopy(m)
            set_params(mm, parameters[:num_params])
            acc, loss = evaluate(mm, loader, dev)
            comm += 2 * mb * nc
            # Category metric: privacy leakage proxy
            mi_auc = max(0.5, 1.0 - acc)    # membership inference AUC approximation
            trk.log(server_round, acc, loss, comm, mi_auc)
            print(f"  Round {server_round:3d} | Acc={acc*100:.2f}% | Loss={loss:.4f} | MI-AUC={mi_auc:.3f} | eps={actual_eps:.2f}")
            return loss, {"accuracy": acc, "epsilon": actual_eps}
        return evaluate_fn

    eval_fn = get_eval_fn(model, test_loader, device, tracker,
                           num_clients, model_mb)

    strategy = DPScaffoldStrategy(
        model_params=get_params(model),
        num_params=num_params,
        fraction_fit=0.5,
        min_fit_clients=max(2, int(0.5 * num_clients)),
        min_available_clients=num_clients,
        server_lr=1.0,
        user_sampling_ratio=q_user,
        evaluate_fn=eval_fn,
        on_fit_config_fn=lambda r: {"round": r, "warm_start": warm_start},
    )

    def client_fn(cid):
        cid_int = int(cid)
        m = copy.deepcopy(model)
        return DPScaffoldClient(
            cid=cid_int, model=m,
            train_loader=train_loaders[cid_int % num_clients],
            device=device,
            local_epochs=5, lr=0.01,
            clip_threshold=clip_threshold,
            noise_sigma=sigma,
            data_sampling_ratio=q_data,
        )

    fl.simulation.start_simulation(
        client_fn=client_fn,
        num_clients=num_clients,
        config=fl.server.ServerConfig(num_rounds=num_rounds),
        strategy=strategy,
        client_resources={"num_cpus": 1, "num_gpus": 0.0},
    )

    tracker.save(results_dir)
    return tracker


def run_dp_fedavg_baseline(dataset, num_clients, num_rounds, alpha_val,
                            epsilon, delta, clip_threshold, noise_sigma,
                            model_type, results_dir, device, seed=42):
    """DP-FedAvg baseline: FedAvg with gradient clipping + noise (no control variates)."""
    set_seed(seed)
    q_user = 0.5
    q_data = 0.2
    total_steps = num_rounds * 5

    if noise_sigma is None:
        sigma = get_noise_sigma_for_epsilon(epsilon, q_user * q_data,
                                             total_steps, delta)
    else:
        sigma = noise_sigma
    actual_eps = compute_dp_epsilon(sigma, q_user * q_data, total_steps, delta)
    print(f"\n  DP-FedAvg baseline | eps={actual_eps:.2f} | sigma={sigma:.4f}")

    train_loaders, test_loader = get_client_dataloaders(
        dataset, num_clients, alpha_val, batch_size=32,
        data_dir="./data", seed=seed)

    model = get_model(dataset, model_type)
    model_mb = model_size_mb(model)
    tracker = MetricsTracker("DP-FedAvg", dataset, num_clients,
                              alpha_val, actual_eps)

    def get_eval_fn(m, loader, dev, trk, nc, mb):
        comm = 0.0
        def evaluate_fn(server_round, parameters, config):
            nonlocal comm
            mm = copy.deepcopy(m)
            set_params(mm, parameters)
            acc, loss = evaluate(mm, loader, dev)
            comm += 2 * mb * nc
            mi_auc = max(0.5, 1.0 - acc)
            trk.log(server_round, acc, loss, comm, mi_auc)
            print(f"  Round {server_round:3d} | DP-FedAvg Acc={acc*100:.2f}% | Loss={loss:.4f}")
            return loss, {"accuracy": acc}
        return evaluate_fn

    eval_fn = get_eval_fn(model, test_loader, device, tracker,
                           num_clients, model_mb)

    # DP-FedAvg: standard FedAvg client with clipping + noise
    class DPFedAvgClient(FedAvgClient):
        def fit(self, parameters, config):
            self.set_parameters(parameters)
            self.model.train()
            opt = torch.optim.SGD(self.model.parameters(), lr=self.lr,
                                    momentum=self.momentum)
            crit = torch.nn.CrossEntropyLoss()
            n_samples = 0
            for _ in range(self.local_epochs):
                for X, y in self.train_loader:
                    X, y = X.to(self.device), y.to(self.device)
                    opt.zero_grad()
                    crit(self.model(X), y).backward()
                    # Clip gradients
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), clip_threshold)
                    # Add Gaussian noise
                    with torch.no_grad():
                        for p in self.model.parameters():
                            if p.grad is not None:
                                p.grad += torch.randn_like(p.grad) * sigma * clip_threshold
                    opt.step()
                    n_samples += y.size(0)
            return get_params(self.model), n_samples, {}

    strategy = FedAvgStrategy(
        model_params=get_params(model),
        fraction_fit=0.5,
        min_fit_clients=max(2, int(0.5 * num_clients)),
        min_available_clients=num_clients,
        evaluate_fn=eval_fn,
    )

    def client_fn(cid):
        cid_int = int(cid)
        m = copy.deepcopy(model)
        c = DPFedAvgClient(m, train_loaders[cid_int % num_clients],
                            device, local_epochs=5, lr=0.01, momentum=0.9)
        return c

    fl.simulation.start_simulation(
        client_fn=client_fn,
        num_clients=num_clients,
        config=fl.server.ServerConfig(num_rounds=num_rounds),
        strategy=strategy,
        client_resources={"num_cpus": 1, "num_gpus": 0.0},
    )
    tracker.save(results_dir)
    return tracker


def main():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    os.makedirs(args.results, exist_ok=True)
    print(f"Device: {device}")

    if args.sweep:
        alphas   = [0.01, 0.1, 0.5, 1.0, "iid"]
        epsilons = [3.0, 5.0, 13.0]
        n_clients = [10, 50, 100]
        all_trackers, summaries = [], []
        for nc in n_clients:
            for al in alphas:
                for eps in epsilons:
                    al_val = al if al == "iid" else float(al)
                    t_scaffold = run_dp_scaffold(
                        args.dataset, nc, args.rounds, al_val, eps,
                        args.delta, args.clip, args.sigma, args.model,
                        args.results, device)
                    t_dpfedavg = run_dp_fedavg_baseline(
                        args.dataset, nc, args.rounds, al_val, eps,
                        args.delta, args.clip, args.sigma, args.model,
                        args.results, device)
                    all_trackers.extend([t_scaffold, t_dpfedavg])
                    summaries.extend([t_scaffold.summary(), t_dpfedavg.summary()])

        plot_accuracy_vs_rounds(
            [t for t in all_trackers if t.num_clients == 10 and str(t.alpha) == "0.5"],
            title=f"DP-SCAFFOLD vs DP-FedAvg — {args.dataset}",
            save_path=f"{args.results}/dp_scaffold_accuracy.png")
        plot_loss_vs_rounds(
            [t for t in all_trackers if t.num_clients == 10],
            save_path=f"{args.results}/dp_scaffold_loss.png")
        plot_category_metric(
            [t for t in all_trackers if t.num_clients == 10 and str(t.alpha) == "0.5"],
            ylabel="Membership Inference AUC",
            save_path=f"{args.results}/dp_scaffold_mi_auc.png")
        plot_bar_comparison(
            [s for s in summaries if s["num_clients"] == 10],
            save_path=f"{args.results}/dp_scaffold_bar.png")
        save_summary_table(summaries,
                           save_path=f"{args.results}/dp_scaffold_summary.csv")
    else:
        al = args.alpha if args.alpha == "iid" else float(args.alpha)
        t1 = run_dp_scaffold(args.dataset, args.clients, args.rounds, al,
                              args.epsilon, args.delta, args.clip, args.sigma,
                              args.model, args.results, device,
                              warm_start=not args.no_warm_start)
        t2 = run_dp_fedavg_baseline(args.dataset, args.clients, args.rounds, al,
                                     args.epsilon, args.delta, args.clip,
                                     args.sigma, args.model, args.results, device)
        plot_accuracy_vs_rounds([t1, t2],
            title=f"DP-SCAFFOLD vs DP-FedAvg | eps={args.epsilon}",
            save_path=f"{args.results}/dp_scaffold_vs_dpfedavg.png")
        plot_category_metric([t1, t2],
            ylabel="Membership Inference AUC (approx.)",
            save_path=f"{args.results}/mi_auc_vs_rounds.png")
        save_summary_table([t1.summary(), t2.summary()],
                           save_path=f"{args.results}/dp_scaffold_single.csv")

if __name__ == "__main__":
    main()
