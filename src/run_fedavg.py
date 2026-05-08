"""
run_fedavg.py  --  FedAvg Baseline Experiment Runner

Usage:
    python experiments/run_fedavg.py --dataset mnist --clients 10 --rounds 100 --alpha 0.5

Sweeps all alpha values and client counts as required by the instructions.
"""
import sys, os, argparse, copy, yaml, torch
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import flwr as fl
import numpy as np

from src.data import get_client_dataloaders, set_seed
from src.model import get_model, model_size_mb
from src.client import FedAvgClient, get_params
from src.server import FedAvgStrategy
from src.utils import (
    MetricsTracker, evaluate, communication_cost_mb,
    plot_accuracy_vs_rounds, plot_loss_vs_rounds,
    plot_iid_vs_noniid, plot_bar_comparison, save_summary_table
)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--config",   default="configs/fedavg_base.yaml")
    p.add_argument("--dataset",  default="mnist")
    p.add_argument("--clients",  type=int, default=10)
    p.add_argument("--rounds",   type=int, default=100)
    p.add_argument("--alpha",    default="0.5")
    p.add_argument("--model",    default="cnn")
    p.add_argument("--results",  default="results")
    p.add_argument("--sweep",    action="store_true",
                   help="Sweep all alphas and client counts")
    return p.parse_args()


def run_single(dataset, num_clients, num_rounds, alpha_val, model_type,
               results_dir, device, seed=42):
    """Run one FedAvg experiment and return metrics tracker."""
    set_seed(seed)
    tag = f"FedAvg_{dataset}_C{num_clients}_a{alpha_val}"
    print(f"\n{'='*60}")
    print(f"  Experiment: {tag}")
    print(f"{'='*60}")

    # Data
    train_loaders, test_loader = get_client_dataloaders(
        dataset, num_clients, alpha_val,
        batch_size=32, data_dir="./data", seed=seed)

    # Model
    model = get_model(dataset, model_type)
    model_mb = model_size_mb(model)

    tracker = MetricsTracker("FedAvg", dataset, num_clients, alpha_val)

    # Evaluate function called by Flower each round
    def get_eval_fn(m, loader, dev, trk, n_clients, rds):
        comm = 0.0
        def evaluate_fn(server_round, parameters, config):
            nonlocal comm
            from src.client import set_params
            mm = copy.deepcopy(m)
            set_params(mm, parameters)
            acc, loss = evaluate(mm, loader, dev)
            comm += 2 * model_mb * n_clients  # per-round cost
            cat_metric = 1.0 - acc  # privacy proxy: lower acc = higher privacy
            trk.log(server_round, acc, loss, comm, cat_metric)
            print(f"  Round {server_round:3d} | Acc={acc*100:.2f}% | Loss={loss:.4f}")
            return loss, {"accuracy": acc}
        return evaluate_fn

    eval_fn = get_eval_fn(model, test_loader, device, tracker,
                           num_clients, num_rounds)

    strategy = FedAvgStrategy(
        model_params=get_params(model),
        fraction_fit=0.5,
        min_fit_clients=max(2, int(0.5 * num_clients)),
        min_available_clients=num_clients,
        evaluate_fn=eval_fn,
        on_fit_config_fn=lambda r: {"round": r},
    )

    def client_fn(cid):
        cid_int = int(cid)
        m = copy.deepcopy(model)
        return FedAvgClient(
            model=m,
            train_loader=train_loaders[cid_int % num_clients],
            device=device,
            local_epochs=5, lr=0.01, momentum=0.9,
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


def main():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    os.makedirs(args.results, exist_ok=True)

    if args.sweep:
        alphas   = [0.01, 0.1, 0.5, 1.0, "iid"]
        n_clients = [10, 50, 100]
        all_trackers, summaries = [], []
        iid_tracker, noniid_tracker = None, None
        for nc in n_clients:
            for al in alphas:
                t = run_single(args.dataset, nc, args.rounds, al,
                               args.model, args.results, device)
                all_trackers.append(t)
                summaries.append(t.summary())
                if al == "iid" and nc == 10:
                    iid_tracker = t
                if al == 0.1 and nc == 10:
                    noniid_tracker = t

        # Mandatory plots
        plot_accuracy_vs_rounds(
            [t for t in all_trackers if t.num_clients == 10],
            title=f"FedAvg Accuracy — {args.dataset} (10 clients)",
            save_path=f"{args.results}/fedavg_accuracy_vs_rounds.png")
        plot_loss_vs_rounds(
            [t for t in all_trackers if t.num_clients == 10],
            save_path=f"{args.results}/fedavg_loss_vs_rounds.png")
        if iid_tracker and noniid_tracker:
            plot_iid_vs_noniid(iid_tracker, noniid_tracker,
                               save_path=f"{args.results}/fedavg_iid_vs_noniid.png")
        save_summary_table(summaries,
                           save_path=f"{args.results}/fedavg_summary.csv")
    else:
        alpha_val = args.alpha if args.alpha == "iid" else float(args.alpha)
        t = run_single(args.dataset, args.clients, args.rounds, alpha_val,
                       args.model, args.results, device)
        save_summary_table([t.summary()],
                           save_path=f"{args.results}/fedavg_single.csv")

if __name__ == "__main__":
    main()
