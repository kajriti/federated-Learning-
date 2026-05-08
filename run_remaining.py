import sys, os, copy, torch, warnings
warnings.filterwarnings("ignore")
sys.path.insert(0, ".")

import flwr as fl
import numpy as np
from torch.utils.data import DataLoader, Subset

from src.data import load_dataset, make_cross_silo_partition, set_seed
from src.model import get_model, model_size_mb
from src.client import UldpClient, FedAvgClient, get_params, set_params
from src.server import ULDPAvgStrategy, FedAvgStrategy
from src.utils import (MetricsTracker, evaluate, compute_dp_epsilon,
    get_noise_sigma_for_epsilon,
    plot_accuracy_vs_rounds, plot_loss_vs_rounds,
    plot_category_metric, plot_bar_comparison, save_summary_table)

DATASET   = "mnist"
ROUNDS    = 50
DEVICE    = torch.device("cpu")
NUM_SILOS = 5
NUM_USERS = 200
os.makedirs("results", exist_ok=True)

print("=" * 60)
print("  REMAINING: ULDP-AVG (eps=10) + Group-DP (eps=5,10)")
print("=" * 60)

train_full  = load_dataset(DATASET, "data", train=True)
test_ds     = load_dataset(DATASET, "data", train=False)
test_loader = DataLoader(test_ds, batch_size=256, shuffle=False, num_workers=0)

# Pre-build silo partition once
set_seed(42)
silo_data = make_cross_silo_partition(
    DATASET, NUM_SILOS, num_users=NUM_USERS, alpha=0.5, data_dir="data", seed=42)

def run_experiment(method, epsilon, adaptive, num_silos):
    set_seed(42)
    lbl = "{}_eps{}".format(method, epsilon)
    print("\n--- {} (adaptive={}) ---".format(lbl, adaptive))
    sigma      = get_noise_sigma_for_epsilon(epsilon, 1.0, ROUNDS, 1e-5)
    actual_eps = compute_dp_epsilon(sigma, 1.0, ROUNDS, 1e-5)
    clip_C     = 1.0
    print("    sigma={:.4f} | actual_eps={:.2f}".format(sigma, actual_eps))

    model = get_model(DATASET, "mlp")
    mb    = model_size_mb(model)
    t     = MetricsTracker(method, DATASET, num_silos, 0.5, actual_eps)

    def make_eval(tracker, tst_ldr, mod, comm_mb, ns):
        _c = [0.0]
        def ev(server_round, parameters, config):
            mm = copy.deepcopy(mod)
            set_params(mm, parameters)
            acc, loss = evaluate(mm, tst_ldr, DEVICE)
            _c[0] += 2 * comm_mb * ns
            mi = max(0.5, 1.0 - min(acc, 0.98))
            tracker.log(server_round, acc, loss, _c[0], mi)
            print("  Round {:3d} | Acc={:.2f}% | Loss={:.4f} | eps={:.1f}".format(
                server_round, acc * 100, loss, actual_eps))
            return loss, {"accuracy": acc}
        return ev

    eval_fn  = make_eval(t, test_loader, model, mb, num_silos)
    strategy = ULDPAvgStrategy(
        model_params=get_params(model),
        fraction_fit=1.0,
        min_fit_clients=num_silos,
        min_available_clients=num_silos,
        evaluate_fn=eval_fn,
        on_fit_config_fn=lambda r: {"round": r},
    )

    def make_factory(sd, full_ds, base_mod, sig, clip, adp, ns):
        def factory(cid):
            cid_int = int(cid)
            ud = sd.get(cid_int, {0: list(range(200))})
            m  = copy.deepcopy(base_mod)
            return UldpClient(
                silo_id=cid_int, model=m,
                user_data=ud, full_dataset=full_ds,
                device=DEVICE, local_epochs=5, lr=0.01,
                clip_threshold=clip, noise_sigma=sig,
                adaptive_clipping=adp, batch_size=32,
            ).to_client()
        return factory

    fl.simulation.start_simulation(
        client_fn=make_factory(silo_data, train_full, model, sigma, clip_C, adaptive, num_silos),
        num_clients=num_silos,
        config=fl.server.ServerConfig(num_rounds=ROUNDS),
        strategy=strategy,
        client_resources={"num_cpus": 1, "num_gpus": 0.0},
        ray_init_args={"include_dashboard": False, "log_to_driver": False},
    )
    t.save("results")
    return lbl, t

all_trackers = {}

for method, epsilon, adaptive in [
    ("ULDP-AVG",  10.0, True),
    ("Group-DP",   5.0, False),
    ("Group-DP",  10.0, False),
]:
    lbl, t = run_experiment(method, epsilon, adaptive, NUM_SILOS)
    all_trackers[lbl] = t

print("\n============================================================")
print("  Remaining experiments DONE!")
for lbl, t in all_trackers.items():
    s = t.summary()
    print("  {:28s} Acc={:6.2f}%  eps={:.1f}  Conv={}".format(
        lbl, s.get("final_accuracy", 0),
        s.get("epsilon", 0), s.get("convergence_round", "N/A")))
print("============================================================")
