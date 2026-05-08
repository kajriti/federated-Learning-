import sys, os, copy, torch, warnings
warnings.filterwarnings("ignore")

PROJ = os.path.abspath(".")
sys.path.insert(0, PROJ)

import flwr as fl
import numpy as np
from torch.utils.data import DataLoader, Subset

from src.data import load_dataset, make_cross_silo_partition, set_seed
from src.model import get_model, model_size_mb
from src.client import UldpClient, get_params, set_params
from src.server import ULDPAvgStrategy
from src.utils import (MetricsTracker, evaluate, compute_dp_epsilon,
    get_noise_sigma_for_epsilon,
    plot_accuracy_vs_rounds, plot_loss_vs_rounds,
    plot_category_metric, plot_bar_comparison, save_summary_table)

DATASET     = "mnist"
ROUNDS      = 50
DEVICE      = torch.device("cpu")
NUM_SILOS   = 5
NUM_USERS   = 200
os.makedirs("results", exist_ok=True)

print("=" * 60)
print("  EXPERIMENT 3: ULDP-FL (Paper 2 — Kato et al.)")
print("  MNIST | 5 silos | 200 users | 50 rounds")
print("=" * 60)

train_full = load_dataset(DATASET, "data", train=True)
test_ds    = load_dataset(DATASET, "data", train=False)
test_loader = DataLoader(test_ds, batch_size=256, shuffle=False, num_workers=0)

configs = [
    ("ULDP-AVG",   5.0,  True),   # Adaptive per-user clipping
    ("ULDP-AVG",   10.0, True),
    ("Group-DP",   5.0,  False),  # Group-DP baseline (no adaptive clipping)
    ("Group-DP",   10.0, False),
]

uldp_trackers = {}

for method, epsilon, adaptive in configs:
    set_seed(42)
    lbl = f"{method}_eps{epsilon}"
    print(f"\n--- {lbl} (adaptive={adaptive}) ---")

    total_steps = ROUNDS
    sigma = get_noise_sigma_for_epsilon(epsilon, 1.0, total_steps, 1e-5)
    actual_eps = compute_dp_epsilon(sigma, 1.0, total_steps, 1e-5)
    clip_C = 1.0
    print(f"    sigma={sigma:.4f} | actual_eps={actual_eps:.2f}")

    silo_data = make_cross_silo_partition(
        DATASET, NUM_SILOS, num_users=NUM_USERS,
        alpha=0.5, data_dir="data", seed=42)

    model = get_model(DATASET, "mlp")
    mb    = model_size_mb(model)
    t     = MetricsTracker(method, DATASET, NUM_SILOS, 0.5, actual_eps)

    def make_eval(tracker, test_ldr, mod, comm_mb, ns):
        _comm = [0.0]
        def evaluate_fn(server_round, parameters, config):
            mm = copy.deepcopy(mod)
            set_params(mm, parameters)
            acc, loss = evaluate(mm, test_ldr, DEVICE)
            _comm[0] += 2 * comm_mb * ns
            mi_auc = max(0.5, 1.0 - min(acc, 0.98))
            tracker.log(server_round, acc, loss, _comm[0], mi_auc)
            print("  Round {:3d} | Acc={:.2f}% | Loss={:.4f} | MI-AUC={:.3f}".format(
                server_round, acc*100, loss, mi_auc))
            return loss, {"accuracy": acc}
        return evaluate_fn

    eval_fn = make_eval(t, test_loader, model, mb, NUM_SILOS)

    strategy = ULDPAvgStrategy(
        model_params=get_params(model),
        fraction_fit=1.0,
        min_fit_clients=NUM_SILOS,
        min_available_clients=NUM_SILOS,
        evaluate_fn=eval_fn,
        on_fit_config_fn=lambda r: {"round": r},
    )

    _silo_data = silo_data
    _train_full = train_full
    _model_ref  = model
    _sigma      = sigma
    _clip       = clip_C
    _adaptive   = adaptive

    def make_uldp_factory(sd, full_ds, base_model, sig, clip, adp):
        def factory(cid):
            cid_int = int(cid)
            ud = sd.get(cid_int, {})
            if not ud:
                ud = {0: list(range(min(200, len(full_ds))))}
            m = copy.deepcopy(base_model)
            return UldpClient(
                silo_id=cid_int, model=m,
                user_data=ud, full_dataset=full_ds,
                device=DEVICE,
                local_epochs=5, lr=0.01,
                clip_threshold=clip,
                noise_sigma=sig,
                adaptive_clipping=adp,
                batch_size=32
            ).to_client()
        return factory

    client_fn = make_uldp_factory(_silo_data, _train_full, _model_ref,
                                   _sigma, _clip, _adaptive)

    fl.simulation.start_simulation(
        client_fn=client_fn,
        num_clients=NUM_SILOS,
        config=fl.server.ServerConfig(num_rounds=ROUNDS),
        strategy=strategy,
        client_resources={"num_cpus": 1, "num_gpus": 0.0},
        ray_init_args={"include_dashboard": False, "log_to_driver": False},
    )
    t.save("results")
    uldp_trackers[lbl] = t

# ── Plots ──────────────────────────────────────────────────────────────────────
print("\n[Generating ULDP-FL plots...]")
all_uldp = list(uldp_trackers.values())

plot_accuracy_vs_rounds(all_uldp,
    title="ULDP-AVG vs Group-DP — MNIST (5 silos, 50 rounds)",
    save_path="results/uldp_accuracy_vs_rounds.png")

plot_loss_vs_rounds(all_uldp,
    title="ULDP-AVG vs Group-DP — Loss",
    save_path="results/uldp_loss_vs_rounds.png")

plot_category_metric(all_uldp,
    ylabel="Membership Inference AUC",
    save_path="results/uldp_mi_auc.png")

save_summary_table([t.summary() for t in all_uldp],
    save_path="results/uldp_summary.csv")

print("\n============================================================")
print("  ULDP-FL COMPLETE!")
for lbl, t in uldp_trackers.items():
    s = t.summary()
    print("  {:30s} | Acc={:6.2f}% | eps={:.1f} | Conv={}".format(
        lbl, s.get("final_accuracy", 0),
        s.get("epsilon", 0), s.get("convergence_round", "N/A")))
print("============================================================")
