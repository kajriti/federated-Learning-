import sys, os, copy, torch, warnings
warnings.filterwarnings("ignore")

PROJ = os.path.abspath(".")
sys.path.insert(0, PROJ)

import flwr as fl
from src.data import get_client_dataloaders, set_seed
from src.model import get_model, model_size_mb
from src.client import FedAvgClient, get_params, set_params, make_client_fn
from src.server import FedAvgStrategy
from src.utils import (MetricsTracker, evaluate,
    plot_accuracy_vs_rounds, plot_loss_vs_rounds,
    plot_iid_vs_noniid, save_summary_table)

DATASET = "mnist"
ROUNDS  = 50
DEVICE  = torch.device("cpu")

print("=" * 60)
print("  EXPERIMENT 1: FedAvg Baseline")
print("  Dataset: MNIST | Clients: 10 | Rounds: 50")
print("  Alphas: IID, 0.5, 0.1 (Non-IID)")
print("=" * 60)

os.makedirs("results", exist_ok=True)
os.makedirs("data",    exist_ok=True)

alphas_to_run = ["iid", 0.5, 0.1, 0.01]
trackers = {}

for alpha in alphas_to_run:
    set_seed(42)
    num_clients = 10
    tag = f"FedAvg_MNIST_C{num_clients}_a{alpha}"
    print(f"\n--- Running: {tag} ---")

    train_loaders, test_loader = get_client_dataloaders(
        DATASET, num_clients, alpha, batch_size=32, data_dir="data")

    model = get_model(DATASET, "mlp")
    mb    = model_size_mb(model)
    t     = MetricsTracker("FedAvg", DATASET, num_clients, alpha)
    comm  = 0.0

    _model_ref   = model
    _test_loader = test_loader
    _tracker     = t

    def make_eval(tracker, test_ldr, mod, comm_mb, nc):
        _comm = [0.0]
        def evaluate_fn(server_round, parameters, config):
            mm = copy.deepcopy(mod)
            set_params(mm, parameters)
            acc, loss = evaluate(mm, test_ldr, DEVICE)
            _comm[0] += 2 * comm_mb * nc
            tracker.log(server_round, acc, loss, _comm[0], 1.0 - acc)
            print("  Round {:3d} | Acc={:.2f}% | Loss={:.4f}".format(
                server_round, acc*100, loss))
            return loss, {"accuracy": acc}
        return evaluate_fn

    eval_fn = make_eval(t, test_loader, model, mb, num_clients)

    strategy = FedAvgStrategy(
        model_params=get_params(model),
        fraction_fit=0.5,
        min_fit_clients=2,
        min_available_clients=num_clients,
        evaluate_fn=eval_fn,
        on_fit_config_fn=lambda r: {"round": r},
    )

    _loaders = train_loaders
    _nc      = num_clients
    _mod_ref = model

    def make_client_factory(loaders, nc, base_model):
        def factory(cid):
            m = copy.deepcopy(base_model)
            return FedAvgClient(m, loaders[int(cid) % nc], DEVICE,
                                local_epochs=5, lr=0.01, momentum=0.9).to_client()
        return factory

    fl.simulation.start_simulation(
        client_fn=make_client_factory(_loaders, _nc, _mod_ref),
        num_clients=num_clients,
        config=fl.server.ServerConfig(num_rounds=ROUNDS),
        strategy=strategy,
        client_resources={"num_cpus": 1, "num_gpus": 0.0},
        ray_init_args={"include_dashboard": False, "log_to_driver": False},
    )

    t.save("results")
    trackers[str(alpha)] = t

# ── Generate mandatory plots ──────────────────────────────────────────────────
print("\n[Generating plots...]")

all_list = list(trackers.values())
plot_accuracy_vs_rounds(
    all_list,
    title="FedAvg — Global Accuracy vs Rounds (MNIST, 10 clients)",
    save_path="results/fedavg_accuracy_vs_rounds.png")

plot_loss_vs_rounds(
    all_list,
    title="FedAvg — Global Loss vs Rounds (MNIST, 10 clients)",
    save_path="results/fedavg_loss_vs_rounds.png")

if "iid" in trackers and "0.1" in trackers:
    plot_iid_vs_noniid(
        trackers["iid"], trackers["0.1"],
        save_path="results/fedavg_iid_vs_noniid.png")

save_summary_table(
    [t.summary() for t in all_list],
    save_path="results/fedavg_summary.csv")

print("\n============================================================")
print("  FedAvg COMPLETE!")
for alpha, t in trackers.items():
    s = t.summary()
    print("  alpha={:5s} | Acc={:6.2f}% | Conv.Round={} | Comm={:.1f}MB".format(
        str(alpha), s.get("final_accuracy", 0),
        s.get("convergence_round", "N/A"),
        s.get("comm_cost_mb", 0)))
print("============================================================")
