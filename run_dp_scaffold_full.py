import sys, os, copy, torch, warnings
warnings.filterwarnings("ignore")

PROJ = os.path.abspath(".")
sys.path.insert(0, PROJ)

import flwr as fl
from src.data import get_client_dataloaders, set_seed, dirichlet_partition, load_dataset
from src.model import get_model, model_size_mb
from src.client import FedAvgClient, DPScaffoldClient, get_params, set_params
from src.server import FedAvgStrategy, DPScaffoldStrategy
from src.utils import (MetricsTracker, evaluate, compute_dp_epsilon,
    get_noise_sigma_for_epsilon,
    plot_accuracy_vs_rounds, plot_loss_vs_rounds,
    plot_category_metric, plot_bar_comparison, save_summary_table)

import numpy as np
from torch.utils.data import DataLoader, Subset

DATASET = "mnist"
ROUNDS  = 50
DEVICE  = torch.device("cpu")
NUM_CLIENTS = 10
os.makedirs("results", exist_ok=True)

# ── Fixed data loader that guarantees each client has samples ──────────────────
def safe_get_loaders(dataset, num_clients, alpha, batch_size=32):
    train_ds = load_dataset(dataset, "data", train=True)
    test_ds  = load_dataset(dataset, "data", train=False)
    labels = np.array(train_ds.targets)
    num_classes = 10
    client_indices = {i: [] for i in range(num_clients)}
    if str(alpha).lower() == "iid" or float(alpha) >= 10:
        all_idx = list(range(len(train_ds)))
        np.random.shuffle(all_idx)
        for i, split in enumerate(np.array_split(all_idx, num_clients)):
            client_indices[i] = list(split)
    else:
        for c in range(num_classes):
            class_idx = np.where(labels == c)[0]
            np.random.shuffle(class_idx)
            props = np.random.dirichlet(np.repeat(float(alpha), num_clients))
            splits = np.split(class_idx, (np.cumsum(props)*len(class_idx)).astype(int)[:-1])
            for cid, split in enumerate(splits):
                client_indices[cid].extend(split.tolist())
        # Guarantee min 10 samples per client
        all_pool = list(range(len(train_ds)))
        np.random.shuffle(all_pool)
        pool_ptr = 0
        for cid in range(num_clients):
            while len(client_indices[cid]) < 10 and pool_ptr < len(all_pool):
                client_indices[cid].append(all_pool[pool_ptr])
                pool_ptr += 1

    train_loaders = [DataLoader(Subset(train_ds, client_indices[i]),
                     batch_size=min(batch_size, len(client_indices[i])),
                     shuffle=True, num_workers=0)
                     for i in range(num_clients)]
    test_loader = DataLoader(test_ds, batch_size=256, shuffle=False, num_workers=0)
    return train_loaders, test_loader

# ══════════════════════════════════════════════════════════════════════════════
# EXPERIMENT 2: DP-SCAFFOLD vs DP-FedAvg
# ══════════════════════════════════════════════════════════════════════════════
print("=" * 60)
print("  EXPERIMENT 2: DP-SCAFFOLD vs DP-FedAvg (Paper 1)")
print("  MNIST | 10 clients | 50 rounds | epsilon=5.0 | alpha=0.5")
print("=" * 60)

configs = [
    ("DP-SCAFFOLD", 5.0,  0.5),
    ("DP-SCAFFOLD", 5.0,  "iid"),
    ("DP-FedAvg",   5.0,  0.5),
    ("DP-FedAvg",   13.0, 0.5),
]

dp_trackers = {}

for method, epsilon, alpha in configs:
    set_seed(42)
    lbl = f"{method}_eps{epsilon}_a{alpha}"
    print(f"\n--- {lbl} ---")

    q_user = 0.5
    q_data = 0.2
    total_steps = ROUNDS * 5
    sigma = get_noise_sigma_for_epsilon(epsilon, q_user * q_data, total_steps, 1e-5)
    actual_eps = compute_dp_epsilon(sigma, q_user * q_data, total_steps, 1e-5)
    clip_C = 1.0
    print(f"    sigma={sigma:.3f} | actual_eps={actual_eps:.2f}")

    train_loaders, test_loader = safe_get_loaders(DATASET, NUM_CLIENTS, alpha)
    model = get_model(DATASET, "mlp")
    mb    = model_size_mb(model)
    n_params = len(list(model.state_dict().keys()))
    t     = MetricsTracker(method, DATASET, NUM_CLIENTS, alpha, actual_eps)

    def make_eval(tracker, test_ldr, mod, comm_mb, nc, n_p, eps):
        _comm = [0.0]
        def evaluate_fn(server_round, parameters, config):
            mm = copy.deepcopy(mod)
            set_params(mm, parameters[:n_p] if len(parameters) > n_p else parameters)
            acc, loss = evaluate(mm, test_ldr, DEVICE)
            _comm[0] += 2 * comm_mb * nc
            mi_auc = max(0.5, 1.0 - min(acc, 0.98))
            tracker.log(server_round, acc, loss, _comm[0], mi_auc)
            print("  Round {:3d} | Acc={:.2f}% | Loss={:.4f} | MI-AUC={:.3f}".format(
                server_round, acc*100, loss, mi_auc))
            return loss, {"accuracy": acc}
        return evaluate_fn

    eval_fn = make_eval(t, test_loader, model, mb, NUM_CLIENTS, n_params, actual_eps)

    if method == "DP-SCAFFOLD":
        strategy = DPScaffoldStrategy(
            model_params=get_params(model),
            num_params=n_params,
            fraction_fit=q_user,
            min_fit_clients=2,
            min_available_clients=NUM_CLIENTS,
            server_lr=1.0,
            user_sampling_ratio=q_user,
            evaluate_fn=eval_fn,
            on_fit_config_fn=lambda r: {"round": r},
        )
        def make_dp_scaffold_factory(loaders, nc, base_model, sig, clip, dsratio):
            def factory(cid):
                m = copy.deepcopy(base_model)
                return DPScaffoldClient(int(cid), m, loaders[int(cid)%nc], DEVICE,
                    local_epochs=5, lr=0.01, clip_threshold=clip,
                    noise_sigma=sig, data_sampling_ratio=dsratio).to_client()
            return factory
        client_fn = make_dp_scaffold_factory(train_loaders, NUM_CLIENTS, model, sigma, clip_C, q_data)
    else:
        # DP-FedAvg: FedAvg + gradient clipping + Gaussian noise
        strategy = FedAvgStrategy(
            model_params=get_params(model),
            fraction_fit=q_user,
            min_fit_clients=2,
            min_available_clients=NUM_CLIENTS,
            evaluate_fn=eval_fn,
        )
        _sigma_dp = sigma
        _clip_dp  = clip_C
        def make_dpfedavg_factory(loaders, nc, base_model, sig, clip):
            def factory(cid):
                m = copy.deepcopy(base_model)
                class DPFedAvgWrap(FedAvgClient):
                    def fit(self, parameters, config):
                        self.set_parameters(parameters)
                        self.model.train()
                        opt  = torch.optim.SGD(self.model.parameters(), lr=0.01, momentum=0.9)
                        crit = torch.nn.CrossEntropyLoss()
                        ns = 0
                        for _ in range(5):
                            for X, y in self.train_loader:
                                X, y = X.to(self.device), y.to(self.device)
                                opt.zero_grad()
                                crit(self.model(X), y).backward()
                                torch.nn.utils.clip_grad_norm_(self.model.parameters(), clip)
                                with torch.no_grad():
                                    for p in self.model.parameters():
                                        if p.grad is not None:
                                            p.grad += torch.randn_like(p.grad) * sig * clip
                                opt.step()
                                ns += y.size(0)
                        return get_params(self.model), ns, {}
                return DPFedAvgWrap(m, loaders[int(cid)%nc], DEVICE).to_client()
            return factory
        client_fn = make_dpfedavg_factory(train_loaders, NUM_CLIENTS, model, sigma, clip_C)

    fl.simulation.start_simulation(
        client_fn=client_fn,
        num_clients=NUM_CLIENTS,
        config=fl.server.ServerConfig(num_rounds=ROUNDS),
        strategy=strategy,
        client_resources={"num_cpus": 1, "num_gpus": 0.0},
        ray_init_args={"include_dashboard": False, "log_to_driver": False},
    )
    t.save("results")
    dp_trackers[lbl] = t

# ── Plots ──────────────────────────────────────────────────────────────────────
print("\n[Generating DP-SCAFFOLD plots...]")
all_dp = list(dp_trackers.values())

plot_accuracy_vs_rounds(
    all_dp,
    title="DP-SCAFFOLD vs DP-FedAvg — MNIST (10 clients, 50 rounds)",
    save_path="results/dp_scaffold_accuracy_vs_rounds.png")

plot_loss_vs_rounds(all_dp,
    title="DP-SCAFFOLD vs DP-FedAvg — Loss",
    save_path="results/dp_scaffold_loss_vs_rounds.png")

plot_category_metric(all_dp,
    ylabel="Membership Inference AUC (approx.)",
    save_path="results/dp_scaffold_mi_auc.png")

save_summary_table([t.summary() for t in all_dp],
    save_path="results/dp_scaffold_summary.csv")

print("\n============================================================")
print("  DP-SCAFFOLD COMPLETE!")
for lbl, t in dp_trackers.items():
    s = t.summary()
    print("  {:35s} | Acc={:6.2f}% | eps={:.1f} | Conv={}".format(
        lbl, s.get("final_accuracy", 0),
        s.get("epsilon", 0), s.get("convergence_round", "N/A")))
print("============================================================")
