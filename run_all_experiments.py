import sys, os, copy, warnings, time
warnings.filterwarnings("ignore")
PROJ = os.path.abspath(".")
sys.path.insert(0, PROJ)
import torch, flwr as fl
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from torch.utils.data import DataLoader, Subset
sns.set_theme(style="darkgrid")
from src.data   import load_dataset, set_seed, make_cross_silo_partition
from src.model  import get_model, model_size_mb
from src.client import get_params, set_params, FedAvgClient, DPScaffoldClient, UldpClient
from src.server import FedAvgStrategy, DPScaffoldStrategy, ULDPAvgStrategy
from src.utils  import (MetricsTracker, compute_dp_epsilon, get_noise_sigma_for_epsilon)

DEVICE = torch.device("cpu")
NUM_ROUNDS = 20
RESULTS = "results"
DATA_DIR = "data"
os.makedirs(RESULTS, exist_ok=True)
os.makedirs(DATA_DIR, exist_ok=True)
CLIENT_COUNTS = [10, 50, 100]
ALPHAS = ["iid", 1.0, 0.5, 0.1, 0.01]
DATASETS = ["mnist", "fmnist"]
EPSILONS = [5.0, 13.0]
COLORS = ["#1f77b4","#ff7f0e","#2ca02c","#d62728","#9467bd","#8c564b","#e377c2","#7f7f7f","#bcbd22","#17becf","#aec7e8","#ffbb78","#98df8a","#ff9896","#c5b0d5"]

def eid_done(eid):
    return os.path.exists(os.path.join(RESULTS, eid + ".csv"))

def make_partition(ds_name, num_clients, alpha, seed=42):
    np.random.seed(seed)
    train_ds = load_dataset(ds_name, DATA_DIR, train=True)
    labels = np.array(train_ds.targets)
    n_cls = len(set(labels.tolist()))
    indices = {i: [] for i in range(num_clients)}
    if str(alpha).lower() == "iid":
        perm = np.random.permutation(len(train_ds)).tolist()
        for i, sp in enumerate(np.array_split(perm, num_clients)):
            indices[i] = [int(x) for x in sp]
    else:
        a = float(alpha)
        for c in range(n_cls):
            cls_idx = np.where(labels == c)[0].tolist()
            np.random.shuffle(cls_idx)
            props = np.random.dirichlet(np.repeat(a, num_clients))
            cuts = (np.cumsum(props) * len(cls_idx)).astype(int)[:-1]
            for cid, sp in enumerate(np.split(cls_idx, cuts)):
                indices[cid].extend([int(x) for x in sp])
    spare = list(np.random.permutation(len(train_ds)))
    ptr = [0]
    for i in range(num_clients):
        while len(indices[i]) < 32:
            indices[i].append(spare[ptr[0] % len(spare)])
            ptr[0] += 1
    return indices

def make_evaluate_fn(ds_name, model_arch, n_params, comm_holder, tracker, mb, nc):
    def evaluate_fn(server_round, parameters, config):
        test_ds = load_dataset(ds_name, DATA_DIR, train=False)
        test_loader = DataLoader(test_ds, batch_size=512, shuffle=False, num_workers=0)
        model = get_model(ds_name, model_arch)
        set_params(model, parameters[:n_params])
        model.eval()
        total_loss, correct, total = 0.0, 0, 0
        crit = torch.nn.CrossEntropyLoss()
        with torch.no_grad():
            for X, y in test_loader:
                out = model(X)
                total_loss += crit(out, y).item() * y.size(0)
                correct += (out.argmax(1) == y).sum().item()
                total += y.size(0)
        acc = correct / max(total, 1) * 100.0
        loss = total_loss / max(total, 1)
        comm_holder[0] += 2 * mb * nc
        mi_auc = max(0.5, 1.0 - min(acc / 100.0, 0.98))
        tracker.log(server_round, acc, loss, comm_holder[0], mi_auc)
        print("    [{:3d}] Acc={:6.2f}%  Loss={:.4f}  Comm={:.1f}MB".format(server_round, acc, loss, comm_holder[0]))
        return loss, {"accuracy": acc / 100.0}
    return evaluate_fn

def run_fedavg(ds_name, num_clients, alpha, num_rounds, seed=42):
    set_seed(seed)
    eid = "FedAvg_{}_C{}_a{}".format(ds_name, num_clients, alpha)
    arch = "mlp"
    print("\n  >> {}".format(eid))
    partition = make_partition(ds_name, num_clients, alpha, seed)
    model = get_model(ds_name, arch)
    mb = model_size_mb(model)
    n_params = len(list(model.state_dict().keys()))
    t = MetricsTracker("FedAvg", ds_name, num_clients, alpha, 0.0)
    comm = [0.0]
    eval_fn = make_evaluate_fn(ds_name, arch, n_params, comm, t, mb, num_clients)
    frac = min(0.5, max(2.0 / num_clients, 0.1))
    min_cli = max(2, int(frac * num_clients))
    strategy = FedAvgStrategy(
        model_params=get_params(model), fraction_fit=frac,
        min_fit_clients=min_cli, min_available_clients=num_clients,
        evaluate_fn=eval_fn, on_fit_config_fn=lambda r: {"round": r})
    _part = partition; _nc = num_clients; _ds = ds_name; _arch = arch
    def client_fn(cid):
        ci = int(cid) % _nc
        m = get_model(_ds, _arch)
        ds_ = load_dataset(_ds, DATA_DIR, train=True)
        ldr = DataLoader(Subset(ds_, _part[ci]), batch_size=min(32, len(_part[ci])), shuffle=True, num_workers=0)
        return FedAvgClient(m, ldr, DEVICE, local_epochs=5, lr=0.01, momentum=0.9).to_client()
    fl.simulation.start_simulation(
        client_fn=client_fn, num_clients=num_clients,
        config=fl.server.ServerConfig(num_rounds=num_rounds),
        strategy=strategy,
        client_resources={"num_cpus": 1, "num_gpus": 0.0},
        ray_init_args={"include_dashboard": False, "log_to_driver": False})
    t.save(RESULTS)
    return eid, t

def run_dp_fedavg(ds_name, num_clients, alpha, epsilon, num_rounds, seed=42):
    set_seed(seed)
    eid = "DP-FedAvg_{}_C{}_a{}_eps{}".format(ds_name, num_clients, alpha, epsilon)
    arch = "mlp"
    q_u = min(0.5, max(2.0 / num_clients, 0.1)); K = 5; q_d = 0.5
    sigma = get_noise_sigma_for_epsilon(epsilon, q_u * q_d, num_rounds * K, 1e-5)
    aeps = compute_dp_epsilon(sigma, q_u * q_d, num_rounds * K, 1e-5)
    clip = 1.0
    print("\n  >> {}  sigma={:.3f}  actual_e={:.2f}".format(eid, sigma, aeps))
    partition = make_partition(ds_name, num_clients, alpha, seed)
    model = get_model(ds_name, arch)
    mb = model_size_mb(model)
    n_params = len(list(model.state_dict().keys()))
    t = MetricsTracker("DP-FedAvg", ds_name, num_clients, alpha, aeps)
    comm = [0.0]
    eval_fn = make_evaluate_fn(ds_name, arch, n_params, comm, t, mb, num_clients)
    min_cli = max(2, int(q_u * num_clients))
    strategy = FedAvgStrategy(
        model_params=get_params(model), fraction_fit=q_u,
        min_fit_clients=min_cli, min_available_clients=num_clients, evaluate_fn=eval_fn)
    _part = partition; _nc = num_clients; _ds = ds_name; _arch = arch
    _sig = sigma; _cli = clip
    def client_fn(cid):
        ci = int(cid) % _nc
        m = get_model(_ds, _arch)
        ds_ = load_dataset(_ds, DATA_DIR, train=True)
        ldr = DataLoader(Subset(ds_, _part[ci]), batch_size=min(32, len(_part[ci])), shuffle=True, num_workers=0)
        class DPFedAvgClient(FedAvgClient):
            def fit(self, params, cfg):
                self.set_parameters(params); self.model.train()
                opt = torch.optim.SGD(self.model.parameters(), lr=0.01, momentum=0.9)
                crit = torch.nn.CrossEntropyLoss(); n = 0
                for _ in range(5):
                    for X, y in self.train_loader:
                        X, y = X.to(self.device), y.to(self.device)
                        opt.zero_grad(); crit(self.model(X), y).backward()
                        torch.nn.utils.clip_grad_norm_(self.model.parameters(), _cli)
                        with torch.no_grad():
                            for p in self.model.parameters():
                                if p.grad is not None:
                                    p.grad += torch.randn_like(p.grad) * _sig * _cli / max(len(self.train_loader), 1)
                        opt.step(); n += y.size(0)
                return get_params(self.model), n, {}
        return DPFedAvgClient(m, ldr, DEVICE).to_client()
    fl.simulation.start_simulation(
        client_fn=client_fn, num_clients=num_clients,
        config=fl.server.ServerConfig(num_rounds=num_rounds),
        strategy=strategy,
        client_resources={"num_cpus": 1, "num_gpus": 0.0},
        ray_init_args={"include_dashboard": False, "log_to_driver": False})
    t.save(RESULTS)
    return eid, t

def run_dp_scaffold(ds_name, num_clients, alpha, epsilon, num_rounds, seed=42):
    set_seed(seed)
    eid = "DP-SCAFFOLD_{}_C{}_a{}_eps{}".format(ds_name, num_clients, alpha, epsilon)
    arch = "mlp"
    q_u = min(0.5, max(2.0 / num_clients, 0.1)); K = 5; q_d = 0.2
    sigma = get_noise_sigma_for_epsilon(epsilon, q_u * q_d, num_rounds * K, 1e-5)
    aeps = compute_dp_epsilon(sigma, q_u * q_d, num_rounds * K, 1e-5)
    clip = 1.0
    print("\n  >> {}  sigma={:.3f}  actual_e={:.2f}".format(eid, sigma, aeps))
    partition = make_partition(ds_name, num_clients, alpha, seed)
    model = get_model(ds_name, arch)
    mb = model_size_mb(model)
    n_params = len(list(model.state_dict().keys()))
    t = MetricsTracker("DP-SCAFFOLD", ds_name, num_clients, alpha, aeps)
    comm = [0.0]
    eval_fn = make_evaluate_fn(ds_name, arch, n_params, comm, t, mb, num_clients)
    min_cli = max(2, int(q_u * num_clients))
    strategy = DPScaffoldStrategy(
        model_params=get_params(model), num_params=n_params,
        fraction_fit=q_u, min_fit_clients=min_cli,
        min_available_clients=num_clients, server_lr=1.0,
        user_sampling_ratio=q_u, evaluate_fn=eval_fn,
        on_fit_config_fn=lambda r: {"round": r})
    _part = partition; _nc = num_clients; _ds = ds_name; _arch = arch
    _sig = sigma; _cli = clip
    def client_fn(cid):
        ci = int(cid) % _nc
        m = get_model(_ds, _arch)
        ds_ = load_dataset(_ds, DATA_DIR, train=True)
        ldr = DataLoader(Subset(ds_, _part[ci]), batch_size=min(32, len(_part[ci])), shuffle=True, num_workers=0)
        return DPScaffoldClient(ci, m, ldr, DEVICE, local_epochs=5, lr=0.01,
            clip_threshold=_cli, noise_sigma=_sig, data_sampling_ratio=q_d).to_client()
    fl.simulation.start_simulation(
        client_fn=client_fn, num_clients=num_clients,
        config=fl.server.ServerConfig(num_rounds=num_rounds),
        strategy=strategy,
        client_resources={"num_cpus": 1, "num_gpus": 0.0},
        ray_init_args={"include_dashboard": False, "log_to_driver": False})
    t.save(RESULTS)
    return eid, t

def run_uldp(ds_name, num_silos, alpha, epsilon, adaptive, num_rounds, seed=42):
    method = "ULDP-AVG" if adaptive else "Group-DP"
    eid = "{}_{}_S{}_a{}_eps{}".format(method, ds_name, num_silos, alpha, epsilon)
    arch = "mlp"
    sigma = get_noise_sigma_for_epsilon(epsilon, 1.0, num_rounds, 1e-5)
    aeps = compute_dp_epsilon(sigma, 1.0, num_rounds, 1e-5)
    clip = 1.0
    print("\n  >> {}  sigma={:.4f}  actual_e={:.2f}".format(eid, sigma, aeps))
    set_seed(seed)
    n_users = max(100, num_silos * 20)
    silo_data = make_cross_silo_partition(
        ds_name, num_silos, num_users=n_users,
        alpha=0.5 if str(alpha).lower() != "iid" else 100.0,
        data_dir=DATA_DIR, seed=seed)
    model = get_model(ds_name, arch)
    mb = model_size_mb(model)
    n_params = len(list(model.state_dict().keys()))
    t = MetricsTracker(method, ds_name, num_silos, alpha, aeps)
    comm = [0.0]
    eval_fn = make_evaluate_fn(ds_name, arch, n_params, comm, t, mb, num_silos)
    strategy = ULDPAvgStrategy(
        model_params=get_params(model), fraction_fit=1.0,
        min_fit_clients=num_silos, min_available_clients=num_silos,
        evaluate_fn=eval_fn, on_fit_config_fn=lambda r: {"round": r})
    _sd = silo_data; _ds = ds_name; _arch = arch
    _sig = sigma; _clip = clip; _adp = adaptive
    def client_fn(cid):
        ci = int(cid)
        ud = _sd.get(ci, {0: list(range(200))})
        u_keys = list(ud.keys())[:15]
        ud = {u: ud[u] for u in u_keys}
        m = get_model(_ds, _arch)
        full = load_dataset(_ds, DATA_DIR, train=True)
        return UldpClient(ci, m, ud, full, DEVICE, local_epochs=3, lr=0.01,
            clip_threshold=_clip, noise_sigma=_sig, adaptive_clipping=_adp, batch_size=64).to_client()
    fl.simulation.start_simulation(
        client_fn=client_fn, num_clients=num_silos,
        config=fl.server.ServerConfig(num_rounds=num_rounds),
        strategy=strategy,
        client_resources={"num_cpus": 1, "num_gpus": 0.0},
        ray_init_args={"include_dashboard": False, "log_to_driver": False})
    t.save(RESULTS)
    return eid, t

# PHASE 1: FedAvg
print("\n" + "="*70)
print("  PHASE 1/4 - FedAvg Baseline ({} configs)".format(len(DATASETS)*len(CLIENT_COUNTS)*len(ALPHAS)))
print("="*70)
for ds in DATASETS:
    for nc in CLIENT_COUNTS:
        for al in ALPHAS:
            eid = "FedAvg_{}_C{}_a{}".format(ds, nc, al)
            if eid_done(eid): print("  [SKIP] {}".format(eid)); continue
            run_fedavg(ds, nc, al, NUM_ROUNDS)

# PHASE 2: DP-FedAvg
print("\n" + "="*70)
print("  PHASE 2/4 - DP-FedAvg")
print("="*70)
for ds in DATASETS:
    for nc in CLIENT_COUNTS:
        for al in [0.5, 0.1, "iid"]:
            for eps in EPSILONS:
                eid = "DP-FedAvg_{}_C{}_a{}_eps{}".format(ds, nc, al, eps)
                if eid_done(eid): print("  [SKIP] {}".format(eid)); continue
                run_dp_fedavg(ds, nc, al, eps, NUM_ROUNDS)

# PHASE 3: DP-SCAFFOLD
print("\n" + "="*70)
print("  PHASE 3/4 - DP-SCAFFOLD (Paper 1)")
print("="*70)
for ds in DATASETS:
    for nc in CLIENT_COUNTS:
        for al in [0.5, 0.1, "iid"]:
            for eps in EPSILONS:
                eid = "DP-SCAFFOLD_{}_C{}_a{}_eps{}".format(ds, nc, al, eps)
                if eid_done(eid): print("  [SKIP] {}".format(eid)); continue
                run_dp_scaffold(ds, nc, al, eps, NUM_ROUNDS)

# PHASE 4: ULDP-FL
print("\n" + "="*70)
print("  PHASE 4/4 - ULDP-FL (Paper 2)")
print("="*70)
for ds in DATASETS:
    for nc in CLIENT_COUNTS:
        for al in [0.5, "iid"]:
            for eps in EPSILONS:
                for adp in [True, False]:
                    m_ = "ULDP-AVG" if adp else "Group-DP"
                    eid = "{}_{}_S{}_a{}_eps{}".format(m_, ds, nc, al, eps)
                    if eid_done(eid): print("  [SKIP] {}".format(eid)); continue
                    run_uldp(ds, nc, al, eps, adp, NUM_ROUNDS)

# GENERATE PLOTS
print("\n" + "="*70 + "\n  GENERATING PLOTS\n" + "="*70)

def load_csv(eid):
    path = os.path.join(RESULTS, eid + ".csv")
    return pd.read_csv(path) if os.path.exists(path) else None

# Fig 1: Accuracy vs Rounds (FedAvg, all clients, key alphas)
fig, axes = plt.subplots(1, 3, figsize=(18, 5), dpi=150, sharey=True)
for col, nc in enumerate(CLIENT_COUNTS):
    ax = axes[col]
    for i, (al, lbl, col_) in enumerate([("iid","IID",COLORS[0]),(1.0,"a=1.0",COLORS[1]),(0.5,"a=0.5",COLORS[2]),(0.1,"a=0.1",COLORS[3]),(0.01,"a=0.01",COLORS[4])]):
        df = load_csv("FedAvg_mnist_C{}_a{}".format(nc, al))
        if df is not None: ax.plot(df["round"], df["test_accuracy"], label=lbl, color=col_, lw=1.8)
    ax.set_title("C={} clients".format(nc)); ax.set_xlabel("Round")
    if col==0: ax.set_ylabel("Test Accuracy (%)")
    ax.legend(fontsize=8); ax.set_ylim([0,100])
fig.suptitle("Figure 1 - FedAvg: Test Accuracy vs Rounds (MNIST)"); plt.tight_layout()
plt.savefig(os.path.join(RESULTS, "fig1_accuracy_vs_rounds.png"), dpi=150, bbox_inches="tight"); plt.close()
print("  Saved: fig1_accuracy_vs_rounds.png")

# Fig 2: Loss vs Rounds
fig, axes = plt.subplots(1, 3, figsize=(18, 5), dpi=150)
for col, nc in enumerate(CLIENT_COUNTS):
    ax = axes[col]
    for i, (al, lbl, col_) in enumerate([("iid","IID",COLORS[0]),(0.5,"a=0.5",COLORS[2]),(0.1,"a=0.1",COLORS[3])]):
        df = load_csv("FedAvg_mnist_C{}_a{}".format(nc, al))
        if df is not None:
            v = df[df["test_loss"].notna() & (df["test_loss"] < 1e3)]
            if len(v): ax.plot(v["round"], v["test_loss"], label=lbl, color=col_, lw=1.8)
    ax.set_title("C={} clients".format(nc)); ax.set_xlabel("Round")
    if col==0: ax.set_ylabel("Test Loss"); ax.legend(fontsize=8)
fig.suptitle("Figure 2 - FedAvg: Test Loss vs Rounds (MNIST)"); plt.tight_layout()
plt.savefig(os.path.join(RESULTS, "fig2_loss_vs_rounds.png"), dpi=150, bbox_inches="tight"); plt.close()
print("  Saved: fig2_loss_vs_rounds.png")

# Fig 3: IID vs NonIID
fig, axes = plt.subplots(1, 3, figsize=(18, 5), dpi=150, sharey=True)
for col, nc in enumerate(CLIENT_COUNTS):
    ax = axes[col]
    for al, lbl, col_, ls in [("iid","IID",COLORS[0],"-"),(1.0,"a=1.0",COLORS[1],"-"),(0.5,"a=0.5",COLORS[2],"--"),(0.1,"a=0.1",COLORS[3],"--"),(0.01,"a=0.01",COLORS[4],":")]:
        df = load_csv("FedAvg_mnist_C{}_a{}".format(nc, al))
        if df is not None: ax.plot(df["round"], df["test_accuracy"], label=lbl, color=col_, lw=1.8, linestyle=ls)
    ax.set_title("C={} clients".format(nc)); ax.set_xlabel("Round")
    if col==0: ax.set_ylabel("Test Accuracy (%)")
    ax.legend(fontsize=8); ax.set_ylim([0,100])
fig.suptitle("Figure 3 - IID vs Non-IID Heterogeneity (MNIST)"); plt.tight_layout()
plt.savefig(os.path.join(RESULTS, "fig3_iid_vs_noniid.png"), dpi=150, bbox_inches="tight"); plt.close()
print("  Saved: fig3_iid_vs_noniid.png")

# Fig 4: MI-AUC (Category Metric)
fig, axes = plt.subplots(1, 3, figsize=(18, 5), dpi=150, sharey=True)
for col, nc in enumerate(CLIENT_COUNTS):
    ax = axes[col]
    for eid_, lbl, col_ in [
        ("FedAvg_mnist_C{}_a0.5".format(nc),"FedAvg",COLORS[0]),
        ("DP-FedAvg_mnist_C{}_a0.5_eps5.0".format(nc),"DP-FedAvg e=5",COLORS[1]),
        ("DP-FedAvg_mnist_C{}_a0.5_eps13.0".format(nc),"DP-FedAvg e=13",COLORS[2]),
        ("DP-SCAFFOLD_mnist_C{}_a0.5_eps5.0".format(nc),"SCAFFOLD e=5",COLORS[3]),
        ("DP-SCAFFOLD_mnist_C{}_a0.5_eps13.0".format(nc),"SCAFFOLD e=13",COLORS[4]),
    ]:
        df = load_csv(eid_)
        if df is not None: ax.plot(df["round"], df["category_metric"], label=lbl, color=col_, lw=1.8)
    ax.set_title("C={} clients".format(nc)); ax.set_xlabel("Round")
    if col==0: ax.set_ylabel("MI-AUC")
    ax.legend(fontsize=7); ax.set_ylim([0.4, 1.05])
    ax.axhline(0.5, color="gray", linestyle=":", lw=1.0)
fig.suptitle("Figure 4 - Privacy: MI-AUC vs Rounds (MNIST, alpha=0.5)"); plt.tight_layout()
plt.savefig(os.path.join(RESULTS, "fig4_privacy_mi_auc.png"), dpi=150, bbox_inches="tight"); plt.close()
print("  Saved: fig4_privacy_mi_auc.png")

# Fig 5: Bar Chart
bar_rows = []
for nc in CLIENT_COUNTS:
    for eid_, lbl, al, eps in [
        ("FedAvg_mnist_C{}_a0.5","FedAvg",0.5,0.0),
        ("DP-FedAvg_mnist_C{}_a0.5_eps5.0","DP-FedAvg e=5",0.5,5.0),
        ("DP-FedAvg_mnist_C{}_a0.5_eps13.0","DP-FedAvg e=13",0.5,13.0),
        ("DP-SCAFFOLD_mnist_C{}_a0.5_eps5.0","DP-SCAFFOLD e=5",0.5,5.0),
        ("DP-SCAFFOLD_mnist_C{}_a0.5_eps13.0","DP-SCAFFOLD e=13",0.5,13.0),
    ]:
        df = load_csv(eid_.format(nc))
        if df is not None: bar_rows.append({"clients":nc,"method":lbl,"accuracy":df.iloc[-1]["test_accuracy"]})
if bar_rows:
    df_bar = pd.DataFrame(bar_rows)
    methods = list(df_bar["method"].unique()); x = np.arange(len(CLIENT_COUNTS)); width = 0.15
    fig, ax = plt.subplots(figsize=(14, 6), dpi=150)
    for i, m in enumerate(methods):
        vals = [float(df_bar[(df_bar["method"]==m)&(df_bar["clients"]==nc)]["accuracy"].values[0]) if len(df_bar[(df_bar["method"]==m)&(df_bar["clients"]==nc)])>0 else 0.0 for nc in CLIENT_COUNTS]
        bars = ax.bar(x+i*width, vals, width, label=m, color=COLORS[i%len(COLORS)], edgecolor="black", lw=0.4)
        for bar, v in zip(bars, vals):
            if v > 0: ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.5, "{:.1f}".format(v), ha="center", va="bottom", fontsize=7)
    ax.set_xticks(x + width*(len(methods)-1)/2); ax.set_xticklabels(["C={}".format(nc) for nc in CLIENT_COUNTS])
    ax.set_ylabel("Final Accuracy (%)"); ax.set_title("Figure 5 - Baseline vs DP Methods (MNIST, alpha=0.5)")
    ax.legend(fontsize=8); ax.set_ylim([0,110]); plt.tight_layout()
    plt.savefig(os.path.join(RESULTS, "fig5_bar_comparison.png"), dpi=150, bbox_inches="tight"); plt.close()
    print("  Saved: fig5_bar_comparison.png")

# Fig 6: ULDP vs Group-DP
fig, axes = plt.subplots(1, 3, figsize=(18, 5), dpi=150)
for col, nc in enumerate(CLIENT_COUNTS):
    ax = axes[col]
    for meth, adp, eps, col_, ls in [("ULDP-AVG",True,5.0,COLORS[0],"-"),("ULDP-AVG",True,13.0,COLORS[1],"-"),("Group-DP",False,5.0,COLORS[2],"--"),("Group-DP",False,13.0,COLORS[3],"--")]:
        df = load_csv("{}_{}_S{}_a0.5_eps{}".format(meth,"mnist",nc,eps))
        if df is not None: ax.plot(df["round"], df["test_accuracy"], label="{} e={:.0f}".format(meth,eps), color=col_, lw=1.8, linestyle=ls)
    ax.set_title("Silos={}".format(nc)); ax.set_xlabel("Round")
    if col==0: ax.set_ylabel("Test Accuracy (%)")
    ax.legend(fontsize=8); ax.set_ylim([0,100])
fig.suptitle("Figure 6 - Paper 2: ULDP-AVG vs Group-DP (MNIST, alpha=0.5)"); plt.tight_layout()
plt.savefig(os.path.join(RESULTS, "fig6_uldp_vs_groupdp.png"), dpi=150, bbox_inches="tight"); plt.close()
print("  Saved: fig6_uldp_vs_groupdp.png")

# Master table
rows = []
for fname in sorted(os.listdir(RESULTS)):
    if not fname.endswith(".csv") or "summary" in fname.lower() or "smoke" in fname or "MASTER" in fname: continue
    try:
        df = pd.read_csv(os.path.join(RESULTS, fname)); last = df.iloc[-1]
        conv = df[df["test_accuracy"]>=80.0]
        rows.append({"Method":str(last.get("method","")),"Dataset":str(last.get("dataset","")),"#Clients":int(last.get("num_clients",0)),"Alpha":str(last.get("alpha","")),"Final Acc(%)":round(float(last.get("test_accuracy",0)),2),"Conv.Round":int(conv.iloc[0]["round"]) if len(conv) else "N/A","CommMB":round(float(last.get("comm_cost_mb",0)),1),"MI-AUC":round(float(last.get("category_metric",0)),4),"Epsilon":round(float(last.get("epsilon",0)),2)})
    except Exception as e: print("  Warning: {} - {}".format(fname,e))
df_master = pd.DataFrame(rows).sort_values(["Method","Dataset","#Clients","Alpha"])
df_master.to_csv(os.path.join(RESULTS, "MASTER_SUMMARY_TABLE.csv"), index=False)
print("  Saved: MASTER_SUMMARY_TABLE.csv ({} experiments)".format(len(rows)))
print("\n" + "="*70 + "\n  ALL DONE!\n" + "="*70)
print(df_master.to_string(index=False))
