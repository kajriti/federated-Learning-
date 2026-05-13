"""
client.py
---------
WHY THIS FILE EXISTS:
    In Flower, each FL client is a class that inherits from flwr.client.NumPyClient.
    This is where ALL local training happens.
    We have 4 client types — one per algorithm.

WHAT THIS FILE DOES:
    1. FedAvgClient     — standard FedAvg (baseline, everyone must implement)
    2. DPFedAvgClient   — FedAvg + Differential Privacy (bridge algorithm)
    3. DPScaffoldClient — Paper 1's main algorithm: SCAFFOLD + DP
    4. ULDPAvgClient    — Paper 2's main algorithm: user-level DP per silo

HOW IT CONNECTS:
    main.py creates clients by calling get_client_fn(algorithm, config)
    which returns the right client class.
    server.py aggregates the updates that clients send back.

FLOWER CLIENT INTERFACE:
    Every client must implement:
    - get_parameters(config) -> list of numpy arrays  (model weights)
    - fit(parameters, config) -> (new_parameters, num_samples, metrics)
    - evaluate(parameters, config) -> (loss, num_samples, metrics)
"""

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import flwr as fl
from typing import Dict, List, Tuple, Optional
from collections import OrderedDict
import copy

from model import get_model, get_parameters, set_parameters
from data import get_client_dataloader, get_test_dataloader
from dp_utils import (
    DPConfig, RDPAccountant,
    clip_model_update, add_gaussian_noise,
    compute_adaptive_clipping_norm
)

# Fixed seed
SEED = 42
torch.manual_seed(SEED)
np.random.seed(SEED)


# ============================================================
# BASE CLIENT — shared helper methods
# ============================================================

class BaseClient(fl.client.NumPyClient):
    """
    Base class with shared utilities.
    All 4 algorithm clients inherit from this.
    """

    def __init__(self, client_id: int, config: dict):
        self.client_id = client_id
        self.config = config
        self.dataset = config.get("dataset", "mnist")
        self.num_clients = config.get("num_clients", 10)
        self.alpha = config.get("alpha", 0.5)
        self.batch_size = config.get("batch_size", 32)
        self.local_epochs = config.get("local_epochs", 5)
        self.lr = config.get("learning_rate", 0.01)
        self.momentum = config.get("momentum", 0.9)
        self.model_type = config.get("model_type", "default")
        self.data_dir = config.get("data_dir", "./data")
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Initialize model
        self.model = get_model(self.dataset, self.model_type).to(self.device)
        self.criterion = nn.CrossEntropyLoss()

    def get_parameters(self, config) -> List[np.ndarray]:
        """Return current model parameters as numpy arrays."""
        return get_parameters(self.model)

    def _get_dataloader(self):
        """Get this client's local DataLoader."""
        return get_client_dataloader(
            client_id=self.client_id,
            dataset_name=self.dataset,
            num_clients=self.num_clients,
            alpha=self.alpha,
            batch_size=self.batch_size,
            data_dir=self.data_dir
        )

    def evaluate(self, parameters: List[np.ndarray], config: dict):
        """Evaluate model on local test data (Flower calls this automatically)."""
        set_parameters(self.model, parameters)
        loader = get_test_dataloader(self.dataset, data_dir=self.data_dir)
        self.model.eval()
        loss, correct, total = 0.0, 0, 0

        with torch.no_grad():
            for x, y in loader:
                x, y = x.to(self.device), y.to(self.device)
                out = self.model(x)
                loss += self.criterion(out, y).item() * len(y)
                correct += (out.argmax(1) == y).sum().item()
                total += len(y)

        accuracy = correct / total if total > 0 else 0.0
        return float(loss / total), total, {"accuracy": accuracy}


# ============================================================
# CLIENT 1: FedAvg (Baseline — mandatory per instruction doc)
# ============================================================

class FedAvgClient(BaseClient):
    """
    Standard FedAvg client (McMahan et al. 2017).
    This is the MANDATORY baseline. All groups must implement this first.

    Local training:
        - Load global model
        - Run K epochs of SGD with momentum
        - Send updated model back

    No privacy, no heterogeneity handling. This is what everything else improves upon.
    """

    def fit(
        self,
        parameters: List[np.ndarray],
        config: dict
    ) -> Tuple[List[np.ndarray], int, dict]:
        """
        Local training step.

        Args:
            parameters: global model parameters from server
            config: round config (can contain round number etc.)

        Returns:
            (updated_parameters, num_samples, metrics_dict)
        """
        # Load global model parameters
        set_parameters(self.model, parameters)

        # Get local data
        loader = self._get_dataloader()
        num_samples = len(loader.dataset)

        # Optimizer: SGD with momentum=0.9 as per instruction doc baseline
        optimizer = optim.SGD(
            self.model.parameters(),
            lr=self.lr,
            momentum=self.momentum,
            weight_decay=0
        )

        # Local training
        self.model.train()
        total_loss = 0.0
        for epoch in range(self.local_epochs):
            for x, y in loader:
                x, y = x.to(self.device), y.to(self.device)
                optimizer.zero_grad()
                out = self.model(x)
                loss = self.criterion(out, y)
                loss.backward()
                optimizer.step()
                total_loss += loss.item()

        avg_loss = total_loss / (self.local_epochs * len(loader))

        return (
            get_parameters(self.model),
            num_samples,
            {"train_loss": avg_loss}
        )


# ============================================================
# CLIENT 2: DP-FedAvg (FedAvg + Differential Privacy)
# ============================================================

class DPFedAvgClient(BaseClient):
    """
    DP-FedAvg: FedAvg with Differential Privacy.
    This is the STATE-OF-THE-ART baseline that Paper 1 (DP-SCAFFOLD) improves upon.

    Corresponds to Algorithm 2 in Paper 1 appendix.

    Key difference from FedAvg:
        After local training, clip the MODEL UPDATE (delta) and add Gaussian noise.
        This provides (epsilon, delta)-DP for the update sent to the server.

    DP mechanism:
        1. Train locally for K epochs (same as FedAvg)
        2. Compute delta = local_model - global_model
        3. Clip: delta_tilde = delta * min(1, C / ||delta||)
        4. Add noise: delta_private = delta_tilde + N(0, C^2 * sigma^2)
        5. Send delta_private to server
    """

    def __init__(self, client_id: int, config: dict):
        super().__init__(client_id, config)

        dp_cfg = config.get("dp", {})
        self.dp_config = DPConfig(
            noise_multiplier=dp_cfg.get("noise_multiplier", 1.0),
            clipping_norm=dp_cfg.get("clipping_norm", 1.0),
            delta=dp_cfg.get("delta", 1e-5),
            sample_rate=config.get("client_fraction", 0.5)
        )
        self.accountant = self.dp_config.get_accountant()

    def fit(
        self,
        parameters: List[np.ndarray],
        config: dict
    ) -> Tuple[List[np.ndarray], int, dict]:

        set_parameters(self.model, parameters)
        global_params = copy.deepcopy(get_parameters(self.model))  # save for delta computation

        loader = self._get_dataloader()
        num_samples = len(loader.dataset)

        optimizer = optim.SGD(
            self.model.parameters(),
            lr=self.lr,
            momentum=self.momentum
        )

        # Step 1: Standard local training (same as FedAvg)
        self.model.train()
        total_loss = 0.0
        for epoch in range(self.local_epochs):
            for x, y in loader:
                x, y = x.to(self.device), y.to(self.device)
                optimizer.zero_grad()
                loss = self.criterion(self.model(x), y)
                loss.backward()
                optimizer.step()
                total_loss += loss.item()

        # Step 2: Compute delta (local update)
        local_params = get_parameters(self.model)
        delta = [lp - gp for lp, gp in zip(local_params, global_params)]

        # Step 3: Clip the model update
        delta_tensors = [torch.tensor(d) for d in delta]
        clipped_delta, actual_norm = clip_model_update(delta_tensors, self.dp_config.clipping_norm)

        # Step 4: Add Gaussian noise
        noisy_delta = add_gaussian_noise(
            clipped_delta,
            noise_multiplier=self.dp_config.noise_multiplier,
            clipping_norm=self.dp_config.clipping_norm,
            num_samples=num_samples
        )

        # Step 5: Reconstruct noisy parameters
        noisy_params = [gp + nd.numpy() for gp, nd in zip(global_params, noisy_delta)]

        # Track privacy budget
        self.accountant.step()
        eps, delta_spent = self.accountant.get_privacy_spent()

        avg_loss = total_loss / (self.local_epochs * len(loader))

        return (
            noisy_params,
            num_samples,
            {
                "train_loss": avg_loss,
                "epsilon": eps,
                "grad_norm": actual_norm
            }
        )


# ============================================================
# CLIENT 3: DP-SCAFFOLD (Paper 1 — Main Algorithm)
# ============================================================

class DPScaffoldClient(BaseClient):
    """
    DP-SCAFFOLD: Paper 1's main contribution.
    Combines SCAFFOLD (variance reduction via control variates)
    with Differential Privacy (Gaussian noise on local gradients).

    This is Algorithm 1 in Paper 1.

    Key insight: SCAFFOLD uses control variates c and c_i to correct
    for client drift (the main problem in non-IID FL). Adding DP noise
    to per-sample gradients (not model updates) keeps the control variates
    private while still correcting drift.

    Why it's better than DP-FedAvg under heterogeneity:
        DP-FedAvg: noise + drift both hurt accuracy
        DP-SCAFFOLD: noise hurts but drift is CORRECTED by control variates
        => DP-SCAFFOLD wins when heterogeneity is high or K (local steps) is large

    Update rule (Eq 1 in Paper 1):
        y_k = y_{k-1} - eta_l * (H_tilde_k + c - c_i)
        where:
            H_tilde_k = noisy mini-batch gradient (DP noise added here)
            c = global control variate (downloaded from server)
            c_i = local control variate (maintained by this client)

    Control variate update:
        c_i_new = c_i_old - c + (1/K*eta_l) * (x_{global} - y_K)
    """

    def __init__(self, client_id: int, config: dict):
        super().__init__(client_id, config)

        dp_cfg = config.get("dp", {})
        self.dp_config = DPConfig(
            noise_multiplier=dp_cfg.get("noise_multiplier", 1.0),
            clipping_norm=dp_cfg.get("clipping_norm", 1.0),
            delta=dp_cfg.get("delta", 1e-5),
            sample_rate=config.get("data_sampling_ratio", 0.2)
        )
        self.accountant = self.dp_config.get_accountant()
        self.global_lr = config.get("global_lr", 1.0)

        # Local control variate c_i (initialized to 0, same as Paper 1)
        # Shape matches model parameters
        self.c_i = None  # will be initialized on first fit()

    def _init_control_variate(self):
        """Initialize local control variate c_i = 0 (zero vector matching model shape)."""
        return [np.zeros_like(p) for p in get_parameters(self.model)]

    def fit(
        self,
        parameters: List[np.ndarray],
        config: dict
    ) -> Tuple[List[np.ndarray], int, dict]:
        """
        DP-SCAFFOLD local update.

        What gets sent to server:
            - delta_y = y_K - x_{t-1}  (model update)
            - delta_c = c_i_new - c_i_old  (control variate update)

        Note: In Flower, we pack both into the returned parameters.
        server.py knows how to unpack them.
        """
        # Load global model x_{t-1} and global control variate c from server
        # Convention: first half = model params, second half = global control variate c
        num_model_params = len(get_parameters(self.model))
        model_params = parameters[:num_model_params]
        c_global = parameters[num_model_params:]  # c from server

        set_parameters(self.model, model_params)
        x_prev = copy.deepcopy(model_params)  # x_{t-1}

        # Initialize c_i on first call
        if self.c_i is None:
            self.c_i = self._init_control_variate()

        loader = self._get_dataloader()
        num_samples = len(loader.dataset)

        # DP-SCAFFOLD local update loop (Algorithm 1, lines 6-18)
        self.model.train()
        total_loss = 0.0
        K = self.local_epochs  # number of local updates
        eta_l = self.lr        # local learning rate

        # y_0 = x_{t-1}
        y_current = copy.deepcopy(model_params)
        set_parameters(self.model, y_current)

        # Keep track of all noisy gradients for control variate update
        all_noisy_grads = [np.zeros_like(p) for p in model_params]
        num_gradient_steps = 0

        for k, (x, y) in enumerate(loader):
            if k >= K:
                break

            x, y = x.to(self.device), y.to(self.device)

            # Compute gradient
            self.model.zero_grad()
            out = self.model(x)
            loss = self.criterion(out, y)
            loss.backward()
            total_loss += loss.item()

            # Get per-sample gradients (approximated as batch gradient here)
            batch_grads = [p.grad.data.clone().cpu().numpy() for p in self.model.parameters()]

            # Step 11: Clip each gradient
            grad_norms = []
            for g in batch_grads:
                norm = np.linalg.norm(g.flatten())
                grad_norms.append(norm)
            C = self.dp_config.clipping_norm

            clipped_grads = []
            for g, norm in zip(batch_grads, grad_norms):
                scale = min(1.0, C / (norm + 1e-8))
                clipped_grads.append(g * scale)

            # Step 13: Add DP noise — H_tilde_k (Eq. from Paper 1)
            # Noise std = 2C / (s*R) * sigma_g
            # Here we simplify: noise_std = C * sigma_g / batch_size
            noise_std = C * self.dp_config.noise_multiplier / max(len(x), 1)
            noisy_grads = [
                g + np.random.randn(*g.shape) * noise_std
                for g in clipped_grads
            ]

            # Step 14: Update y using drift correction
            # y_k = y_{k-1} - eta_l * (H_tilde_k - c_i + c)
            y_updated = []
            for y_p, ng, ci, cg in zip(y_current, noisy_grads, self.c_i, c_global):
                correction = ng - ci + cg  # drift correction: H_tilde - c_i + c
                y_updated.append(y_p - eta_l * correction)
            y_current = y_updated

            # Accumulate noisy grads for control variate update
            for j, ng in enumerate(noisy_grads):
                all_noisy_grads[j] += ng
            num_gradient_steps += 1

        # Step 15: Update control variate c_i (Eq from Paper 1)
        # c_i_new = c_i - c + (1/K*eta_l) * (x_{t-1} - y_K)
        c_i_new = []
        for ci, cg, xp, yk in zip(self.c_i, c_global, x_prev, y_current):
            c_i_update = ci - cg + (1.0 / (K * eta_l)) * (xp - yk)
            c_i_new.append(c_i_update)

        # Compute deltas to send to server
        delta_y = [yk - xp for yk, xp in zip(y_current, x_prev)]
        delta_c = [cin - ci for cin, ci in zip(c_i_new, self.c_i)]

        # Update local c_i
        self.c_i = c_i_new

        # Track privacy
        self.accountant.step()
        eps, _ = self.accountant.get_privacy_spent()

        avg_loss = total_loss / max(num_gradient_steps, 1)

        # Pack delta_y and delta_c together for server
        # Server will unpack: first half = delta_y, second half = delta_c
        combined = delta_y + delta_c

        return (
            combined,
            num_samples,
            {
                "train_loss": avg_loss,
                "epsilon": eps,
                "algorithm": "dp_scaffold"
            }
        )


# ============================================================
# CLIENT 4: ULDP-AVG (Paper 2 — User-Level DP)
# ============================================================

class ULDPAvgClient(BaseClient):
    """
    ULDP-AVG: Paper 2's main algorithm.
    User-Level Differential Privacy for cross-silo FL.

    Key problem it solves:
        In cross-silo FL, one USER may have records in MULTIPLE silos.
        Standard record-level DP doesn't protect users in this setting.
        We need user-level DP: indistinguishability for entire user removal.

    How it works (Algorithm 3 in Paper 2):
        For each user u in this silo:
            1. Train model on user u's data only (per-user training)
            2. Compute delta_u = local_model_u - global_model
            3. Clip: delta_u_tilde = w_{s,u} * delta_u * min(1, C/||delta_u||)
        Sum all clipped deltas + add one Gaussian noise per silo
        Send sum to server

    Why this works for user-level DP:
        Each user's contribution is bounded by C (clipping)
        Weights w_{s,u} sum to 1 across silos, so user sensitivity = C globally
        Noise calibrated to C/|S| per silo (|S| silos each add noise)

    Note: In our experiment, we treat each client as a "silo" and each
    sample as a "user" (simplified version without cross-silo user tracking).
    This is the standard simplification for cross-device/silo experiments.
    """

    def __init__(self, client_id: int, config: dict):
        super().__init__(client_id, config)

        dp_cfg = config.get("dp", {})
        self.dp_config = DPConfig(
            noise_multiplier=dp_cfg.get("noise_multiplier", 1.0),
            clipping_norm=dp_cfg.get("clipping_norm", 1.0),
            delta=dp_cfg.get("delta", 1e-5),
            sample_rate=config.get("user_sampling_rate", 1.0)
        )
        self.accountant = self.dp_config.get_accountant()
        self.num_silos = config.get("num_clients", 10)  # |S| = number of silos
        self.weighting_strategy = config.get("weighting_strategy", "uniform")  # "uniform" or "optimal"
        self.local_epochs_per_user = config.get("local_epochs_per_user", 1)

    def fit(
        self,
        parameters: List[np.ndarray],
        config: dict
    ) -> Tuple[List[np.ndarray], int, dict]:
        """
        ULDP-AVG local training.

        We simulate per-user training by treating each mini-batch as one "user's" data.
        This captures the spirit of per-user clipping without requiring
        explicit user IDs in the dataset.
        """
        set_parameters(self.model, parameters)
        global_params = copy.deepcopy(get_parameters(self.model))

        loader = self._get_dataloader()
        num_samples = len(loader.dataset)
        num_users_in_silo = len(loader)  # number of mini-batches = simulated users

        C = self.dp_config.clipping_norm
        S = self.num_silos

        # Weight for this silo: w_{s,u} = 1/|S| for uniform weighting
        # For optimal weighting: w_{s,u} = n_{s,u} / sum_s(n_{s,u})
        # We use uniform here (can extend to optimal)
        w_su = 1.0 / S  # uniform weight

        # Accumulate weighted clipped deltas across all users
        aggregated_delta = [np.zeros_like(p) for p in global_params]

        total_loss = 0.0
        user_count = 0

        # Per-user training (Algorithm 3, lines 9-16 in Paper 2)
        for x_u, y_u in loader:
            x_u, y_u = x_u.to(self.device), y_u.to(self.device)

            # Reset model to global params for each user
            user_model = get_model(self.dataset, self.model_type).to(self.device)
            set_parameters(user_model, global_params)
            user_model.train()

            # Train on this user's data for Q epochs
            optimizer = optim.SGD(user_model.parameters(), lr=self.lr, momentum=self.momentum)
            for q in range(self.local_epochs_per_user):
                optimizer.zero_grad()
                out = user_model(x_u)
                loss = self.criterion(out, y_u)
                loss.backward()
                optimizer.step()
                total_loss += loss.item()

            # Compute delta_u = local_u - global
            local_u_params = get_parameters(user_model)
            delta_u = [lu - gp for lu, gp in zip(local_u_params, global_params)]

            # Per-user clipping: delta_u_tilde = w_{s,u} * delta_u * min(1, C/||delta_u||)
            delta_u_tensors = [torch.tensor(d) for d in delta_u]
            clipped_delta_u, _ = clip_model_update(delta_u_tensors, C)

            # Apply weight and accumulate
            for j, cd in enumerate(clipped_delta_u):
                aggregated_delta[j] += w_su * cd.numpy()

            user_count += 1

        # Add one Gaussian noise per silo (Algorithm 3, line 17)
        # Noise variance: sigma^2 * C^2 / |S|
        # This calibrates noise to the sensitivity C/|S| per silo
        noise_scale = C / np.sqrt(S)
        aggregated_delta_tensors = [torch.tensor(d) for d in aggregated_delta]
        noisy_delta = add_gaussian_noise(
            aggregated_delta_tensors,
            noise_multiplier=self.dp_config.noise_multiplier,
            clipping_norm=noise_scale,
            num_samples=1  # noise added once per silo
        )

        # Final parameters = global + noisy aggregated delta
        final_params = [gp + nd.numpy() for gp, nd in zip(global_params, noisy_delta)]

        # Track privacy
        self.accountant.step()
        eps, _ = self.accountant.get_privacy_spent()

        avg_loss = total_loss / max(user_count * self.local_epochs_per_user, 1)

        return (
            final_params,
            num_samples,
            {
                "train_loss": avg_loss,
                "epsilon": eps,
                "num_users": user_count,
                "algorithm": "uldp_avg"
            }
        )


# ============================================================
# FACTORY FUNCTION — main.py calls this
# ============================================================

def get_client_fn(algorithm: str, config: dict):
    """
    Returns a Flower client_fn for the given algorithm.

    Flower expects: client_fn(cid: str) -> fl.client.Client

    Args:
        algorithm: "fedavg", "dp_fedavg", "dp_scaffold", "uldp_avg"
        config: experiment configuration dict

    Returns:
        A function that creates the right client for a given cid

    Usage in main.py:
        client_fn = get_client_fn("dp_scaffold", config)
        fl.simulation.start_simulation(client_fn=client_fn, ...)
    """

    algorithm_map = {
        "fedavg": FedAvgClient,
        "dp_fedavg": DPFedAvgClient,
        "dp_scaffold": DPScaffoldClient,
        "uldp_avg": ULDPAvgClient,
    }

    if algorithm not in algorithm_map:
        raise ValueError(f"Unknown algorithm: {algorithm}. "
                         f"Choose from: {list(algorithm_map.keys())}")

    ClientClass = algorithm_map[algorithm]

    def client_fn(cid: str) -> fl.client.Client:
        return ClientClass(client_id=int(cid), config=config).to_client()

    return client_fn
