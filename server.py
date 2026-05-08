"""
server.py
---------
WHY THIS FILE EXISTS:
    In Flower, the server aggregates model updates from clients.
    Different algorithms need different aggregation strategies.

WHAT THIS FILE DOES:
    1. FedAvgStrategy      — standard weighted average of model parameters
    2. DPFedAvgStrategy    — same as FedAvg (noise already added client-side)
    3. DPScaffoldStrategy  — aggregates BOTH model updates AND control variates
    4. ULDPAvgStrategy     — standard average (DP already applied client-side)
    5. evaluate_global()   — evaluates global model on test set every round

HOW IT CONNECTS:
    main.py creates the strategy and passes it to fl.simulation.start_simulation()
    The strategy is called by Flower after each round with all client updates.
    utils.py handles logging the results from each round.

FLOWER STRATEGY INTERFACE:
    A strategy must implement:
    - aggregate_fit() -> aggregated parameters
    - aggregate_evaluate() -> aggregated metrics
    - configure_fit() -> per-client training configs
    - configure_evaluate() -> per-client eval configs

FROM THE INSTRUCTION DOC:
    "Client Fraction per Round: 0.5 (50% of total clients sampled per round)"
    "Algorithm: FedAvg for baseline, then category-specific on top"
"""

import numpy as np
import flwr as fl
from flwr.common import (
    Metrics, Parameters, Scalar,
    FitIns, FitRes, EvaluateIns, EvaluateRes,
    parameters_to_ndarrays, ndarrays_to_parameters
)
from flwr.server.client_proxy import ClientProxy
from flwr.server.strategy import FedAvg
from typing import Dict, List, Optional, Tuple, Union
import torch
from collections import OrderedDict

from model import get_model, set_parameters, get_parameters
from data import get_test_dataloader

# Fixed seed
import random
random.seed(42)
np.random.seed(42)


def weighted_average(metrics: List[Tuple[int, Metrics]]) -> Metrics:
    """
    Aggregate metrics from multiple clients using weighted average.
    Flower calls this after evaluate() on all clients.

    Weight = num_samples (more data = more weight in average)
    """
    if not metrics:
        return {}

    total_samples = sum(num_samples for num_samples, _ in metrics)

    aggregated = {}
    for num_samples, m in metrics:
        weight = num_samples / total_samples
        for key, value in m.items():
            if isinstance(value, (int, float)):
                aggregated[key] = aggregated.get(key, 0.0) + weight * float(value)

    return aggregated


def get_eval_fn(model_name: str, dataset: str, data_dir: str = "./data"):
    """
    Returns a server-side evaluation function.
    Called by Flower strategy after each aggregation round.

    This is where we compute:
    - Global Test Accuracy (required by instruction doc)
    - Global Test Loss (required by instruction doc)
    - Convergence Round (first round to exceed 80% accuracy)

    Args:
        model_name: model architecture name
        dataset: dataset name
        data_dir: data directory

    Returns:
        eval_fn(server_round, parameters, config) -> (loss, metrics)
    """
    def evaluate(
        server_round: int,
        parameters: Parameters,
        config: dict
    ) -> Optional[Tuple[float, dict]]:

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = get_model(dataset, model_name).to(device)
        set_parameters(model, parameters_to_ndarrays(parameters))

        loader = get_test_dataloader(dataset, data_dir=data_dir)
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

        avg_loss = total_loss / total if total > 0 else 0.0
        accuracy = correct / total if total > 0 else 0.0

        print(f"  [Server Round {server_round}] "
              f"Loss: {avg_loss:.4f} | Accuracy: {accuracy:.4f}")

        return avg_loss, {
            "global_accuracy": accuracy,
            "global_loss": avg_loss,
            "server_round": server_round
        }

    return evaluate


# ============================================================
# STRATEGY 1: FedAvg (Baseline)
# ============================================================

class FedAvgStrategy(FedAvg):
    """
    Standard FedAvg strategy.
    Inherits from Flower's built-in FedAvg — we extend it with logging.

    Aggregation rule:
        x_new = sum(n_i * x_i) / sum(n_i)
        where n_i = num_samples of client i
    """

    def __init__(self, config: dict, results_logger=None):
        self.exp_config = config
        self.results_logger = results_logger
        self.round_results = []

        eval_fn = get_eval_fn(
            model_name=config.get("model_type", "default"),
            dataset=config.get("dataset", "mnist"),
            data_dir=config.get("data_dir", "./data")
        )

        super().__init__(
            fraction_fit=config.get("client_fraction", 0.5),
            fraction_evaluate=0.0,  # don't evaluate on clients, use server eval
            min_fit_clients=max(2, int(config.get("num_clients", 10) * 0.5)),
            min_evaluate_clients=0,
            min_available_clients=config.get("num_clients", 10),
            evaluate_fn=eval_fn,
            on_fit_config_fn=self._fit_config,
        )

    def _fit_config(self, server_round: int) -> dict:
        """Send round number to clients (clients can use this if needed)."""
        return {"server_round": server_round}

    def aggregate_fit(
        self,
        server_round: int,
        results: List[Tuple[ClientProxy, FitRes]],
        failures: List[Union[Tuple[ClientProxy, FitRes], BaseException]]
    ) -> Tuple[Optional[Parameters], dict]:

        if not results:
            return None, {}

        # Standard FedAvg aggregation
        aggregated_parameters, metrics = super().aggregate_fit(
            server_round, results, failures
        )

        # Log metrics
        client_metrics = [(res.num_examples, res.metrics) for _, res in results]
        agg_metrics = weighted_average(client_metrics)

        # Log to results
        self.round_results.append({
            "round": server_round,
            "algorithm": "fedavg",
            **agg_metrics
        })

        return aggregated_parameters, agg_metrics


# ============================================================
# STRATEGY 2: DP-FedAvg
# ============================================================

class DPFedAvgStrategy(FedAvgStrategy):
    """
    DP-FedAvg strategy.
    Same aggregation as FedAvg — the DP is applied CLIENT-SIDE.
    We just track the additional privacy metrics.

    The server simply averages the already-noisy updates.
    This is the "honest-but-curious server" model from Paper 1.
    """

    def aggregate_fit(self, server_round, results, failures):
        aggregated_parameters, metrics = super().aggregate_fit(
            server_round, results, failures
        )

        # Extract epsilon from client metrics
        epsilons = [res.metrics.get("epsilon", 0.0) for _, res in results]
        avg_epsilon = float(np.mean(epsilons)) if epsilons else 0.0

        if self.round_results:
            self.round_results[-1]["epsilon"] = avg_epsilon
            self.round_results[-1]["algorithm"] = "dp_fedavg"

        return aggregated_parameters, {**metrics, "epsilon": avg_epsilon}


# ============================================================
# STRATEGY 3: DP-SCAFFOLD
# ============================================================

class DPScaffoldStrategy(FedAvg):
    """
    DP-SCAFFOLD server-side aggregation.

    This is MORE COMPLEX than FedAvg because we must handle control variates.

    What the server maintains:
        x_t: global model (same as FedAvg)
        c_t: global control variate (NEW — needs special aggregation)

    What clients send:
        delta_y: model update (y_K - x_{t-1})
        delta_c: control variate update (c_i_new - c_i_old)

    Server aggregation (Step 20-21 in Algorithm 1, Paper 1):
        delta_x = (1/lM) * sum(delta_y_i for i in C_t)
        delta_c = (1/M) * sum(delta_c_i for all i)  [not just sampled]
        x_t = x_{t-1} + eta_g * delta_x
        c_t = c_{t-1} + l * delta_c

    Key difference from FedAvg:
        Server must send BOTH x_t AND c_t to clients each round.
        Clients send back BOTH delta_y AND delta_c.
    """

    def __init__(self, config: dict, results_logger=None):
        self.exp_config = config
        self.results_logger = results_logger
        self.round_results = []
        self.num_clients = config.get("num_clients", 10)
        self.client_fraction = config.get("client_fraction", 0.5)
        self.global_lr = config.get("global_lr", 1.0)
        self.dataset = config.get("dataset", "mnist")
        self.model_type = config.get("model_type", "default")
        self.data_dir = config.get("data_dir", "./data")

        # Initialize global control variate c = 0
        device = torch.device("cpu")
        init_model = get_model(self.dataset, self.model_type)
        self.c_global = [np.zeros_like(p) for p in get_parameters(init_model)]
        self.global_params = None  # will be set on first round

        eval_fn = get_eval_fn(self.model_type, self.dataset, self.data_dir)

        super().__init__(
            fraction_fit=self.client_fraction,
            fraction_evaluate=0.0,
            min_fit_clients=max(2, int(self.num_clients * self.client_fraction)),
            min_evaluate_clients=0,
            min_available_clients=self.num_clients,
            evaluate_fn=eval_fn,
            on_fit_config_fn=self._fit_config,
            initial_parameters=self._get_initial_parameters()
        )

    def _get_initial_parameters(self):
        """Initialize model + control variate = zeros concatenated."""
        model = get_model(self.dataset, self.model_type)
        model_params = get_parameters(model)
        # Send model params + c_global (both zero initially)
        combined = model_params + self.c_global
        return ndarrays_to_parameters(combined)

    def _fit_config(self, server_round: int) -> dict:
        return {"server_round": server_round}

    def aggregate_fit(self, server_round, results, failures):
        if not results:
            return None, {}

        num_model_params = len(self.c_global)  # same shape as model params

        # Unpack delta_y and delta_c from each client
        all_delta_y = []
        all_delta_c = []
        weights = []

        for client_proxy, fit_res in results:
            params = parameters_to_ndarrays(fit_res.parameters)
            n = fit_res.num_examples

            # First half = delta_y, second half = delta_c
            delta_y = params[:num_model_params]
            delta_c = params[num_model_params:]

            all_delta_y.append((n, delta_y))
            all_delta_c.append(delta_c)
            weights.append(n)

        total_weight = sum(weights)

        # Aggregate delta_y: weighted average (Step 20a in Algorithm 1)
        l = self.client_fraction  # client sampling ratio
        agg_delta_y = []
        for j in range(num_model_params):
            layer_update = sum(
                (w / total_weight) * dy[j]
                for (w, dy) in all_delta_y
            )
            agg_delta_y.append(layer_update)

        # Aggregate delta_c: simple average over sampled clients (Step 20b)
        agg_delta_c = []
        n_sampled = len(all_delta_c)
        for j in range(num_model_params):
            layer_c = sum(dc[j] for dc in all_delta_c) / n_sampled
            agg_delta_c.append(layer_c)

        # Update global model: x_t = x_{t-1} + eta_g * delta_x (Step 21a)
        if self.global_params is None:
            # First round: extract from initial parameters
            init_params = parameters_to_ndarrays(self._get_initial_parameters())
            self.global_params = init_params[:num_model_params]

        self.global_params = [
            xp + self.global_lr * dy
            for xp, dy in zip(self.global_params, agg_delta_y)
        ]

        # Update global control variate: c_t = c_{t-1} + l * delta_c (Step 21b)
        self.c_global = [
            cg + l * dc
            for cg, dc in zip(self.c_global, agg_delta_c)
        ]

        # Pack model + control variate to send to clients next round
        combined = self.global_params + self.c_global
        new_parameters = ndarrays_to_parameters(combined)

        # Collect metrics
        client_metrics = [(res.num_examples, res.metrics) for _, res in results]
        agg_metrics = weighted_average(client_metrics)

        epsilons = [res.metrics.get("epsilon", 0.0) for _, res in results]
        avg_eps = float(np.mean(epsilons)) if epsilons else 0.0

        result = {
            "round": server_round,
            "algorithm": "dp_scaffold",
            "epsilon": avg_eps,
            **agg_metrics
        }
        self.round_results.append(result)

        return new_parameters, {**agg_metrics, "epsilon": avg_eps}


# ============================================================
# STRATEGY 4: ULDP-AVG
# ============================================================

class ULDPAvgStrategy(FedAvgStrategy):
    """
    ULDP-AVG server-side aggregation.

    Same as FedAvg aggregation — the user-level DP is applied CLIENT-SIDE.
    Server aggregates the already-privatized silo updates.

    From Paper 2, Algorithm 3, line 6:
        x_{t+1} = x_t + eta_g * (1/|U|*|S|) * sum_s(Delta_s_t)

    Note the normalization by |U| instead of |S| (different from standard FedAvg).
    """

    def aggregate_fit(self, server_round, results, failures):
        aggregated_parameters, metrics = super().aggregate_fit(
            server_round, results, failures
        )

        epsilons = [res.metrics.get("epsilon", 0.0) for _, res in results]
        avg_eps = float(np.mean(epsilons)) if epsilons else 0.0

        if self.round_results:
            self.round_results[-1]["epsilon"] = avg_eps
            self.round_results[-1]["algorithm"] = "uldp_avg"

        return aggregated_parameters, {**metrics, "epsilon": avg_eps}


# ============================================================
# FACTORY FUNCTION
# ============================================================

def get_strategy(algorithm: str, config: dict, results_logger=None):
    """
    Returns the right Flower strategy for the given algorithm.

    Args:
        algorithm: "fedavg", "dp_fedavg", "dp_scaffold", "uldp_avg"
        config: experiment config dict
        results_logger: optional logger for saving results

    Returns:
        Flower strategy object
    """
    strategy_map = {
        "fedavg": FedAvgStrategy,
        "dp_fedavg": DPFedAvgStrategy,
        "dp_scaffold": DPScaffoldStrategy,
        "uldp_avg": ULDPAvgStrategy,
    }

    if algorithm not in strategy_map:
        raise ValueError(f"Unknown algorithm: {algorithm}. "
                         f"Choose from: {list(strategy_map.keys())}")

    return strategy_map[algorithm](config, results_logger)
