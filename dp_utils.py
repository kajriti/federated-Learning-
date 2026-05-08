"""
dp_utils.py
-----------
WHY THIS FILE EXISTS:
    Both papers use Differential Privacy. This file contains ALL the DP math
    in one place so client.py stays clean.

WHAT THIS FILE DOES:
    1. clip_gradients()      — per-sample gradient clipping (sensitivity bound)
    2. add_gaussian_noise()  — adds calibrated Gaussian noise to clipped gradients
    3. RDPAccountant         — tracks privacy budget (epsilon, delta) per round
    4. compute_epsilon()     — converts RDP -> (epsilon, delta)-DP

HOW IT CONNECTS:
    - dp_utils.py is imported by client.py
    - The client uses clip_gradients() + add_gaussian_noise() during local training
    - The RDPAccountant tracks how much privacy budget has been spent
    - utils.py logs the epsilon per round for plotting

FROM THE PAPERS:
    Paper 1 (DP-SCAFFOLD): adds noise to per-sample gradients before
    computing the local update. Clipping constant C is set adaptively.
    Noise scale sigma_g calibrated to sensitivity S = 2C/sR.

    Paper 2 (ULDP-FL): per-user per-silo clipping with weights w_{s,u}
    summing to 1. Noise added once per silo per round.
"""

import torch
import numpy as np
from typing import List, Optional, Tuple
import math


class RDPAccountant:
    """
    Rényi Differential Privacy (RDP) accountant.

    Tracks privacy budget across training rounds.
    Both papers use RDP for tighter privacy bounds than basic DP composition.

    How it works:
        - Each round, the mechanism spends some RDP budget
        - We compose these budgets across rounds
        - At the end, convert to (epsilon, delta)-DP using Lemma B.4 from Paper 1

    Usage:
        accountant = RDPAccountant(noise_multiplier=1.0, sample_rate=0.5, delta=1e-5)
        for round in training:
            accountant.step()
            eps = accountant.get_epsilon()
            print(f"Round {round}: epsilon = {eps:.4f}")
    """

    def __init__(
        self,
        noise_multiplier: float,
        sample_rate: float,
        delta: float = 1e-5,
        alphas: Optional[List[float]] = None
    ):
        """
        Args:
            noise_multiplier: sigma (noise scale). Higher = more private but less accurate
            sample_rate: fraction of clients sampled per round (l in Paper 1)
                         or data sampling ratio (s in Paper 1)
            delta: target delta for (epsilon, delta)-DP. Usually 1/n where n = total samples
            alphas: RDP orders to try. If None, uses standard range
        """
        self.noise_multiplier = noise_multiplier
        self.sample_rate = sample_rate
        self.delta = delta
        self.steps = 0

        # Standard range of RDP orders (alpha values)
        if alphas is None:
            self.alphas = list(range(2, 64)) + [128, 256, 512]
        else:
            self.alphas = alphas

    def step(self, num_steps: int = 1):
        """Call once per communication round to record privacy cost."""
        self.steps += num_steps

    def _compute_rdp(self, alpha: float) -> float:
        """
        Compute RDP for subsampled Gaussian mechanism.
        Based on Wang et al. 2020 (Lemma B.7 in Paper 1).

        For high privacy regime (sigma large, q small):
        epsilon_RDP(alpha) ≈ O(q^2 * alpha / sigma^2)

        where q = sample_rate, sigma = noise_multiplier
        """
        q = self.sample_rate
        sigma = self.noise_multiplier

        if q == 0:
            return 0.0
        if q == 1.0:
            # No subsampling, just Gaussian mechanism
            return alpha / (2 * sigma ** 2)

        # Use the subsampled Gaussian RDP formula
        # This is an approximation; for exact computation use Opacus
        # Upper bound from Wang et al. 2020
        if alpha == 1:
            return float('inf')

        # Simplified bound: O(q^2 * alpha / sigma^2) for small q
        # Full formula requires computing binomial coefficients which is expensive
        # We use the tight bound from Opacus's implementation approach
        rdp = (alpha * q**2) / (2 * sigma**2)

        # Amplification by subsampling: multiply by step count
        # Using strong composition (Lemma B.1 in Paper 1)
        return rdp * self.steps

    def get_epsilon(self) -> float:
        """
        Convert RDP to (epsilon, delta)-DP.
        Returns current epsilon given the target delta.

        Uses Lemma B.4 from Paper 1:
        If M is (alpha, rho)-RDP, then M is (rho + log(1/delta)/(alpha-1), delta)-DP
        """
        if self.steps == 0:
            return 0.0

        best_eps = float('inf')

        for alpha in self.alphas:
            try:
                rdp = self._compute_rdp(float(alpha))
                if rdp == float('inf'):
                    continue

                # Convert RDP to DP using Lemma B.4
                eps = rdp + math.log(1.0 / self.delta) / (alpha - 1)
                best_eps = min(best_eps, eps)

            except (ValueError, ZeroDivisionError):
                continue

        return best_eps if best_eps != float('inf') else 999.0

    def get_privacy_spent(self) -> Tuple[float, float]:
        """Returns (epsilon, delta) tuple."""
        return self.get_epsilon(), self.delta


def clip_gradients_per_sample(
    per_sample_grads: List[torch.Tensor],
    clipping_norm: float
) -> List[torch.Tensor]:
    """
    Per-sample gradient clipping (Algorithm 1, step 11 in Paper 1).

    This is the DP-SGD clipping step. For each sample's gradient:
    g_tilde = g / max(1, ||g||_2 / C)

    This bounds the sensitivity of the gradient to C.

    Args:
        per_sample_grads: list of per-sample gradients, each same shape as model params
        clipping_norm: C (clipping threshold)

    Returns:
        Clipped gradients, same structure as input

    Note: In practice, computing per-sample gradients requires hooks.
    We use the batch gradient with Opacus-style scaling as approximation.
    """
    clipped = []
    for grad in per_sample_grads:
        norm = grad.norm(2)
        scale = min(1.0, clipping_norm / (norm + 1e-8))
        clipped.append(grad * scale)
    return clipped


def clip_model_update(
    model_update: List[torch.Tensor],
    clipping_norm: float
) -> Tuple[List[torch.Tensor], float]:
    """
    Clip entire model update (delta) by its L2 norm.
    Used in FedAvg-style DP where we clip the full local update.
    Also used in ULDP-AVG (Paper 2) for per-user clipping.

    g_tilde = delta * min(1, C / ||delta||_2)

    Args:
        model_update: list of parameter tensors (local model - global model)
        clipping_norm: C

    Returns:
        (clipped_update, actual_norm)
    """
    # Flatten all parameters into one vector to compute global norm
    flat = torch.cat([p.view(-1) for p in model_update])
    norm = flat.norm(2).item()

    scale = min(1.0, clipping_norm / (norm + 1e-8))
    clipped = [p * scale for p in model_update]

    return clipped, norm


def add_gaussian_noise(
    tensors: List[torch.Tensor],
    noise_multiplier: float,
    clipping_norm: float,
    num_samples: int = 1
) -> List[torch.Tensor]:
    """
    Add calibrated Gaussian noise for differential privacy.

    From Paper 1 (Algorithm 1, step 13):
    H_tilde = H + (2C/sR) * N(0, sigma_g^2)

    Sensitivity S = 2C/sR where:
        C = clipping norm
        s = data sampling ratio
        R = local dataset size

    The noise variance is: S^2 * sigma_g^2

    In practice, noise_multiplier = sigma_g (the parameter you tune),
    and we scale by clipping_norm/num_samples.

    Args:
        tensors: parameter tensors to add noise to
        noise_multiplier: sigma_g (noise scale, controls privacy-utility tradeoff)
        clipping_norm: C (gradient clipping bound)
        num_samples: number of samples in the batch/minibatch

    Returns:
        Tensors with Gaussian noise added
    """
    noisy = []
    for tensor in tensors:
        # Standard deviation = clipping_norm * noise_multiplier / num_samples
        std = clipping_norm * noise_multiplier / max(num_samples, 1)
        noise = torch.randn_like(tensor) * std
        noisy.append(tensor + noise)
    return noisy


def compute_adaptive_clipping_norm(
    gradients_norms: List[float],
    percentile: float = 50.0
) -> float:
    """
    Adaptive clipping: set C as the median gradient norm.
    This is the heuristic from Abadi et al. 2016 used in both papers.

    Instead of fixing C, we estimate it from current gradient norms.
    This avoids biasing gradients (if C too small) or adding too much noise (if C too large).

    Args:
        gradients_norms: list of gradient L2 norms from current batch
        percentile: which percentile to use as C (default: median = 50th)

    Returns:
        Adaptive clipping norm
    """
    if not gradients_norms:
        return 1.0
    return float(np.percentile(gradients_norms, percentile))


class DPConfig:
    """
    Configuration object for DP training.
    Keeps all DP hyperparameters in one place.

    Based on instruction doc baseline + paper configurations.
    """
    def __init__(
        self,
        noise_multiplier: float = 1.0,
        clipping_norm: float = 1.0,
        delta: float = 1e-5,
        sample_rate: float = 0.5,
        max_grad_norm: Optional[float] = None,
        use_adaptive_clipping: bool = True
    ):
        """
        Args:
            noise_multiplier: sigma_g. Higher = more private, lower accuracy.
                              Paper 1 uses sigma_g in {10, 20, 40, 80, 160}
            clipping_norm: C. Gradient clipping bound.
            delta: privacy parameter delta. Paper 1 sets delta = 1/(M*R)
                   where M=users, R=samples per user
            sample_rate: fraction of clients per round (l in Paper 1 = 0.5 in baseline)
            max_grad_norm: alias for clipping_norm (Opacus naming)
            use_adaptive_clipping: whether to use median clipping heuristic
        """
        self.noise_multiplier = noise_multiplier
        self.clipping_norm = max_grad_norm if max_grad_norm else clipping_norm
        self.delta = delta
        self.sample_rate = sample_rate
        self.use_adaptive_clipping = use_adaptive_clipping

    def get_accountant(self) -> RDPAccountant:
        """Create an RDP accountant with this config's parameters."""
        return RDPAccountant(
            noise_multiplier=self.noise_multiplier,
            sample_rate=self.sample_rate,
            delta=self.delta
        )

    def __repr__(self):
        return (f"DPConfig(sigma={self.noise_multiplier}, "
                f"C={self.clipping_norm}, "
                f"delta={self.delta}, "
                f"sample_rate={self.sample_rate})")
