import torch
import torch.nn as nn
import numpy as np
from typing import Tuple, Dict, Optional
from sklearn.metrics import roc_auc_score


# ============================================================
# ATTACK 1: Gradient Inversion (simplified DLG)
# ============================================================

def gradient_inversion_attack(
    model: nn.Module,
    real_data: torch.Tensor,
    real_labels: torch.Tensor,
    device: torch.device,
    num_iters: int = 100,
    lr: float = 0.1
) -> Tuple[float, float]:
    """
    Simplified Deep Leakage from Gradients (DLG) attack.
    Reference: Zhu et al. 2019 "Deep Leakage from Gradients"

    HOW IT WORKS:
        1. Attacker observes gradients from the model (what the server sees in FL)
        2. Attacker initializes random dummy data
        3. Attacker optimizes dummy data so that its gradients match the real gradients
        4. If successful, dummy data ≈ real data (privacy breach!)

    WHY DP HELPS:
        DP adds noise to gradients before sharing.
        This noise prevents the attacker from precisely matching gradients.
        Higher noise (lower epsilon) = lower attack success.

    Args:
        model: the current global model
        real_data: batch of real training data [B, C, H, W]
        real_labels: corresponding labels [B]
        device: cpu or cuda
        num_iters: optimization iterations (more = better attack but slower)
        lr: learning rate for dummy data optimization

    Returns:
        (reconstruction_mse, attack_success_rate)
        - reconstruction_mse: MSE between dummy and real data (lower = better attack)
        - attack_success_rate: fraction of batches where MSE < threshold
    """
    model.eval()
    criterion = nn.CrossEntropyLoss()

    # Step 1: Compute real gradients (what the server observes in FL)
    real_data = real_data.to(device)
    real_labels = real_labels.to(device)

    model.zero_grad()
    real_output = model(real_data)
    real_loss = criterion(real_output, real_labels)
    real_grads = torch.autograd.grad(real_loss, model.parameters())
    real_grads = [g.detach() for g in real_grads]

    # Step 2: Initialize dummy data randomly
    dummy_data = torch.randn_like(real_data, requires_grad=True, device=device)
    dummy_labels = real_labels.clone()  # assume labels known (common assumption)

    optimizer = torch.optim.LBFGS([dummy_data], lr=lr, max_iter=20)

    best_mse = float('inf')

    # Step 3: Optimize dummy data to match real gradients
    for iteration in range(num_iters):
        def closure():
            optimizer.zero_grad()
            model.zero_grad()

            dummy_output = model(dummy_data)
            dummy_loss = criterion(dummy_output, dummy_labels)
            dummy_grads = torch.autograd.grad(
                dummy_loss, model.parameters(),
                create_graph=True, allow_unused=True
            )

            # Gradient matching loss: ||dummy_grads - real_grads||^2
            grad_diff = sum(
                ((dg - rg) ** 2).sum()
                for dg, rg in zip(dummy_grads, real_grads)
                if dg is not None
            )
            grad_diff.backward()
            return grad_diff

        optimizer.step(closure)

        # Track MSE between dummy and real
        with torch.no_grad():
            mse = ((dummy_data - real_data) ** 2).mean().item()
            best_mse = min(best_mse, mse)

        if iteration % 20 == 0:
            pass  # silent

    # Normalize MSE to [0, 1] for attack success rate
    # MSE < 0.01 = successful reconstruction
    attack_threshold = 0.01
    attack_success = 1.0 if best_mse < attack_threshold else 0.0

    return best_mse, attack_success


def run_gradient_inversion_experiment(
    model: nn.Module,
    test_loader,
    device: torch.device,
    num_batches: int = 5,
    num_iters: int = 50
) -> Dict[str, float]:
    """
    Run gradient inversion attack on multiple batches.
    Returns average MSE and attack success rate.

    Instruction doc metric: "reconstruction MSE, attack success rate"
    """
    all_mse = []
    all_success = []

    model.eval()
    for i, (x, y) in enumerate(test_loader):
        if i >= num_batches:
            break

        # Only use single sample for DLG (cleaner attack)
        x_single = x[:1].to(device)
        y_single = y[:1].to(device)

        try:
            mse, success = gradient_inversion_attack(
                model, x_single, y_single, device,
                num_iters=num_iters
            )
            all_mse.append(mse)
            all_success.append(success)
        except Exception as e:
            print(f"    Attack batch {i} failed: {e}")
            all_mse.append(1.0)
            all_success.append(0.0)

    return {
        "reconstruction_mse": float(np.mean(all_mse)) if all_mse else 1.0,
        "attack_success_rate": float(np.mean(all_success)) if all_success else 0.0,
        "num_batches_attacked": len(all_mse)
    }


# ============================================================
# ATTACK 2: Membership Inference Attack
# ============================================================

def membership_inference_attack(
    model: nn.Module,
    train_loader,
    test_loader,
    device: torch.device,
    num_shadow_samples: int = 200
) -> Dict[str, float]:
    """
    Simple membership inference attack using confidence scores.
    Reference: Shokri et al. 2017 "Membership Inference Attacks Against ML Models"

    HOW IT WORKS:
        1. For training samples (members): model has HIGH confidence
        2. For test samples (non-members): model has LOWER confidence
        3. Attacker uses confidence threshold to decide: member or not?
        4. AUC measures how well this works.

    WHY DP HELPS:
        DP prevents overfitting, so model confidence is similar for
        members and non-members.
        Better privacy = lower AUC (closer to 0.5 = random guessing)

    Instruction doc metric: "membership inference AUC"

    Returns:
        dict with 'membership_inference_auc' key
    """
    model.eval()
    criterion = nn.CrossEntropyLoss(reduction='none')

    member_scores = []
    nonmember_scores = []

    # Members: samples from training set
    count = 0
    with torch.no_grad():
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            out = model(x)
            probs = torch.softmax(out, dim=1)
            # Confidence = probability of correct class
            conf = probs[range(len(y)), y].cpu().numpy()
            member_scores.extend(conf.tolist())
            count += len(y)
            if count >= num_shadow_samples:
                break

    # Non-members: samples from test set
    count = 0
    with torch.no_grad():
        for x, y in test_loader:
            x, y = x.to(device), y.to(device)
            out = model(x)
            probs = torch.softmax(out, dim=1)
            conf = probs[range(len(y)), y].cpu().numpy()
            nonmember_scores.extend(conf.tolist())
            count += len(y)
            if count >= num_shadow_samples:
                break

    # Compute AUC: can attacker distinguish members from non-members?
    n_members = len(member_scores)
    n_nonmembers = len(nonmember_scores)
    all_scores = member_scores + nonmember_scores
    all_labels = [1] * n_members + [0] * n_nonmembers  # 1=member, 0=non-member

    if len(set(all_labels)) < 2 or len(all_scores) < 4:
        return {"membership_inference_auc": 0.5}

    try:
        auc = roc_auc_score(all_labels, all_scores)
    except Exception:
        auc = 0.5

    # AUC = 0.5: attacker can't distinguish (perfect privacy)
    # AUC = 1.0: attacker perfectly identifies members (no privacy)
    return {
        "membership_inference_auc": float(auc),
        "member_conf_mean": float(np.mean(member_scores)),
        "nonmember_conf_mean": float(np.mean(nonmember_scores)),
    }


# ============================================================
# COMBINED: Run all attacks for Cat. 1 report
# ============================================================

def run_all_privacy_attacks(
    model: nn.Module,
    train_loader,
    test_loader,
    device: torch.device,
    run_gradient_inversion: bool = True,
    run_membership_inference: bool = True,
) -> Dict[str, float]:
    """
    Run all Category 1 privacy attacks and return combined metrics.

    Call this after training completes for each algorithm.
    Results show how well the DP mechanism protects against attacks.

    Expected results:
        FedAvg:       high attack success, high AUC  (no protection)
        DP-FedAvg:    medium attack success, medium AUC
        DP-SCAFFOLD:  lower attack success than DP-FedAvg (Paper 1 claim)
        ULDP-AVG:     low attack success (user-level protection)

    Args:
        model: trained global model
        train_loader: training data (to sample member examples)
        test_loader: test data (non-members)
        device: computation device
        run_gradient_inversion: whether to run DLG attack (slow)
        run_membership_inference: whether to run MI attack (fast)

    Returns:
        dict with all attack metrics
    """
    results = {}

    print("  Running privacy attacks...")

    if run_membership_inference:
        print("    [1/2] Membership inference attack...")
        mi_results = membership_inference_attack(model, train_loader, test_loader, device)
        results.update(mi_results)
        print(f"    MI AUC: {mi_results['membership_inference_auc']:.4f}")

    if run_gradient_inversion:
        print("    [2/2] Gradient inversion attack (DLG)...")
        gi_results = run_gradient_inversion_experiment(
            model, test_loader, device,
            num_batches=3,
            num_iters=30
        )
        results.update(gi_results)
        print(f"    Reconstruction MSE: {gi_results['reconstruction_mse']:.6f}")
        print(f"    Attack success rate: {gi_results['attack_success_rate']:.4f}")

    return results
