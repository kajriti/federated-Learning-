# federated-Learning-
This project implements Federated Learning with Differential Privacy using the [Flower framework](https://flower.ai?utm_source=chatgpt.com). It compares FedAvg, DP-SCAFFOLD, and ULDP-FL methods on MNIST, FashionMNIST, and CIFAR-10 datasets to improve privacy, security, and performance in non-IID environments. 


# Federated Learning with Differential Privacy

**Course:** Federated Learning — Jan-April 2026 (BDS)
**Category:** Cat. 1 — Privacy & Inference
**Framework:** [Flower (flwr)](https://flower.ai)

---

## Assigned Research Papers

| # | Title | Venue | Key Contribution |
|---|-------|-------|-----------------|
| 1 | [Differentially Private Federated Learning on Heterogeneous Data](https://arxiv.org/abs/2111.09278) | AISTATS 2022 | **DP-SCAFFOLD**: Integrates DP into SCAFFOLD via control variates to handle heterogeneity + privacy jointly |
| 2 | [Uldp-FL: Federated Learning with Across-Silo User-Level Differential Privacy](https://doi.org/10.14778/3681954.3681966) | VLDB 2024 | **ULDP-AVG/SGD**: User-level DP in cross-silo FL via per-user weighted clipping |



---

## Setup Instructions

### 1. Install Dependencies

```bash
# Create virtual environment (recommended)
python -m venv flenv
flenv\Scripts\activate        # Windows
# source flenv/bin/activate    # Linux/Mac

# Install requirements
pip install -r requirements.txt
```

### 2. Verify Installation

```bash
python -c "import flwr, torch, torchvision; print('All OK')"
```

---

## Running Experiments

### Baseline — FedAvg

```bash
# Single run
python experiments/run_fedavg.py --dataset mnist --clients 10 --rounds 100 --alpha 0.5

# Full sweep (all client counts & alpha values)
python experiments/run_fedavg.py --sweep --dataset mnist --rounds 100
```

### Paper 1 — DP-SCAFFOLD vs DP-FedAvg

```bash
# Single run
python experiments/run_dp_scaffold.py --dataset mnist --clients 10 --rounds 100 --epsilon 5.0 --alpha 0.5

# Compare with sweep
python experiments/run_dp_scaffold.py --sweep --dataset mnist --rounds 100
```

### Paper 2 — ULDP-FL (Cross-Silo)

```bash
# Single run
python experiments/run_uldp_fl.py --dataset mnist --silos 5 --clients 100 --rounds 100 --epsilon 5.0

# Full sweep
python experiments/run_uldp_fl.py --sweep --dataset mnist --rounds 100
```

---

## Experimental Settings (Per Course Instructions)

| Parameter | Value |
|-----------|-------|
| Framework | Flower 1.8.0 |
| Seed | 42 |
| Client fraction | 0.5 |
| Local epochs | 5 |
| Batch size | 32 |
| Optimizer | SGD, momentum=0.9, lr=0.01 |
| Loss | Cross-Entropy |
| Data partition | Dirichlet α ∈ {0.01, 0.1, 0.5, 1.0, IID} |
| Clients | 10, 50, 100 |
| Datasets | MNIST, FashionMNIST, CIFAR-10 |

---

## Algorithms Implemented

### FedAvg (Baseline)
Standard Federated Averaging (McMahan et al., 2017).

### DP-SCAFFOLD (Paper 1)
> Noble et al., "Differentially Private Federated Learning on Heterogeneous Data", AISTATS 2022

Key features:
- **Control variates** to correct user drift in non-IID settings
- **Gradient clipping** at threshold C
- **Gaussian DP noise** calibrated to ℓ₂-sensitivity S = 2C/sR
- **Rényi DP accounting** for tight privacy budget tracking
- Warm-start initialization for control variates

### ULDP-AVG / Group-DP (Paper 2)
> Kato et al., "Uldp-FL: Federated Learning with Across-Silo User-Level DP", VLDB 2024

Key features:
- **Cross-silo FL**: users' data spans multiple silos
- **Per-user weighted clipping**: bounds user-level sensitivity directly (avoids group-privacy superlinear degradation)
- **Adaptive weighting**: weights based on user record distribution
- **Group-DP baseline** for comparison

---

## Evaluation Metrics

### Universal Metrics (All Experiments)
- Global Test Accuracy (every round)
- Global Test Loss
- Convergence Round (first round exceeding 80%)
- Communication Cost (MB)

### Category 1 — Privacy Metrics
- Membership Inference AUC (approximated)
- Privacy budget ε (RDP accounting)

---

## Output Files

All results are saved to `results/`:

| File | Description |
|------|-------------|
| `*_accuracy_vs_rounds.png` | Accuracy curves (Figure 1) |
| `*_loss_vs_rounds.png` | Loss curves (Figure 2) |
| `*_mi_auc.png` | Privacy metric vs rounds (Figure 3) |
| `*_iid_vs_noniid.png` | IID vs Non-IID comparison (Figure 4) |
| `*_bar.png` | FedAvg vs proposed bar chart (Figure 5) |
| `*_summary.csv` | Results table per instructions §6.2 |



## References

1. McMahan et al. (2017). *Communication-Efficient Learning of Deep Networks from Decentralized Data.* AISTATS.
2. Karimireddy et al. (2020). *SCAFFOLD: Stochastic Controlled Averaging for Federated Learning.* ICML.
3. Noble et al. (2022). *Differentially Private Federated Learning on Heterogeneous Data.* AISTATS. [arXiv:2111.09278]
4. Kato et al. (2024). *Uldp-FL: Federated Learning with Across-Silo User-Level Differential Privacy.* VLDB.
5. Dwork & Roth (2014). *The Algorithmic Foundations of Differential Privacy.*
6. Beutel et al. (2020). *Flower: A Friendly Federated Learning Research Framework.*
