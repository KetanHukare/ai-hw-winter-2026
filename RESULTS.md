# Adversarial Attack Results on MNIST

## Model Performance

**Clean Test Accuracy**: 99.10%

The CNN model was trained for 10 epochs and achieved 99.34% training accuracy and 99.10% test accuracy.

## Attack Results (ε=0.3, α=0.01, iterations=40)

| Attack Method | Recognition Rate (Before) | Recognition Rate (After) | Attack Success Rate (ASR) |
|---------------|---------------------------|--------------------------|---------------------------|
| **FGSM** | 99.10% | 87.83% | **11.37%** |
| **I-FGSM** | 99.10% | 74.14% | **25.19%** |
| **PGD** | 99.10% | 74.14% | **25.19%** |
| **MI-FGSM** | 99.10% | 77.17% | **22.13%** |

## Analysis

### Attack Effectiveness Ranking
1. **I-FGSM/PGD** (25.19% ASR) - Most effective
2. **MI-FGSM** (22.13% ASR) - Second most effective
3. **FGSM** (11.37% ASR) - Least effective

### Key Findings

- **Iterative attacks (I-FGSM/PGD)** are significantly more effective than single-step FGSM, achieving over 2x higher ASR
- **I-FGSM and PGD** produce identical results (as expected, since PGD is essentially I-FGSM with projection)
- **MI-FGSM** performs slightly worse than I-FGSM/PGD despite using momentum, likely because the momentum helps with transferability but not necessarily white-box attacks
- All attacks successfully fool the model to some degree, with iterative methods reducing accuracy from 99.10% to ~74%

### Attack Parameters Used
- **Epsilon (ε)**: 0.3 - Maximum perturbation budget in normalized space
- **Alpha (α)**: 0.01 - Step size for iterative attacks
- **Iterations**: 40 - Number of attack iterations
- **Momentum Decay (μ)**: 1.0 - Momentum factor for MI-FGSM

## How to Reproduce

```bash
# Train the model
source .venv/bin/activate
python train.py

# Run attacks with default parameters
python evaluate.py

# Run attacks with custom parameters
python evaluate.py --epsilon 0.2 --alpha 0.005 --num-iter 50
```

## Notes

- All attacks work in the normalized input space (mean=0.1307, std=0.3081)
- The model uses CPU for inference (can be accelerated with GPU)
- Attack Success Rate (ASR) = (Correctly classified before - Correctly classified after) / Correctly classified before
