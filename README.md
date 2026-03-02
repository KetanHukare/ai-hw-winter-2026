# MNIST Recognition & Adversarial Attacks

A comprehensive PyTorch project combining multiple neural network architectures for MNIST digit classification with state-of-the-art adversarial attack implementations.

## Overview

This project consists of two main components:
1. **Model Training & Comparison**: Train and compare 4 different neural network architectures (MNISTNet, MLP, CNN, Transformer)
2. **Adversarial Attacks**: Implement and evaluate 3 adversarial attack algorithms (FGSM, I-FGSM/PGD, MI-FGSM)

## Dataset

- **Source**: MNIST dataset from torchvision.datasets
- **Training**: 60,000 samples (28×28 grayscale images)
- **Testing**: 10,000 samples
- **Classes**: 10 (digits 0-9)
- **Normalization**: mean=0.1307, std=0.3081

## Model Architectures

### 1. MNISTNet (Original)
Compact CNN optimized for adversarial robustness testing:
- 3 convolutional layers (32, 64, 64 filters)
- Max pooling layers
- 2 fully connected layers (128, 10 neurons)
- Dropout (0.5) for regularization
- **Parameters**: ~1.2M

### 2. MLP (Multi-Layer Perceptron)
Simple feedforward network:
- 2 hidden layers (256, 128 neurons)
- ReLU activation
- Dropout (0.2)
- **Parameters**: ~234K

### 3. CNN (Enhanced Convolutional Network)
Deeper CNN architecture:
- 3 convolutional layers (32, 64, 128 filters)
- 2 max pooling layers
- Fully connected layers (256, 10 neurons)
- Dropout (0.25)
- **Parameters**: ~1.7M

### 4. Transformer Encoder
Vision Transformer-style architecture:
- Patch embedding (7×7 patches)
- 3 transformer encoder layers
- 4 attention heads
- GELU activation
- **Parameters**: ~53K

## Adversarial Attacks

### 1. FGSM (Fast Gradient Sign Method)
**Formula**: `x_adv = x + ε × sign(∇Loss)`
- **Type**: Single-step attack
- **Speed**: Very fast (1 gradient computation)
- **Effectiveness**: Moderate
- **ASR**: ~11.37% (ε=0.3)

### 2. I-FGSM / PGD (Iterative FGSM / Projected Gradient Descent)
**Algorithm**:
```
for i in 1 to 40:
    x += α × sign(∇Loss)
    x = clip(x - original, -ε, ε)  # Project to ε-ball
```
- **Type**: Multi-step iterative attack
- **Speed**: Slower (40 iterations)
- **Effectiveness**: High
- **ASR**: ~25.19% (ε=0.3, α=0.01, 40 iterations)
- **Note**: I-FGSM and PGD are mathematically equivalent in our implementation

### 3. MI-FGSM (Momentum Iterative FGSM)
**Algorithm**:
```
momentum = 0
for i in 1 to 40:
    momentum = μ × momentum + normalized_gradient
    x += α × sign(momentum)
    x = clip(x - original, -ε, ε)
```
- **Type**: Iterative attack with momentum
- **Speed**: Slower (40 iterations)
- **Effectiveness**: High (especially for transferability)
- **ASR**: ~22.13% (ε=0.3, α=0.01, μ=1.0, 40 iterations)

## Evaluation Metrics

- **Recognition Rate Before Attack**: Accuracy on clean test images (baseline)
- **Recognition Rate After Attack**: Accuracy on adversarial examples
- **Attack Success Rate (ASR)**: `(Correct_before - Correct_after) / Correct_before × 100%`
  - Percentage of originally correct predictions that became incorrect
  - Primary metric for attack effectiveness

## Installation

```bash
pip install -r requirements.txt
```

## Usage

### Training Models

**Train default model (MNISTNet)**:
```bash
python train.py
```

**Train specific architecture**:
```bash
python train.py --model mlp
python train.py --model cnn
python train.py --model transformer
python train.py --model mnistnet
```

**Custom training parameters**:
```bash
python train.py --model cnn --epochs 20 --batch-size 256 --lr 0.0005
```

**Training arguments**:
- `--model`: Model architecture (mnistnet, mlp, cnn, transformer) [default: mnistnet]
- `--epochs`: Number of training epochs [default: 10]
- `--batch-size`: Batch size [default: 128]
- `--lr`: Learning rate [default: 0.001]

**Output**:
- Trained model saved to `checkpoints/{model_name}_model.pth`
- Training progress printed to console

### Evaluating Adversarial Attacks

**Evaluate default model (MNISTNet)**:
```bash
python evaluate.py
```

**Evaluate specific model**:
```bash
python evaluate.py --model mlp
python evaluate.py --model cnn
python evaluate.py --model transformer
```

**Custom attack parameters**:
```bash
python evaluate.py --model cnn --epsilon 0.2 --alpha 0.005 --num-iter 50
```

**Evaluation arguments**:
- `--model`: Model architecture to evaluate [default: mnistnet]
- `--epsilon`: Perturbation budget (ε) [default: 0.3]
- `--alpha`: Step size for iterative attacks (α) [default: 0.01]
- `--num-iter`: Number of iterations [default: 40]
- `--decay`: Momentum decay factor (μ) [default: 1.0]
- `--batch-size`: Batch size [default: 128]
- `--model-path`: Custom model path (auto-detected if not provided)

**Output**:
- Clean test accuracy
- Attack results for all 3 methods (FGSM, I-FGSM/PGD, MI-FGSM)
- Detailed metrics: Recognition rates before/after, ASR
- Summary comparison table

### Generate Presentation

```bash
python generate_presentation.py
```

Creates `Adversarial_Attacks_MNIST_Presentation.pptx` with 17 slides covering:
- Project overview
- Model architectures
- Attack algorithms
- Results and analysis

## Results (MNISTNet, ε=0.3)

### Model Performance
- **Training Accuracy**: 99.34%
- **Test Accuracy**: 99.10%

### Attack Results

| Attack Method | ASR | Accuracy After Attack | Effectiveness |
|---------------|-----|----------------------|---------------|
| **FGSM** | 11.37% | 87.83% | ⭐⭐ |
| **I-FGSM** | 25.19% | 74.14% | ⭐⭐⭐⭐ |
| **PGD** | 25.19% | 74.14% | ⭐⭐⭐⭐ |
| **MI-FGSM** | 22.13% | 77.17% | ⭐⭐⭐ |

### Key Findings
1. **Iterative attacks are 2.2× more effective** than single-step FGSM
2. **I-FGSM and PGD produce identical results** (mathematically equivalent)
3. **MI-FGSM slightly underperforms** in white-box setting but excels in transferability
4. **Best attack fooled 1 in 4** correctly classified images

### Parameter Impact
- **Higher ε**: Stronger attacks but more visible perturbations
- **More iterations**: Better ASR but slower execution
- **Smaller α**: More precise optimization but requires more iterations

## Project Structure

```
.
├── model.py                    # All 4 model architectures
│   ├── MNISTNet               # Original compact CNN
│   ├── MLP                    # Multi-layer perceptron
│   ├── CNN                    # Enhanced CNN
│   └── TransformerEncoder     # Vision Transformer
├── train.py                    # Training script with multi-model support
├── attacks.py                  # 3 adversarial attack implementations
│   ├── FGSM
│   ├── I-FGSM/PGD
│   └── MI-FGSM
├── evaluate.py                 # Evaluation with comprehensive metrics
├── generate_presentation.py    # Auto-generate PowerPoint
├── requirements.txt            # Python dependencies
├── README.md                   # This file
├── RESULTS.md                  # Detailed results and analysis
└── checkpoints/                # Saved model weights
    ├── mnistnet_model.pth
    ├── mlp_model.pth
    ├── cnn_model.pth
    └── transformer_model.pth
```

## Technical Details

### Attack Parameters
- **ε (epsilon)**: 0.3 - Maximum perturbation budget in normalized space
- **α (alpha)**: 0.01 - Step size for iterative attacks
- **Iterations**: 40 - Number of attack steps for I-FGSM/PGD/MI-FGSM
- **μ (mu/decay)**: 1.0 - Momentum retention factor for MI-FGSM

### Normalization
All images are normalized with MNIST statistics:
- Mean: 0.1307
- Std: 0.3081
- Valid range: approximately [-0.42, 2.82]

### Critical Implementation Notes
1. **SSL Fix**: Added `ssl._create_unverified_context` for MNIST download
2. **Multiprocessing**: Set `num_workers=0` for macOS compatibility
3. **Normalization Handling**: Attacks clamp to normalized range, not [0, 1]
4. **Model Selection**: Automatic model path detection based on architecture name

## Dependencies

- PyTorch >= 2.0.0
- torchvision >= 0.15.0
- numpy >= 1.24.0
- python-pptx >= 0.6.21
- tqdm >= 4.65.0
- matplotlib >= 3.7.0
- datasets >= 2.14.0
- pillow >= 9.0.0

## Project Evolution

This project combines:
1. **ai-hw-spring-2026**: Original MNIST training with multiple architectures (MLP, CNN, Transformer)
2. **Attack MNIST Recognition**: Adversarial attack implementations (FGSM, I-FGSM/PGD, MI-FGSM)

The integration provides a comprehensive framework for both model comparison and adversarial robustness evaluation.

## References

### Adversarial Attacks
- **FGSM**: Goodfellow et al., "Explaining and Harnessing Adversarial Examples" (ICLR 2015)
- **PGD**: Madry et al., "Towards Deep Learning Models Resistant to Adversarial Attacks" (ICLR 2018)
- **MI-FGSM**: Dong et al., "Boosting Adversarial Attacks with Momentum" (CVPR 2018)

### Model Architectures
- **Vision Transformer**: Dosovitskiy et al., "An Image is Worth 16x16 Words" (ICLR 2021)
- **CNN**: LeCun et al., "Gradient-Based Learning Applied to Document Recognition" (1998)

## License

See LICENSE file for details.

## Acknowledgments

- MNIST dataset: Yann LeCun, Corinna Cortes, Christopher J.C. Burges
- PyTorch framework and community
- Adversarial robustness research community
