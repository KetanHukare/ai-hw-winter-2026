# Adversarial Attacks on MNIST Recognition

This project implements adversarial attacks on a trained MNIST digit recognition model using PyTorch.

## Dataset

- **Source**: MNIST dataset from torchvision.datasets
- **Training**: 60,000 samples
- **Testing**: 10,000 samples

## Model Architecture

A CNN-based classifier with:
- 3 convolutional layers (32, 64, 64 filters)
- Max pooling layers
- 2 fully connected layers (128, 10 neurons)
- Dropout for regularization

## Implemented Attacks

1. **FGSM (Fast Gradient Sign Method)**
   - Single-step attack using sign of gradient
   - Fast but less effective

2. **I-FGSM (Iterative FGSM) / PGD (Projected Gradient Descent)**
   - Multi-step iterative attack
   - Projects perturbations within epsilon ball
   - More effective than FGSM

3. **MI-FGSM (Momentum Iterative FGSM)**
   - Adds momentum to iterative attacks
   - Stabilizes update direction
   - Often achieves highest attack success rate

## Metrics

- **Recognition Rate Before Attack**: Accuracy on clean test images
- **Recognition Rate After Attack**: Accuracy on adversarial images
- **Attack Success Rate (ASR)**: Percentage of correctly classified samples that become misclassified after attack

## Usage

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Train the Model

```bash
python train.py
```

This will:
- Download MNIST dataset automatically
- Train the model for 10 epochs
- Save the trained model to `checkpoints/mnist_model.pth`

### 3. Evaluate Attacks

```bash
python evaluate.py
```

Optional arguments:
- `--epsilon`: Perturbation budget (default: 0.3)
- `--alpha`: Step size for iterative attacks (default: 0.01)
- `--num-iter`: Number of iterations (default: 40)
- `--decay`: Momentum decay factor (default: 1.0)
- `--batch-size`: Batch size for testing (default: 128)
- `--model-path`: Path to trained model (default: checkpoints/mnist_model.pth)

Example with custom parameters:
```bash
python evaluate.py --epsilon 0.2 --alpha 0.005 --num-iter 50
```

## Expected Results

With epsilon=0.3:
- **Clean Accuracy**: ~99%
- **FGSM ASR**: ~40-60%
- **I-FGSM/PGD ASR**: ~80-95%
- **MI-FGSM ASR**: ~85-98%

Higher epsilon values lead to stronger attacks but more visible perturbations.

## Project Structure

```
.
├── model.py          # CNN model definition
├── train.py          # Training script
├── attacks.py        # Adversarial attack implementations
├── evaluate.py       # Evaluation script with metrics
├── requirements.txt  # Python dependencies
└── README.md         # This file
```

## References

- FGSM: Goodfellow et al., "Explaining and Harnessing Adversarial Examples" (2015)
- PGD: Madry et al., "Towards Deep Learning Models Resistant to Adversarial Attacks" (2018)
- MI-FGSM: Dong et al., "Boosting Adversarial Attacks with Momentum" (2018)
