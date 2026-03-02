import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from model import MNISTNet, MLP, CNN, TransformerEncoder, get_model
from attacks import AdversarialAttacks
import numpy as np
import argparse
import ssl


def evaluate_clean_accuracy(model, test_loader, device):
    model.eval()
    correct = 0
    total = 0
    
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            outputs = model(data)
            _, predicted = outputs.max(1)
            total += target.size(0)
            correct += predicted.eq(target).sum().item()
    
    accuracy = 100. * correct / total
    return accuracy


def evaluate_attack(model, test_loader, attack_fn, device, attack_name):
    model.eval()
    correct_before = 0
    correct_after = 0
    total = 0
    
    for data, target in test_loader:
        data, target = data.to(device), target.to(device)
        
        with torch.no_grad():
            outputs_clean = model(data)
            _, pred_clean = outputs_clean.max(1)
            correct_before += pred_clean.eq(target).sum().item()
        
        adv_data = attack_fn(data, target)
        
        with torch.no_grad():
            outputs_adv = model(adv_data)
            _, pred_adv = outputs_adv.max(1)
            correct_after += pred_adv.eq(target).sum().item()
        
        total += target.size(0)
    
    acc_before = 100. * correct_before / total
    acc_after = 100. * correct_after / total
    asr = 100. * (correct_before - correct_after) / correct_before if correct_before > 0 else 0
    
    print(f"\n{attack_name} Results:")
    print(f"  Recognition Rate (Before Attack): {acc_before:.2f}%")
    print(f"  Recognition Rate (After Attack):  {acc_after:.2f}%")
    print(f"  Attack Success Rate (ASR):        {asr:.2f}%")
    
    return {
        'attack_name': attack_name,
        'acc_before': acc_before,
        'acc_after': acc_after,
        'asr': asr
    }


def main():
    parser = argparse.ArgumentParser(description='Evaluate adversarial attacks on MNIST')
    parser.add_argument('--model', type=str, default='mnistnet',
                        choices=['mnistnet', 'mlp', 'cnn', 'transformer'],
                        help='Model architecture to evaluate')
    parser.add_argument('--batch-size', type=int, default=128, help='batch size for testing')
    parser.add_argument('--epsilon', type=float, default=0.3, help='perturbation budget')
    parser.add_argument('--alpha', type=float, default=0.01, help='step size for iterative attacks')
    parser.add_argument('--num-iter', type=int, default=40, help='number of iterations for iterative attacks')
    parser.add_argument('--decay', type=float, default=1.0, help='momentum decay factor')
    parser.add_argument('--model-path', type=str, default=None, help='path to trained model (auto-detected if not provided)')
    args = parser.parse_args()
    
    if args.model_path is None:
        args.model_path = f'checkpoints/{args.model}_model.pth'
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    print(f"Model: {args.model}")
    print(f"Loading from: {args.model_path}")
    
    ssl._create_default_https_context = ssl._create_unverified_context
    
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    
    test_dataset = datasets.MNIST(root='./data', train=False, download=True, transform=transform)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=0)
    
    model = get_model(args.model).to(device)
    
    # Check if model file exists
    import os
    if not os.path.exists(args.model_path):
        raise FileNotFoundError(
            f"Model file not found: {args.model_path}\n"
            f"Please train the model first using: python train.py --model {args.model}"
        )
    
    # Load model with error handling
    try:
        model.load_state_dict(torch.load(args.model_path, map_location=device))
    except Exception as e:
        raise RuntimeError(
            f"Failed to load model from {args.model_path}\n"
            f"Error: {str(e)}\n"
            f"The checkpoint might be corrupted or incompatible with the model architecture."
        )
    
    model.eval()
    
    print(f"\nTotal test samples: {len(test_dataset)}")
    
    clean_acc = evaluate_clean_accuracy(model, test_loader, device)
    print(f"\nClean Test Accuracy: {clean_acc:.2f}%")
    
    attacker = AdversarialAttacks(model, device)
    
    results = []
    
    print("\n" + "="*60)
    print("ADVERSARIAL ATTACK EVALUATION")
    print("="*60)
    print(f"Epsilon: {args.epsilon}")
    print(f"Alpha: {args.alpha}")
    print(f"Iterations: {args.num_iter}")
    print(f"Momentum Decay: {args.decay}")
    print("="*60)
    
    fgsm_result = evaluate_attack(
        model, test_loader,
        lambda x, y: attacker.fgsm_attack(x, y, args.epsilon),
        device,
        f"FGSM (ε={args.epsilon})"
    )
    results.append(fgsm_result)
    
    ifgsm_result = evaluate_attack(
        model, test_loader,
        lambda x, y: attacker.ifgsm_attack(x, y, args.epsilon, args.alpha, args.num_iter),
        device,
        f"I-FGSM (ε={args.epsilon}, α={args.alpha}, iter={args.num_iter})"
    )
    results.append(ifgsm_result)
    
    pgd_result = evaluate_attack(
        model, test_loader,
        lambda x, y: attacker.pgd_attack(x, y, args.epsilon, args.alpha, args.num_iter),
        device,
        f"PGD (ε={args.epsilon}, α={args.alpha}, iter={args.num_iter})"
    )
    results.append(pgd_result)
    
    mifgsm_result = evaluate_attack(
        model, test_loader,
        lambda x, y: attacker.mifgsm_attack(x, y, args.epsilon, args.alpha, args.num_iter, args.decay),
        device,
        f"MI-FGSM (ε={args.epsilon}, α={args.alpha}, iter={args.num_iter}, μ={args.decay})"
    )
    results.append(mifgsm_result)
    
    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)
    print(f"{'Attack':<40} {'Acc Before':<12} {'Acc After':<12} {'ASR':<10}")
    print("-"*60)
    for r in results:
        print(f"{r['attack_name']:<40} {r['acc_before']:>10.2f}% {r['acc_after']:>10.2f}% {r['asr']:>8.2f}%")
    print("="*60)


if __name__ == '__main__':
    main()
