import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from model import MNISTNet, MLP, CNN, TransformerEncoder, get_model
import os
import ssl
import argparse


def train_model(model_name='mnistnet', epochs=10, batch_size=128, lr=0.001, device='cuda'):
    ssl._create_default_https_context = ssl._create_unverified_context
    
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    
    train_dataset = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0)
    
    model = get_model(model_name).to(device)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"\nModel: {model_name}")
    print(f"Total parameters: {total_params:,}")
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    
    print(f"Training on {device}")
    print(f"Total training samples: {len(train_dataset)}")
    
    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        
        for batch_idx, (data, target) in enumerate(train_loader):
            data, target = data.to(device), target.to(device)
            
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
            _, predicted = output.max(1)
            total += target.size(0)
            correct += predicted.eq(target).sum().item()
            
            if (batch_idx + 1) % 100 == 0:
                print(f'Epoch [{epoch+1}/{epochs}], Step [{batch_idx+1}/{len(train_loader)}], '
                      f'Loss: {running_loss/100:.4f}, Acc: {100.*correct/total:.2f}%')
                running_loss = 0.0
        
        epoch_acc = 100. * correct / total
        print(f'Epoch [{epoch+1}/{epochs}] completed - Accuracy: {epoch_acc:.2f}%')
    
    os.makedirs('checkpoints', exist_ok=True)
    model_path = f'checkpoints/{model_name}_model.pth'
    torch.save(model.state_dict(), model_path)
    print(f"Model saved to {model_path}")
    
    return model


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train MNIST models')
    parser.add_argument('--model', type=str, default='mnistnet',
                        choices=['mnistnet', 'mlp', 'cnn', 'transformer'],
                        help='Model architecture to train')
    parser.add_argument('--epochs', type=int, default=10, help='Number of epochs')
    parser.add_argument('--batch-size', type=int, default=128, help='Batch size')
    parser.add_argument('--lr', type=float, default=0.001, help='Learning rate')
    args = parser.parse_args()
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    train_model(model_name=args.model, epochs=args.epochs, 
                batch_size=args.batch_size, lr=args.lr, device=device)
