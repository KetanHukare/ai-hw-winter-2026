import torch
import torch.nn as nn


class AdversarialAttacks:
    def __init__(self, model, device='cuda', mean=0.1307, std=0.3081):
        self.model = model
        self.device = device
        self.model.eval()
        self.mean = mean
        self.std = std
        self.min_val = (0 - mean) / std
        self.max_val = (1 - mean) / std
    
    def fgsm_attack(self, images, labels, epsilon):
        images = images.clone().detach().to(self.device)
        labels = labels.clone().detach().to(self.device)
        images.requires_grad = True
        
        outputs = self.model(images)
        loss = nn.CrossEntropyLoss()(outputs, labels)
        
        self.model.zero_grad()
        loss.backward()
        
        data_grad = images.grad.data
        perturbed_images = images + epsilon * data_grad.sign()
        perturbed_images = torch.clamp(perturbed_images, self.min_val, self.max_val)
        
        return perturbed_images.detach()
    
    def ifgsm_attack(self, images, labels, epsilon, alpha, num_iter):
        images = images.clone().detach().to(self.device)
        labels = labels.clone().detach().to(self.device)
        
        original_images = images.clone().detach()
        
        for i in range(num_iter):
            images.requires_grad = True
            outputs = self.model(images)
            loss = nn.CrossEntropyLoss()(outputs, labels)
            
            self.model.zero_grad()
            loss.backward()
            
            data_grad = images.grad.data
            images = images.detach() + alpha * data_grad.sign()
            
            delta = torch.clamp(images - original_images, min=-epsilon, max=epsilon)
            images = torch.clamp(original_images + delta, self.min_val, self.max_val).detach()
        
        return images
    
    def pgd_attack(self, images, labels, epsilon, alpha, num_iter):
        return self.ifgsm_attack(images, labels, epsilon, alpha, num_iter)
    
    def mifgsm_attack(self, images, labels, epsilon, alpha, num_iter, decay=1.0):
        images = images.clone().detach().to(self.device)
        labels = labels.clone().detach().to(self.device)
        
        original_images = images.clone().detach()
        momentum = torch.zeros_like(images).to(self.device)
        
        for i in range(num_iter):
            images.requires_grad = True
            outputs = self.model(images)
            loss = nn.CrossEntropyLoss()(outputs, labels)
            
            self.model.zero_grad()
            loss.backward()
            
            data_grad = images.grad.data
            grad_norm = torch.norm(data_grad.view(data_grad.shape[0], -1), p=1, dim=1)
            grad_norm = grad_norm.view(-1, 1, 1, 1)
            normalized_grad = data_grad / (grad_norm + 1e-8)
            
            momentum = decay * momentum + normalized_grad
            
            images = images.detach() + alpha * momentum.sign()
            
            delta = torch.clamp(images - original_images, min=-epsilon, max=epsilon)
            images = torch.clamp(original_images + delta, self.min_val, self.max_val).detach()
        
        return images
