import torch
import torch.nn.functional as F
import cv2
import numpy as np


class SpatialCorrespondenceLoss:
    """Custom loss functions to enforce spatial correspondence in CycleGAN"""
    
    def __init__(self, lambda_spatial=1.0):
        self.lambda_spatial = lambda_spatial
    
    def sobel_edge_loss(self, input_tensor, target_tensor):
        """Compare edge structures using Sobel operators"""
        batch_size = input_tensor.size(0)
        
        # Sobel kernels
        sobel_x = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=torch.float32)
        sobel_y = torch.tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=torch.float32)
        
        sobel_x = sobel_x.view(1, 1, 3, 3).to(input_tensor.device)
        sobel_y = sobel_y.view(1, 1, 3, 3).to(input_tensor.device)
        
        # Apply Sobel to both tensors
        input_edge_x = F.conv2d(input_tensor, sobel_x, padding=1)
        input_edge_y = F.conv2d(input_tensor, sobel_y, padding=1)
        input_edges = torch.sqrt(input_edge_x**2 + input_edge_y**2)
        
        target_edge_x = F.conv2d(target_tensor, sobel_x, padding=1)
        target_edge_y = F.conv2d(target_tensor, sobel_y, padding=1)
        target_edges = torch.sqrt(target_edge_x**2 + target_edge_y**2)
        
        return F.l1_loss(input_edges, target_edges)
    
    def gradient_loss(self, input_tensor, target_tensor):
        """Compare gradients to preserve structure"""
        # Compute gradients
        input_grad_x = input_tensor[:, :, :, 1:] - input_tensor[:, :, :, :-1]
        input_grad_y = input_tensor[:, :, 1:, :] - input_tensor[:, :, :-1, :]
        
        target_grad_x = target_tensor[:, :, :, 1:] - target_tensor[:, :, :, :-1]
        target_grad_y = target_tensor[:, :, 1:, :] - target_tensor[:, :, :-1, :]
        
        return F.l1_loss(input_grad_x, target_grad_x) + F.l1_loss(input_grad_y, target_grad_y)
    
    def mask_correspondence_loss(self, mask, generated_fluor):
        """
        Enforce that high-intensity regions in generated fluorescence
        correspond to mask regions
        """
        # Normalize both to [0, 1]
        mask_norm = (mask + 1) / 2.0  # From [-1, 1] to [0, 1]
        fluor_norm = (generated_fluor + 1) / 2.0
        
        # Create binary mask (threshold at 0.5)
        binary_mask = (mask_norm > 0.5).float()
        
        # Compute mean fluorescence in mask vs background regions
        mask_fluor = (binary_mask * fluor_norm).sum() / (binary_mask.sum() + 1e-8)
        bg_fluor = ((1 - binary_mask) * fluor_norm).sum() / ((1 - binary_mask).sum() + 1e-8)
        
        # Encourage higher fluorescence in mask regions
        correspondence_loss = F.relu(bg_fluor - mask_fluor + 0.2)  # margin of 0.2
        
        return correspondence_loss
    
    def __call__(self, real_mask, fake_fluor):
        """Compute combined spatial correspondence loss"""
        total_loss = 0
        
        # Edge preservation loss
        edge_loss = self.sobel_edge_loss(real_mask, fake_fluor)
        total_loss += edge_loss
        
        # Gradient preservation loss
        grad_loss = self.gradient_loss(real_mask, fake_fluor)
        total_loss += grad_loss * 0.5
        
        # Mask correspondence loss
        corr_loss = self.mask_correspondence_loss(real_mask, fake_fluor)
        total_loss += corr_loss * 2.0
        
        return total_loss * self.lambda_spatial


class AttentionCorrespondenceLoss:
    """Use attention maps to enforce correspondence"""
    
    def __init__(self, lambda_attention=1.0):
        self.lambda_attention = lambda_attention
    
    def attention_loss(self, mask, generated):
        """Compute attention-based correspondence loss"""
        # Compute attention weights from mask
        mask_norm = (mask + 1) / 2.0
        attention_weights = torch.sigmoid(mask_norm * 4 - 2)  # Sharp attention
        
        # Apply attention to generated image
        attended_gen = generated * attention_weights
        
        # Encourage high values in attended regions
        attention_loss = -attended_gen.mean()
        
        return attention_loss
    
    def __call__(self, real_mask, fake_fluor):
        return self.attention_loss(real_mask, fake_fluor) * self.lambda_attention
