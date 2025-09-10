import torch
import numpy as np
import os
import logging
from PIL import Image
import matplotlib.pyplot as plt


def tensor_to_image(tensor):
    """Convert a tensor to a numpy image array"""
    # Denormalize from [-1, 1] to [0, 1]
    image = (tensor.cpu().detach().numpy() + 1.0) / 2.0
    
    # Handle different tensor shapes
    if len(image.shape) == 3:  # (C, H, W)
        if image.shape[0] == 1:  # Single channel
            image = image.squeeze(0)  # Remove channel dimension -> (H, W)
        else:  # Multi-channel
            image = np.transpose(image, (1, 2, 0))  # (C, H, W) -> (H, W, C)
    
    # Clip to valid range and convert to uint8
    image = np.clip(image * 255, 0, 255).astype(np.uint8)
    return image


def save_images(visuals, save_dir, epoch, step):
    """Save visual results"""
    os.makedirs(save_dir, exist_ok=True)
    
    # Create a figure with subplots
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    
    titles = ['Real A (Masks)', 'Fake B (Generated Fluorescent)', 'Reconstructed A',
              'Real B (Fluorescent)', 'Fake A (Generated Masks)', 'Reconstructed B']
    
    images = [
        tensor_to_image(visuals['real_A'][0]),
        tensor_to_image(visuals['fake_B'][0]),
        tensor_to_image(visuals['rec_A'][0]),
        tensor_to_image(visuals['real_B'][0]),
        tensor_to_image(visuals['fake_A'][0]),
        tensor_to_image(visuals['rec_B'][0])
    ]
    
    for i, (ax, img, title) in enumerate(zip(axes.flat, images, titles)):
        # Display grayscale images with appropriate colormap
        if len(img.shape) == 2:  # Grayscale
            ax.imshow(img, cmap='gray')
        else:  # RGB
            ax.imshow(img)
        ax.set_title(title)
        ax.axis('off')
    
    plt.tight_layout()
    plt.savefig(f'{save_dir}/epoch_{epoch+1}_step_{step}.png', dpi=150, bbox_inches='tight')
    plt.close()
    
    # Save individual images
    for name, tensor in visuals.items():
        img = tensor_to_image(tensor[0])
        # Save grayscale images properly
        if len(img.shape) == 2:  # Grayscale
            Image.fromarray(img, mode='L').save(f'{save_dir}/{name}_epoch_{epoch+1}_step_{step}.png')
        else:  # RGB
            Image.fromarray(img).save(f'{save_dir}/{name}_epoch_{epoch+1}_step_{step}.png')


def setup_logger(log_dir):
    """Setup logging"""
    os.makedirs(log_dir, exist_ok=True)
    
    # Create logger
    logger = logging.getLogger('cyclegan')
    logger.setLevel(logging.INFO)
    
    # Remove existing handlers
    for handler in logger.handlers[:]:
        logger.removeHandler(handler)
    
    # Create file handler
    file_handler = logging.FileHandler(os.path.join(log_dir, 'training.log'))
    file_handler.setLevel(logging.INFO)
    
    # Create console handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    
    # Create formatter
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    file_handler.setFormatter(formatter)
    console_handler.setFormatter(formatter)
    
    # Add handlers to logger
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)
    
    return logger


def create_synthetic_masks(output_dir, num_images=100, image_size=256):
    """Create synthetic segmentation masks for testing"""
    os.makedirs(output_dir, exist_ok=True)
    
    print(f"Creating {num_images} synthetic masks...")
    
    for i in range(num_images):
        # Create a blank image
        mask = np.zeros((image_size, image_size, 3), dtype=np.uint8)
        
        # Add random shapes
        num_shapes = np.random.randint(3, 8)
        
        for _ in range(num_shapes):
            # Random shape type
            shape_type = np.random.choice(['circle', 'rectangle', 'ellipse'])
            
            # Random color
            color = tuple(np.random.randint(0, 256, 3).tolist())
            
            if shape_type == 'circle':
                center = (np.random.randint(50, image_size-50), np.random.randint(50, image_size-50))
                radius = np.random.randint(10, 50)
                y, x = np.ogrid[:image_size, :image_size]
                circle_mask = (x - center[0])**2 + (y - center[1])**2 <= radius**2
                mask[circle_mask] = color
                
            elif shape_type == 'rectangle':
                x1 = np.random.randint(0, image_size//2)
                y1 = np.random.randint(0, image_size//2)
                x2 = np.random.randint(x1+20, image_size)
                y2 = np.random.randint(y1+20, image_size)
                mask[y1:y2, x1:x2] = color
                
            elif shape_type == 'ellipse':
                center = (np.random.randint(50, image_size-50), np.random.randint(50, image_size-50))
                a = np.random.randint(15, 40)  # semi-major axis
                b = np.random.randint(10, 30)  # semi-minor axis
                y, x = np.ogrid[:image_size, :image_size]
                ellipse_mask = ((x - center[0])/a)**2 + ((y - center[1])/b)**2 <= 1
                mask[ellipse_mask] = color
        
        # Save mask
        Image.fromarray(mask).save(f'{output_dir}/mask_{i:04d}.png')
    
    print(f"Synthetic masks saved to {output_dir}")


def load_checkpoint(model, checkpoint_path, device):
    """Load model checkpoint"""
    if os.path.exists(checkpoint_path):
        print(f'Loading checkpoint from {checkpoint_path}')
        checkpoint = torch.load(checkpoint_path, map_location=device)
        model.netG_A.load_state_dict(checkpoint['netG_A_state_dict'])
        model.netG_B.load_state_dict(checkpoint['netG_B_state_dict'])
        if hasattr(model, 'netD_A'):
            model.netD_A.load_state_dict(checkpoint['netD_A_state_dict'])
            model.netD_B.load_state_dict(checkpoint['netD_B_state_dict'])
        return checkpoint.get('epoch', 0)
    else:
        print(f'No checkpoint found at {checkpoint_path}')
        return 0


def calculate_fid(real_images, fake_images):
    """Calculate Frechet Inception Distance (FID) - placeholder implementation"""
    # This is a simplified placeholder. For actual FID calculation,
    # you would need to use a pre-trained Inception network
    print("FID calculation not implemented - would require Inception network")
    return 0.0


def visualize_results(real_A, fake_B, real_B, fake_A, save_path):
    """Create a visualization of CycleGAN results"""
    fig, axes = plt.subplots(2, 4, figsize=(16, 8))
    
    # Convert tensors to images
    real_A_img = tensor_to_image(real_A)
    fake_B_img = tensor_to_image(fake_B)
    real_B_img = tensor_to_image(real_B)
    fake_A_img = tensor_to_image(fake_A)
    
    # Determine if images are grayscale
    is_grayscale = len(real_A_img.shape) == 2
    cmap = 'gray' if is_grayscale else None
    
    # Plot images
    axes[0, 0].imshow(real_A_img, cmap=cmap)
    axes[0, 0].set_title('Real A (Mask)')
    axes[0, 0].axis('off')
    
    axes[0, 1].imshow(fake_B_img, cmap=cmap)
    axes[0, 1].set_title('Fake B (Generated Fluorescent)')
    axes[0, 1].axis('off')
    
    axes[0, 2].imshow(real_B_img, cmap=cmap)
    axes[0, 2].set_title('Real B (Fluorescent)')
    axes[0, 2].axis('off')
    
    axes[0, 3].imshow(fake_A_img, cmap=cmap)
    axes[0, 3].set_title('Fake A (Generated Mask)')
    axes[0, 3].axis('off')
    
    # Add difference maps
    diff_AB = np.abs(real_A_img.astype(float) - fake_B_img.astype(float))
    diff_BA = np.abs(real_B_img.astype(float) - fake_A_img.astype(float))
    
    axes[1, 0].imshow(diff_AB.astype(np.uint8), cmap=cmap)
    axes[1, 0].set_title('|Real A - Fake B|')
    axes[1, 0].axis('off')
    
    axes[1, 1].imshow(diff_BA.astype(np.uint8), cmap=cmap)
    axes[1, 1].set_title('|Real B - Fake A|')
    axes[1, 1].axis('off')
    
    # Hide empty subplots
    axes[1, 2].axis('off')
    axes[1, 3].axis('off')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
