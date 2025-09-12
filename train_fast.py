#!/usr/bin/env python3
"""
Fast training script for CycleGAN with optimized settings for speed
"""

import torch
import torch.nn as nn
import os
import time
import itertools
import argparse
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from models.cycle_gan_model import CycleGANModel
from data.dataset import create_dataloader
from config import FastConfig  # Use the fast configuration
from utils import save_images, save_representative_images_as_tiff, setup_logger


def train_cyclegan_fast(args):
    """Main training function for CycleGAN with speed optimizations"""
    
    # Initialize fast configuration with command line arguments
    config = FastConfig(
        dataroot=args.dataroot,
        name=args.name,
        checkpoints_dir=args.checkpoints_dir,
        results_dir=args.results_dir,
        n_epochs=args.n_epochs,
        n_epochs_decay=args.n_epochs_decay,
        save_epoch_freq=args.save_epoch_freq,
        display_freq=args.display_freq,
        print_freq=args.print_freq,
        batch_size=args.batch_size,
        lr=args.lr,
        image_size=args.image_size
    )
    
    # Set device and enable optimizations
    device = torch.device('cuda:0' if torch.cuda.is_available() and len(config.gpu_ids) > 0 else 'cpu')
    print(f'Using device: {device}')
    
    # Enable CUDA optimizations for faster training
    if torch.cuda.is_available():
        torch.backends.cudnn.benchmark = True  # Optimize for fixed input sizes
        torch.backends.cudnn.deterministic = False  # Allow non-deterministic for speed
        print('✓ CUDA optimizations enabled')
    
    # Create directories
    os.makedirs(config.checkpoints_dir, exist_ok=True)
    os.makedirs(config.results_dir, exist_ok=True)
    os.makedirs(f'{config.checkpoints_dir}/{config.name}', exist_ok=True)
    
    # Setup logging
    logger = setup_logger(f'{config.checkpoints_dir}/{config.name}')
    writer = SummaryWriter(f'{config.checkpoints_dir}/{config.name}/logs')
    
    # Create data loaders with optimizations
    train_loader = create_dataloader(
        config.dataroot, 
        phase='train',
        batch_size=config.batch_size,
        num_workers=config.num_workers,
        image_size=config.image_size
    )
    
    # Create representative images output directory
    representative_dir = os.path.join(config.results_dir, config.name, 'representative_images')
    os.makedirs(representative_dir, exist_ok=True)
    
    print(f'Training dataset size: {len(train_loader.dataset)}')
    print(f'Batch size: {config.batch_size}, Effective batches per epoch: {len(train_loader)}')
    print(f'Image size: {config.image_size}x{config.image_size}')
    print(f'Generator: {config.netG} with {config.ngf} filters')
    print(f'Discriminator: {config.netD} with {config.ndf} filters, {config.n_layers_D} layers')
    
    # Initialize model
    model = CycleGANModel(config)
    
    # Enable mixed precision training for speed (if available)
    scaler = None
    if torch.cuda.is_available() and hasattr(torch.cuda.amp, 'GradScaler'):
        scaler = torch.cuda.amp.GradScaler()
        print('✓ Mixed precision training enabled')
    
    # Training loop with optimizations
    total_steps = 0
    start_time = time.time()
    
    for epoch in range(config.n_epochs + config.n_epochs_decay):
        epoch_start_time = time.time()
        epoch_iter = 0
        
        # Update learning rate
        if epoch > config.n_epochs:
            lr = config.lr * (config.n_epochs + config.n_epochs_decay - epoch) / config.n_epochs_decay
            for param_group in model.optimizer_G.param_groups:
                param_group['lr'] = lr
            for param_group in model.optimizer_D.param_groups:
                param_group['lr'] = lr
        
        model.train()
        
        # Use tqdm with better formatting for speed monitoring
        pbar = tqdm(train_loader, desc=f'Epoch {epoch+1}/{config.n_epochs + config.n_epochs_decay}')
        
        for i, data in enumerate(pbar):
            total_steps += config.batch_size
            epoch_iter += config.batch_size
            
            # Set input and optimize with mixed precision if available
            model.set_input(data)
            
            if scaler is not None:
                # Mixed precision training
                with torch.cuda.amp.autocast():
                    model.forward()
                    model.backward_G()
                scaler.scale(model.loss_G).backward()
                scaler.step(model.optimizer_G)
                
                with torch.cuda.amp.autocast():
                    model.backward_D_A()
                    model.backward_D_B()
                scaler.scale(model.loss_D_A).backward()
                scaler.scale(model.loss_D_B).backward()
                scaler.step(model.optimizer_D)
                scaler.update()
                
                model.optimizer_G.zero_grad()
                model.optimizer_D.zero_grad()
            else:
                # Standard training
                model.optimize_parameters()
            
            # Update progress bar with current losses
            if i % 10 == 0:  # Update every 10 batches to avoid slowdown
                losses = model.get_current_losses()
                loss_str = ' | '.join([f'{k}: {v:.3f}' for k, v in list(losses.items())[:4]])  # Show first 4 losses
                pbar.set_postfix_str(loss_str)
            
            # Logging (less frequent for speed)
            if total_steps % config.print_freq == 0:
                losses = model.get_current_losses()
                
                # Log to tensorboard
                for name, value in losses.items():
                    writer.add_scalar(f'Loss/{name}', value, total_steps)
                
                # Print losses
                loss_str = ' | '.join([f'{k}: {v:.4f}' for k, v in losses.items()])
                print(f'Epoch: {epoch+1}, Step: {total_steps}, {loss_str}')
                logger.info(f'Epoch: {epoch+1}, Step: {total_steps}, {loss_str}')
            
            # Save images (less frequent for speed)
            if total_steps % config.display_freq == 0:
                visuals = model.get_current_visuals()
                save_images(visuals, f'{config.results_dir}/{config.name}', epoch, total_steps)
        
        # Save checkpoints
        if (epoch + 1) % config.save_epoch_freq == 0:
            model.save_networks(f'{config.checkpoints_dir}/{config.name}', epoch + 1)
            print(f'Saved checkpoint for epoch {epoch+1}')
        
        # Save representative images every N epochs
        if (epoch + 1) % args.save_images_freq == 0:
            print(f'Saving representative images for epoch {epoch+1}...')
            model.eval()
            with torch.no_grad():
                # Get a batch of data for visualization
                sample_data = next(iter(train_loader))
                model.set_input(sample_data)
                model.forward()
                visuals = model.get_current_visuals()
                
                # Save comprehensive representative images as TIFF files
                save_representative_images_as_tiff(visuals, representative_dir, epoch+1)
                
                print(f'Representative TIFF images saved to {representative_dir}')
            model.train()
        
        epoch_time = time.time() - epoch_start_time
        total_time = time.time() - start_time
        avg_epoch_time = total_time / (epoch + 1)
        remaining_epochs = config.n_epochs + config.n_epochs_decay - (epoch + 1)
        eta = remaining_epochs * avg_epoch_time
        
        print(f'End of epoch {epoch+1} / {config.n_epochs + config.n_epochs_decay}')
        print(f'  Time: {epoch_time:.2f}s | Avg: {avg_epoch_time:.2f}s/epoch | ETA: {eta/3600:.1f}h')
    
    writer.close()
    total_training_time = time.time() - start_time
    print(f'Training completed in {total_training_time/3600:.2f} hours!')


def main():
    parser = argparse.ArgumentParser(description='Train CycleGAN for fluorescent image generation (FAST)')
    
    # Data arguments
    parser.add_argument('--dataroot', type=str, default='./data',
                        help='Path to data directory containing trainA, trainB, testA, testB folders')
    parser.add_argument('--mask_dir', type=str, default=None,
                        help='Path to mask images directory (will be copied to dataroot/trainA and dataroot/testA)')
    parser.add_argument('--fluorescent_dir', type=str, default=None,
                        help='Path to fluorescent images directory (will be copied to dataroot/trainB and dataroot/testB)')
    
    # Output arguments
    parser.add_argument('--results_dir', type=str, default='./results',
                        help='Directory to save training results')
    parser.add_argument('--checkpoints_dir', type=str, default='./checkpoints',
                        help='Directory to save model checkpoints')
    parser.add_argument('--name', type=str, default='fluorescent_cyclegan_fast',
                        help='Name of the experiment (creates subdirectories)')
    
    # Training arguments - Optimized defaults for speed
    parser.add_argument('--n_epochs', type=int, default=50,
                        help='Number of epochs with initial learning rate (reduced for speed)')
    parser.add_argument('--n_epochs_decay', type=int, default=50,
                        help='Number of epochs to linearly decay learning rate to zero (reduced for speed)')
    parser.add_argument('--batch_size', type=int, default=4,
                        help='Input batch size (increased for speed)')
    parser.add_argument('--lr', type=float, default=0.0003,
                        help='Initial learning rate for Adam (increased for faster convergence)')
    parser.add_argument('--image_size', type=int, default=128,
                        help='Scale images to this size (reduced for speed)')
    
    # Saving and logging arguments
    parser.add_argument('--save_epoch_freq', type=int, default=10,
                        help='Frequency of saving checkpoints (in epochs)')
    parser.add_argument('--save_images_freq', type=int, default=10,
                        help='Frequency of saving representative images (in epochs)')
    parser.add_argument('--display_freq', type=int, default=200,
                        help='Frequency of displaying/saving images during training (in iterations)')
    parser.add_argument('--print_freq', type=int, default=50,
                        help='Frequency of printing training losses (in iterations)')
    
    args = parser.parse_args()
    
    # If mask_dir and fluorescent_dir are provided, prepare the data first
    if args.mask_dir and args.fluorescent_dir:
        print("Preparing data from provided directories...")
        from prepare_data import prepare_data
        
        if not os.path.exists(args.mask_dir):
            raise ValueError(f"Mask directory does not exist: {args.mask_dir}")
        if not os.path.exists(args.fluorescent_dir):
            raise ValueError(f"Fluorescent directory does not exist: {args.fluorescent_dir}")
        
        prepare_data(args.mask_dir, args.fluorescent_dir, args.dataroot, train_split=0.8)
        print("Data preparation completed!")
    
    # Validate data directory structure
    required_dirs = ['trainA', 'trainB', 'testA', 'testB']
    for dir_name in required_dirs:
        dir_path = os.path.join(args.dataroot, dir_name)
        if not os.path.exists(dir_path):
            raise ValueError(f"Required directory does not exist: {dir_path}")
        
        # Check if directory has images
        image_files = [f for f in os.listdir(dir_path) if f.lower().endswith(('.png', '.jpg', '.jpeg', '.tif', '.tiff'))]
        if len(image_files) == 0:
            print(f"Warning: No images found in {dir_path}")
    
    print("=== FAST CYCLEGAN TRAINING ===")
    print(f"🚀 Speed optimizations enabled:")
    print(f"  - Reduced image size: {args.image_size}x{args.image_size}")
    print(f"  - Increased batch size: {args.batch_size}")
    print(f"  - Fewer epochs: {args.n_epochs} + {args.n_epochs_decay} decay")
    print(f"  - Lighter architecture: resnet_6blocks, 32 filters")
    print(f"  - Reduced loss weights for faster convergence")
    print(f"  - CUDA optimizations enabled")
    print(f"  - Mixed precision training (if available)")
    print("")
    print(f"📁 Directories:")
    print(f"  Data: {args.dataroot}")
    print(f"  Results: {args.results_dir}")
    print(f"  Checkpoints: {args.checkpoints_dir}")
    print(f"  Experiment: {args.name}")
    print("")
    
    train_cyclegan_fast(args)


if __name__ == '__main__':
    main()
