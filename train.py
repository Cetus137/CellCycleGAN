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
from config import Config
from utils import save_images, setup_logger


def train_cyclegan(args):
    """Main training function for CycleGAN"""
    
    # Initialize configuration with command line arguments
    config = Config(
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
    
    # Set device
    device = torch.device('cuda:0' if torch.cuda.is_available() and len(config.gpu_ids) > 0 else 'cpu')
    print(f'Using device: {device}')
    
    # Create directories
    os.makedirs(config.checkpoints_dir, exist_ok=True)
    os.makedirs(config.results_dir, exist_ok=True)
    os.makedirs(f'{config.checkpoints_dir}/{config.name}', exist_ok=True)
    
    # Setup logging
    logger = setup_logger(f'{config.checkpoints_dir}/{config.name}')
    writer = SummaryWriter(f'{config.checkpoints_dir}/{config.name}/logs')
    
    # Create data loaders
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
    
    # Initialize model
    model = CycleGANModel(config)
    
    # Training loop
    total_steps = 0
    
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
        
        for i, data in enumerate(tqdm(train_loader, desc=f'Epoch {epoch+1}')):
            total_steps += config.batch_size
            epoch_iter += config.batch_size
            
            # Set input and optimize
            model.set_input(data)
            model.optimize_parameters()
            
            # Logging
            if total_steps % config.print_freq == 0:
                losses = model.get_current_losses()
                
                # Log to tensorboard
                for name, value in losses.items():
                    writer.add_scalar(f'Loss/{name}', value, total_steps)
                
                # Print losses
                loss_str = ' | '.join([f'{k}: {v:.4f}' for k, v in losses.items()])
                print(f'Epoch: {epoch+1}, Step: {total_steps}, {loss_str}')
                logger.info(f'Epoch: {epoch+1}, Step: {total_steps}, {loss_str}')
            
            # Save images
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
                
                # Save representative images
                save_images(visuals, representative_dir, epoch, epoch+1)
                print(f'Representative images saved to {representative_dir}')
            model.train()
        
        epoch_time = time.time() - epoch_start_time
        print(f'End of epoch {epoch+1} / {config.n_epochs + config.n_epochs_decay} \t Time Taken: {epoch_time:.2f} sec')
    
    writer.close()
    print('Training completed!')


def main():
    parser = argparse.ArgumentParser(description='Train CycleGAN for fluorescent image generation')
    
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
    parser.add_argument('--name', type=str, default='fluorescent_cyclegan',
                        help='Name of the experiment (creates subdirectories)')
    
    # Training arguments
    parser.add_argument('--n_epochs', type=int, default=100,
                        help='Number of epochs with initial learning rate')
    parser.add_argument('--n_epochs_decay', type=int, default=100,
                        help='Number of epochs to linearly decay learning rate to zero')
    parser.add_argument('--batch_size', type=int, default=1,
                        help='Input batch size')
    parser.add_argument('--lr', type=float, default=0.0002,
                        help='Initial learning rate for Adam')
    parser.add_argument('--image_size', type=int, default=256,
                        help='Scale images to this size')
    
    # Saving and logging arguments
    parser.add_argument('--save_epoch_freq', type=int, default=5,
                        help='Frequency of saving checkpoints (in epochs)')
    parser.add_argument('--save_images_freq', type=int, default=10,
                        help='Frequency of saving representative images (in epochs)')
    parser.add_argument('--display_freq', type=int, default=400,
                        help='Frequency of displaying/saving images during training (in iterations)')
    parser.add_argument('--print_freq', type=int, default=100,
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
    
    print(f"Starting training with:")
    print(f"  Data directory: {args.dataroot}")
    print(f"  Results directory: {args.results_dir}")
    print(f"  Checkpoints directory: {args.checkpoints_dir}")
    print(f"  Experiment name: {args.name}")
    print(f"  Epochs: {args.n_epochs} + {args.n_epochs_decay} decay")
    print(f"  Representative images saved every {args.save_images_freq} epochs")
    
    train_cyclegan(args)


if __name__ == '__main__':
    main()
