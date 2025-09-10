import torch
import torch.nn as nn
import os
import time
import itertools
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from models.cycle_gan_model import CycleGANModel
from data.dataset import create_dataloader
from config import Config
from utils import save_images, setup_logger


def train_cyclegan():
    """Main training function for CycleGAN"""
    
    # Initialize configuration
    config = Config()
    
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
        
        epoch_time = time.time() - epoch_start_time
        print(f'End of epoch {epoch+1} / {config.n_epochs + config.n_epochs_decay} \t Time Taken: {epoch_time:.2f} sec')
    
    writer.close()
    print('Training completed!')


if __name__ == '__main__':
    train_cyclegan()
