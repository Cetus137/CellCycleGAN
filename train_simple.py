import torch
import torch.nn as nn
import os
import time
import itertools
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from models.networks import define_G, define_D
from models.losses import ImagePool, GANLoss
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
    
    # Initialize networks
    netG_A = define_G(config.input_nc, config.output_nc, config.ngf, config.netG, 
                      config.norm, not config.no_dropout, config.init_type, config.init_gain, config.gpu_ids)
    netG_B = define_G(config.output_nc, config.input_nc, config.ngf, config.netG,
                      config.norm, not config.no_dropout, config.init_type, config.init_gain, config.gpu_ids)
    netD_A = define_D(config.output_nc, config.ndf, config.netD, config.n_layers_D,
                      config.norm, config.init_type, config.init_gain, config.gpu_ids)
    netD_B = define_D(config.input_nc, config.ndf, config.netD, config.n_layers_D,
                      config.norm, config.init_type, config.init_gain, config.gpu_ids)
    
    # Set up criterion for losses
    criterionGAN = GANLoss(config.gan_mode).to(device)
    criterionCycle = nn.L1Loss()
    criterionIdt = nn.L1Loss()
    
    # Set up optimizers
    optimizer_G = torch.optim.Adam(
        itertools.chain(netG_A.parameters(), netG_B.parameters()),
        lr=config.lr, betas=(config.beta1, 0.999)
    )
    optimizer_D = torch.optim.Adam(
        itertools.chain(netD_A.parameters(), netD_B.parameters()),
        lr=config.lr, betas=(config.beta1, 0.999)
    )
    
    # Initialize image pools for discriminator training
    fake_A_pool = ImagePool(config.pool_size)
    fake_B_pool = ImagePool(config.pool_size)
    
    # Training loop
    total_steps = 0
    
    for epoch in range(config.n_epochs + config.n_epochs_decay):
        epoch_start_time = time.time()
        epoch_iter = 0
        
        # Update learning rate
        if epoch > config.n_epochs:
            lr = config.lr * (config.n_epochs + config.n_epochs_decay - epoch) / config.n_epochs_decay
            for param_group in optimizer_G.param_groups:
                param_group['lr'] = lr
            for param_group in optimizer_D.param_groups:
                param_group['lr'] = lr
        
        # Set to training mode
        netG_A.train()
        netG_B.train()
        netD_A.train()
        netD_B.train()
        
        for i, data in enumerate(tqdm(train_loader, desc=f'Epoch {epoch+1}')):
            total_steps += config.batch_size
            epoch_iter += config.batch_size
            
            # Move data to device
            real_A = data['A'].to(device)
            real_B = data['B'].to(device)
            
            # ================== Update Generators ==================
            # Set discriminators to not require gradients
            for param in netD_A.parameters():
                param.requires_grad = False
            for param in netD_B.parameters():
                param.requires_grad = False
            
            optimizer_G.zero_grad()
            
            # Forward pass
            fake_B = netG_A(real_A)  # Generate fake B from real A
            rec_A = netG_B(fake_B)   # Reconstruct A from fake B
            fake_A = netG_B(real_B)  # Generate fake A from real B
            rec_B = netG_A(fake_A)   # Reconstruct B from fake A
            
            # Identity loss (optional)
            if config.lambda_identity > 0:
                idt_A = netG_A(real_B)
                idt_B = netG_B(real_A)
                loss_idt_A = criterionIdt(idt_A, real_B) * config.lambda_B * config.lambda_identity
                loss_idt_B = criterionIdt(idt_B, real_A) * config.lambda_A * config.lambda_identity
            else:
                loss_idt_A = 0
                loss_idt_B = 0
            
            # GAN losses
            loss_G_A = criterionGAN(netD_A(fake_B), True)
            loss_G_B = criterionGAN(netD_B(fake_A), True)
            
            # Cycle consistency losses
            loss_cycle_A = criterionCycle(rec_A, real_A) * config.lambda_A
            loss_cycle_B = criterionCycle(rec_B, real_B) * config.lambda_B
            
            # Combined generator loss
            loss_G = loss_G_A + loss_G_B + loss_cycle_A + loss_cycle_B + loss_idt_A + loss_idt_B
            loss_G.backward()
            optimizer_G.step()
            
            # ================== Update Discriminators ==================
            # Set discriminators to require gradients
            for param in netD_A.parameters():
                param.requires_grad = True
            for param in netD_B.parameters():
                param.requires_grad = True
            
            optimizer_D.zero_grad()
            
            # Discriminator A
            fake_B_pool_sample = fake_B_pool.query(fake_B)
            pred_real_B = netD_A(real_B)
            pred_fake_B = netD_A(fake_B_pool_sample.detach())
            loss_D_A_real = criterionGAN(pred_real_B, True)
            loss_D_A_fake = criterionGAN(pred_fake_B, False)
            loss_D_A = (loss_D_A_real + loss_D_A_fake) * 0.5
            loss_D_A.backward()
            
            # Discriminator B
            fake_A_pool_sample = fake_A_pool.query(fake_A)
            pred_real_A = netD_B(real_A)
            pred_fake_A = netD_B(fake_A_pool_sample.detach())
            loss_D_B_real = criterionGAN(pred_real_A, True)
            loss_D_B_fake = criterionGAN(pred_fake_A, False)
            loss_D_B = (loss_D_B_real + loss_D_B_fake) * 0.5
            loss_D_B.backward()
            
            optimizer_D.step()
            
            # Logging
            if total_steps % config.print_freq == 0:
                losses = {
                    'G_A': loss_G_A.item(),
                    'G_B': loss_G_B.item(),
                    'D_A': loss_D_A.item(),
                    'D_B': loss_D_B.item(),
                    'Cycle_A': loss_cycle_A.item(),
                    'Cycle_B': loss_cycle_B.item(),
                    'G_total': loss_G.item()
                }
                
                # Log to tensorboard
                for name, value in losses.items():
                    writer.add_scalar(f'Loss/{name}', value, total_steps)
                
                # Print losses
                loss_str = ' | '.join([f'{k}: {v:.4f}' for k, v in losses.items()])
                print(f'Epoch: {epoch+1}, Step: {total_steps}, {loss_str}')
                logger.info(f'Epoch: {epoch+1}, Step: {total_steps}, {loss_str}')
            
            # Save images
            if total_steps % config.display_freq == 0:
                with torch.no_grad():
                    visuals = {
                        'real_A': real_A,
                        'fake_B': fake_B,
                        'rec_A': rec_A,
                        'real_B': real_B,
                        'fake_A': fake_A,
                        'rec_B': rec_B
                    }
                    save_images(visuals, f'{config.results_dir}/{config.name}', epoch, total_steps)
        
        # Save checkpoints
        if (epoch + 1) % config.save_epoch_freq == 0:
            checkpoint = {
                'epoch': epoch,
                'netG_A_state_dict': netG_A.state_dict(),
                'netG_B_state_dict': netG_B.state_dict(),
                'netD_A_state_dict': netD_A.state_dict(),
                'netD_B_state_dict': netD_B.state_dict(),
                'optimizer_G_state_dict': optimizer_G.state_dict(),
                'optimizer_D_state_dict': optimizer_D.state_dict(),
            }
            torch.save(checkpoint, f'{config.checkpoints_dir}/{config.name}/epoch_{epoch+1}.pth')
            torch.save(checkpoint, f'{config.checkpoints_dir}/{config.name}/latest.pth')
            print(f'Saved checkpoint for epoch {epoch+1}')
        
        epoch_time = time.time() - epoch_start_time
        print(f'End of epoch {epoch+1} / {config.n_epochs + config.n_epochs_decay} \t Time Taken: {epoch_time:.2f} sec')
    
    writer.close()
    print('Training completed!')


if __name__ == '__main__':
    train_cyclegan()
