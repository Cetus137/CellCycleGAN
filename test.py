import torch
import os
from PIL import Image
import numpy as np
from tqdm import tqdm

from models.cycle_gan_model import CycleGANModel
from data.dataset import create_dataloader
from config import Config
from utils import tensor_to_image, setup_logger


def test_cyclegan():
    """Test the trained CycleGAN model"""
    
    # Initialize configuration
    config = Config(isTrain=False, phase='test')
    
    # Set device
    device = torch.device('cuda:0' if torch.cuda.is_available() and len(config.gpu_ids) > 0 else 'cpu')
    print(f'Using device: {device}')
    
    # Create results directory
    results_dir = f'{config.results_dir}/{config.name}/test'
    os.makedirs(results_dir, exist_ok=True)
    os.makedirs(f'{results_dir}/fake_B', exist_ok=True)  # A->B results
    os.makedirs(f'{results_dir}/fake_A', exist_ok=True)  # B->A results
    
    # Setup logging
    logger = setup_logger(results_dir)
    
    # Create data loader
    test_loader = create_dataloader(
        config.dataroot,
        phase='test',
        batch_size=1,
        num_workers=1,
        image_size=config.image_size
    )
    
    print(f'Test dataset size: {len(test_loader.dataset)}')
    
    # Initialize model
    model = CycleGANModel(config)
    
    # Load trained model
    checkpoint_path = f'{config.checkpoints_dir}/{config.name}/latest.pth'
    if not os.path.exists(checkpoint_path):
        # Try to find the latest epoch checkpoint
        checkpoint_files = [f for f in os.listdir(f'{config.checkpoints_dir}/{config.name}') if f.endswith('.pth')]
        if checkpoint_files:
            checkpoint_path = f'{config.checkpoints_dir}/{config.name}/{sorted(checkpoint_files)[-1]}'
        else:
            raise FileNotFoundError(f"No checkpoint found in {config.checkpoints_dir}/{config.name}")
    
    print(f'Loading model from {checkpoint_path}')
    model.load_networks(checkpoint_path)
    
    # Set to evaluation mode
    model.eval()
    
    # Test loop
    with torch.no_grad():
        for i, data in enumerate(tqdm(test_loader, desc='Testing')):
            if i >= config.num_test:
                break
                
            # Set input and forward
            model.set_input(data)
            model.forward()
            
            # Get results
            visuals = model.get_current_visuals()
            
            # Convert tensors to images and save
            real_A_img = tensor_to_image(visuals['real_A'][0])
            real_B_img = tensor_to_image(visuals['real_B'][0])
            fake_A_img = tensor_to_image(visuals['fake_A'][0])
            fake_B_img = tensor_to_image(visuals['fake_B'][0])
            rec_A_img = tensor_to_image(visuals['rec_A'][0])
            rec_B_img = tensor_to_image(visuals['rec_B'][0])
            
            # Save individual results (handle grayscale)
            if len(fake_B_img.shape) == 2:  # Grayscale
                Image.fromarray(fake_B_img, mode='L').save(f'{results_dir}/fake_B/{i:04d}_fake_B.png')
                Image.fromarray(fake_A_img, mode='L').save(f'{results_dir}/fake_A/{i:04d}_fake_A.png')
            else:  # RGB
                Image.fromarray(fake_B_img).save(f'{results_dir}/fake_B/{i:04d}_fake_B.png')
                Image.fromarray(fake_A_img).save(f'{results_dir}/fake_A/{i:04d}_fake_A.png')
            
            # Create comparison image
            comparison = np.concatenate([
                np.concatenate([real_A_img, fake_B_img, rec_A_img], axis=1),  # A -> B -> A
                np.concatenate([real_B_img, fake_A_img, rec_B_img], axis=1)   # B -> A -> B
            ], axis=0)
            
            Image.fromarray(comparison).save(f'{results_dir}/{i:04d}_comparison.png')
            
            if i % 10 == 0:
                print(f'Processed {i+1} test images')
                logger.info(f'Processed {i+1} test images')
    
    print(f'Testing completed! Results saved to {results_dir}')
    logger.info(f'Testing completed! Results saved to {results_dir}')


if __name__ == '__main__':
    test_cyclegan()
