import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import os
import numpy as np
import albumentations as A
from albumentations.pytorch import ToTensorV2
import cv2


class FluorescentDataset(Dataset):
    """Dataset for fluorescent image generation from synthetic masks"""
    
    def __init__(self, root_dir, phase='train', transform=None, mode='A'):
        """
        Args:
            root_dir (string): Directory with all the images.
            phase (string): 'train' or 'test'
            transform (callable, optional): Optional transform to be applied on a sample.
            mode (string): 'A' for synthetic masks, 'B' for fluorescent images
        """
        self.root_dir = root_dir
        self.phase = phase
        self.mode = mode
        self.transform = transform
        
        # Set up paths
        if mode == 'A':
            self.image_dir = os.path.join(root_dir, f'{phase}A')
        else:
            self.image_dir = os.path.join(root_dir, f'{phase}B')
            
        # Get all image files
        self.image_files = []
        if os.path.exists(self.image_dir):
            for file in os.listdir(self.image_dir):
                if file.lower().endswith(('.png', '.jpg', '.jpeg', '.tif', '.tiff')):
                    self.image_files.append(file)
        
        self.image_files.sort()
        
    def __len__(self):
        return len(self.image_files)
    
    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
            
        img_path = os.path.join(self.image_dir, self.image_files[idx])
        
        try:
            # Load image - use PIL for TIFF files to handle 32-bit images
            if img_path.lower().endswith(('.tif', '.tiff')):
                # Use PIL for TIFF files (handles 32-bit images better)
                with Image.open(img_path) as pil_img:
                    # Get image as numpy array first
                    image = np.array(pil_img)
                    
                    # Handle different bit depths
                    if image.dtype == np.float32 or image.dtype == np.float64:
                        # 32-bit float images - normalize to 0-255
                        image = ((image - image.min()) / (image.max() - image.min()) * 255)
                        image = image.astype(np.uint8)
                    elif image.dtype == np.uint16:
                        # 16-bit images - scale down to 8-bit
                        original_min, original_max = image.min(), image.max()
                        image = ((image - original_min) * (255.0 / (original_max - original_min))).astype(np.uint8)
                    elif image.dtype == np.uint32:
                        # 32-bit integer images - scale down to 8-bit
                        image = (image / (2**24)).astype(np.uint8)
                    
                    # Ensure grayscale
                    if len(image.shape) == 3:
                        image = np.mean(image, axis=2).astype(np.uint8)
            else:
                # Use OpenCV for other formats
                image = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
                if image is None:
                    # Fallback to PIL
                    image = np.array(Image.open(img_path).convert('L'))
        except Exception as e:
            print(f"Error loading image {img_path}: {e}")
            # Create a black image as fallback
            image = np.zeros((256, 256), dtype=np.uint8)
        
        # Ensure image is 2D (H, W) for single channel
        if len(image.shape) == 3:
            image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        
        # Normalize to 0-255 range if needed (for 32-bit images)
        if image.dtype != np.uint8:
            # Normalize to 0-255 range
            image = ((image - image.min()) / (image.max() - image.min()) * 255).astype(np.uint8)
        
        # Add channel dimension for albumentations (H, W) -> (H, W, 1)
        image = np.expand_dims(image, axis=2)
        
        # Apply transforms
        if self.transform:
            transformed = self.transform(image=image)
            image = transformed['image']
        
        return image


class UnpairedDataset(Dataset):
    """Dataset for unpaired training (CycleGAN)"""
    
    def __init__(self, root_dir, phase='train', transform_A=None, transform_B=None):
        """
        Args:
            root_dir (string): Directory with all the images.
            phase (string): 'train' or 'test'
            transform_A (callable, optional): Transform for domain A (masks)
            transform_B (callable, optional): Transform for domain B (fluorescent)
        """
        self.dataset_A = FluorescentDataset(root_dir, phase, transform_A, 'A')
        self.dataset_B = FluorescentDataset(root_dir, phase, transform_B, 'B')
        
        # Use the larger dataset size for unpaired training
        self.size_A = len(self.dataset_A)
        self.size_B = len(self.dataset_B)
        
    def __len__(self):
        return max(self.size_A, self.size_B)
    
    def __getitem__(self, idx):
        # Get images from both domains
        idx_A = idx % self.size_A
        idx_B = idx % self.size_B
        
        image_A = self.dataset_A[idx_A]
        image_B = self.dataset_B[idx_B]
        
        return {
            'A': image_A,
            'B': image_B,
            'A_paths': self.dataset_A.image_files[idx_A],
            'B_paths': self.dataset_B.image_files[idx_B]
        }


def get_transforms(image_size=256, is_train=True):
    """Get data transforms for training or testing"""
    
    if is_train:
        transform_list = [
            A.Resize(height=image_size, width=image_size),
            A.HorizontalFlip(p=0.5),
            A.RandomBrightnessContrast(p=0.2),
            A.Normalize(mean=[0.5], std=[0.5]),  # Single channel normalization
            ToTensorV2()
        ]
    else:
        transform_list = [
            A.Resize(height=image_size, width=image_size),
            A.Normalize(mean=[0.5], std=[0.5]),  # Single channel normalization
            ToTensorV2()
        ]
    
    return A.Compose(transform_list)


def create_dataloader(root_dir, phase='train', batch_size=1, num_workers=4, image_size=256):
    """Create dataloader for CycleGAN training/testing"""
    
    is_train = (phase == 'train')
    transform = get_transforms(image_size, is_train)
    
    dataset = UnpairedDataset(
        root_dir=root_dir,
        phase=phase,
        transform_A=transform,
        transform_B=transform
    )
    
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=is_train,
        num_workers=num_workers,
        drop_last=is_train
    )
    
    return dataloader
