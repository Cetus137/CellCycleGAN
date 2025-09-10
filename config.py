import torch


class Config:
    """Configuration class for CycleGAN training"""
    
    # Data parameters
    dataroot = './data'
    input_nc = 1  # number of input image channels (single-channel masks)
    output_nc = 1  # number of output image channels (single-channel fluorescent)
    image_size = 256
    
    # Model parameters
    ngf = 64  # number of generator filters in the last conv layer
    ndf = 64  # number of discriminator filters in the first conv layer
    netG = 'resnet_9blocks'  # generator architecture
    netD = 'basic'  # discriminator architecture
    n_layers_D = 3  # number of layers in discriminator
    norm = 'instance'  # instance normalization or batch normalization
    init_type = 'normal'  # network initialization
    init_gain = 0.02  # scaling factor for normal initialization
    no_dropout = False  # no dropout for the generator
    
    # Training parameters
    isTrain = True
    gpu_ids = [0] if torch.cuda.is_available() else []
    batch_size = 1
    num_workers = 4
    lr = 0.0002  # initial learning rate for Adam
    beta1 = 0.5  # momentum term of Adam
    gan_mode = 'lsgan'  # GAN loss type
    pool_size = 50  # size of image buffer for discriminator
    
    # Loss weights
    lambda_A = 10.0  # weight for cycle loss (A -> B -> A)
    lambda_B = 10.0  # weight for cycle loss (B -> A -> B)
    lambda_identity = 0.5  # weight for identity loss
    
    # Training schedule
    n_epochs = 100  # number of epochs with the initial learning rate
    n_epochs_decay = 100  # number of epochs to linearly decay learning rate to zero
    lr_policy = 'linear'  # learning rate policy
    lr_decay_iters = 50  # multiply by a gamma every lr_decay_iters iterations
    
    # Logging and saving
    save_epoch_freq = 5  # frequency of saving checkpoints
    save_latest_freq = 5000  # frequency of saving the latest results
    print_freq = 100  # frequency of showing training results
    display_freq = 400  # frequency of displaying images
    
    # Directory settings
    checkpoints_dir = './checkpoints'
    results_dir = './results'
    name = 'fluorescent_cyclegan'  # experiment name
    model = 'cycle_gan'
    direction = 'AtoB'  # AtoB or BtoA
    
    # Testing parameters
    phase = 'test'
    num_test = 50  # number of test images to run
    aspect_ratio = 1.0  # aspect ratio of result images
    
    def __init__(self, **kwargs):
        # Update config with any provided keyword arguments
        for key, value in kwargs.items():
            if hasattr(self, key):
                setattr(self, key, value)
            else:
                print(f"Warning: Unknown config parameter '{key}'")
