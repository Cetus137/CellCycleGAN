import torch
import torch.nn as nn
import torch.nn.functional as F
import itertools
from .networks import define_G, define_D, init_net
from .losses import GANLoss, ImagePool
# Spatial correspondence loss for boundary preservation
class SpatialCorrespondenceLoss:
    """Distance-based spatial loss for gentle boundary guidance"""
    
    def __init__(self, lambda_spatial=0.05):  # Much lower weight
        self.lambda_spatial = lambda_spatial
    
    def create_distance_guidance(self, mask):
        """Create distance-based guidance maps from masks"""
        import cv2
        import numpy as np
        
        # Convert to numpy for OpenCV processing
        mask_np = mask.detach().cpu().numpy()
        batch_size = mask_np.shape[0]
        
        guidance_maps = []
        for i in range(batch_size):
            # Get single channel mask
            single_mask = mask_np[i, 0]
            
            # Normalize to 0-255 and convert to uint8
            mask_norm = (single_mask - single_mask.min()) / (single_mask.max() - single_mask.min() + 1e-8)
            mask_uint8 = (mask_norm * 255).astype(np.uint8)
            
            # Create binary mask from threshold
            _, binary = cv2.threshold(mask_uint8, 127, 255, cv2.THRESH_BINARY)
            
            # Compute distance transform from boundaries
            distance_inside = cv2.distanceTransform(binary, cv2.DIST_L2, 5)
            distance_outside = cv2.distanceTransform(255 - binary, cv2.DIST_L2, 5)
            
            # Create signed distance (positive inside, negative outside)
            signed_distance = distance_inside - distance_outside
            
            # Normalize to create gentle guidance field
            if np.abs(signed_distance).max() > 0:
                guidance = signed_distance / (np.abs(signed_distance).max() + 1e-8)
            else:
                guidance = np.zeros_like(signed_distance)
            
            guidance_maps.append(guidance)
        
        # Convert back to tensor
        guidance_tensor = torch.from_numpy(np.stack(guidance_maps)).float().unsqueeze(1).to(mask.device)
        return guidance_tensor
    
    def gentle_structure_loss(self, pred, target_mask):
        """Compute structural guidance loss with boundary focus"""
        try:
            # Create guidance fields from mask
            target_guidance = self.create_distance_guidance(target_mask)
            pred_guidance = self.create_distance_guidance(pred)
            
            # Distance transform loss for overall structure
            structure_loss = F.smooth_l1_loss(pred_guidance, target_guidance, reduction='mean')
            
            # Add boundary-focused component - penalize when fluorescent signal doesn't align with mask boundaries
            # Create binary masks for boundary detection
            target_binary = (target_mask > 0.5).float()
            pred_binary = (pred > 0.5).float()
            
            # Simple boundary alignment: encourage fluorescent signal to be higher inside mask regions
            boundary_alignment = F.mse_loss(pred * target_binary, target_binary * 0.7)  # Encourage ~70% intensity in mask regions
            
            return structure_loss + 0.5 * boundary_alignment
            
        except Exception as e:
            # Fallback: boundary alignment loss only
            target_binary = (target_mask > 0.5).float()
            return F.mse_loss(pred * target_binary, target_binary * 0.5) * 0.5
    
    def structural_alignment_loss(self, pred, target):
        """Direct structural alignment - bright areas should align with mask regions"""
        # Normalize both to [0,1]
        pred_norm = (pred - pred.min()) / (pred.max() - pred.min() + 1e-8)
        target_norm = (target - target.min()) / (target.max() - target.min() + 1e-8)
        
        # Create binary masks
        target_mask = (target_norm > 0.5).float()
        
        # Penalize bright fluorescent outside mask regions
        outside_mask_penalty = torch.mean(pred_norm * (1 - target_mask))
        
        # Encourage bright fluorescent inside mask regions
        inside_mask_reward = torch.mean((1 - pred_norm) * target_mask)
        
        return outside_mask_penalty + inside_mask_reward
    
    def __call__(self, real_mask, fake_fluor):
        """Compute strong spatial correspondence loss for direct correlation"""
        # Direct correlation loss: fluorescent intensity should follow mask pattern
        mask_binary = (real_mask > 0.5).float()  # Binary mask regions
        
        # Primary loss: fluorescent signal should be high inside mask, low outside
        inside_loss = F.mse_loss(fake_fluor * mask_binary, mask_binary * 0.8)  # Target 80% intensity inside
        outside_loss = F.mse_loss(fake_fluor * (1 - mask_binary), torch.zeros_like(fake_fluor) * (1 - mask_binary))  # Target low intensity outside
        
        # Boundary preservation loss
        gentle_loss = self.gentle_structure_loss(fake_fluor, real_mask)
        
        # Combine losses with emphasis on direct correlation
        total_loss = 3.0 * inside_loss + 2.0 * outside_loss + 0.5 * gentle_loss  # reduced structural component
        return self.lambda_spatial * total_loss
import torch.nn as nn


class CycleGANModel(nn.Module):
    """
    This class implements the CycleGAN model, for learning image-to-image translation without paired data.
    
    CycleGAN paper: https://arxiv.org/pdf/1703.10593.pdf
    """
    def __init__(self, opt):
        """Initialize the CycleGAN class.

        Parameters:
            opt (Option class)-- stores all the experiment flags; needs to be a subclass of BaseOptions
        """
        super(CycleGANModel, self).__init__()
        
        self.opt = opt
        self.device = torch.device('cuda:0' if torch.cuda.is_available() and len(opt.gpu_ids) > 0 else 'cpu')
        
        # specify the training losses you want to print out. The training/test scripts will call <BaseModel.get_current_losses>
        self.loss_names = ['D_A', 'G_A', 'cycle_A', 'idt_A', 'D_B', 'G_B', 'cycle_B', 'idt_B', 'spatial_A', 'spatial_B']
        # specify the images you want to save/display. The training/test scripts will call <BaseModel.get_current_visuals>
        visual_names_A = ['real_A', 'fake_B', 'rec_A']
        visual_names_B = ['real_B', 'fake_A', 'rec_B']
        if opt.lambda_identity > 0.0:  # if identity loss is used, we also visualize idt_B=G_A(B) ad idt_A=G_A(B)
            visual_names_A.append('idt_B')
            visual_names_B.append('idt_A')

        self.visual_names = visual_names_A + visual_names_B  # combine visualizations for A and B
        # specify the models you want to save to the disk. The training/test scripts will call <BaseModel.save_networks> and <BaseModel.load_networks>.
        if opt.isTrain:
            self.model_names = ['G_A', 'G_B', 'D_A', 'D_B']
        else:  # during test time, only load Gs
            self.model_names = ['G_A', 'G_B']

        # define networks (both Generators and discriminators)
        # The naming is different from those used in the paper.
        # Code (vs. paper): G_A (G), G_B (F), D_A (D_Y), D_B (D_X)
        self.netG_A = define_G(opt.input_nc, opt.output_nc, opt.ngf, opt.netG, opt.norm,
                                        not opt.no_dropout, opt.init_type, opt.init_gain, opt.gpu_ids)
        self.netG_B = define_G(opt.output_nc, opt.input_nc, opt.ngf, opt.netG, opt.norm,
                                        not opt.no_dropout, opt.init_type, opt.init_gain, opt.gpu_ids)

        if opt.isTrain:  # define discriminators
            self.netD_A = define_D(opt.output_nc, opt.ndf, opt.netD,
                                            opt.n_layers_D, opt.norm, opt.init_type, opt.init_gain, opt.gpu_ids)
            self.netD_B = define_D(opt.input_nc, opt.ndf, opt.netD,
                                            opt.n_layers_D, opt.norm, opt.init_type, opt.init_gain, opt.gpu_ids)

            # define loss functions
            self.criterionGAN = GANLoss(opt.gan_mode).to(self.device)  # define GAN loss.
            self.criterionCycle = torch.nn.L1Loss()
            self.criterionIdt = torch.nn.L1Loss()
            self.criterionSpatial = SpatialCorrespondenceLoss(lambda_spatial=1.0)  # strong spatial correspondence loss for correlation
            
            # initialize optimizers; schedulers will be automatically created by function <BaseModel.setup>.
            self.optimizer_G = torch.optim.Adam(itertools.chain(self.netG_A.parameters(), self.netG_B.parameters()), lr=opt.lr, betas=(opt.beta1, 0.999))
            self.optimizer_D = torch.optim.Adam(itertools.chain(self.netD_A.parameters(), self.netD_B.parameters()), lr=opt.lr, betas=(opt.beta1, 0.999))
            self.optimizers = [self.optimizer_G, self.optimizer_D]
            
            # Initialize image pools
            self.fake_A_pool = ImagePool(opt.pool_size)
            self.fake_B_pool = ImagePool(opt.pool_size)

    def set_input(self, input):
        """Unpack input data from the dataloader and perform necessary pre-processing steps.

        Parameters:
            input (dict): include the data itself and its metadata information.

        The option 'direction' can be used to swap domain A and domain B.
        """
        AtoB = self.opt.direction == 'AtoB'
        self.real_A = input['A' if AtoB else 'B'].to(self.device)
        self.real_B = input['B' if AtoB else 'A'].to(self.device)
        self.image_paths = input['A_paths' if AtoB else 'B_paths']

    def forward(self):
        """Run forward pass; called by both functions <optimize_parameters> and <test>."""
        self.fake_B = self.netG_A(self.real_A)  # G_A(A)
        self.rec_A = self.netG_B(self.fake_B)   # G_B(G_A(A))
        self.fake_A = self.netG_B(self.real_B)  # G_B(B)
        self.rec_B = self.netG_A(self.fake_A)   # G_A(G_B(B))

    def backward_D_basic(self, netD, real, fake):
        """Calculate GAN loss for the discriminator

        Parameters:
            netD (network)      -- the discriminator D
            real (tensor array) -- real images
            fake (tensor array) -- images generated by a generator

        Return the discriminator loss.
        We also call loss_D.backward() to calculate the gradients.
        """
        # Real
        pred_real = netD(real)
        loss_D_real = self.criterionGAN(pred_real, True)
        # Fake
        pred_fake = netD(fake.detach())
        loss_D_fake = self.criterionGAN(pred_fake, False)
        # Combined loss and calculate gradients
        loss_D = (loss_D_real + loss_D_fake) * 0.5
        loss_D.backward()
        return loss_D

    def backward_D_A(self):
        """Calculate GAN loss for discriminator D_A"""
        fake_B = self.fake_B_pool.query(self.fake_B)
        self.loss_D_A = self.backward_D_basic(self.netD_A, self.real_B, fake_B)

    def backward_D_B(self):
        """Calculate GAN loss for discriminator D_B"""
        fake_A = self.fake_A_pool.query(self.fake_A)
        self.loss_D_B = self.backward_D_basic(self.netD_B, self.real_A, fake_A)

    def backward_G(self):
        """Calculate the loss for generators G_A and G_B"""
        lambda_idt = self.opt.lambda_identity
        lambda_A = self.opt.lambda_A
        lambda_B = self.opt.lambda_B
        # Identity loss
        if lambda_idt > 0:
            # G_A should be identity if real_B is fed: ||G_A(B) - B||
            self.idt_A = self.netG_A(self.real_B)
            self.loss_idt_A = self.criterionIdt(self.idt_A, self.real_B) * lambda_B * lambda_idt
            # G_B should be identity if real_A is fed: ||G_B(A) - A||
            self.idt_B = self.netG_B(self.real_A)
            self.loss_idt_B = self.criterionIdt(self.idt_B, self.real_A) * lambda_A * lambda_idt
        else:
            self.loss_idt_A = 0
            self.loss_idt_B = 0

        # GAN loss D_A(G_A(A))
        self.loss_G_A = self.criterionGAN(self.netD_A(self.fake_B), True)
        # GAN loss D_B(G_B(B))
        self.loss_G_B = self.criterionGAN(self.netD_B(self.fake_A), True)
        # Forward cycle loss || G_B(G_A(A)) - A||
        self.loss_cycle_A = self.criterionCycle(self.rec_A, self.real_A) * lambda_A
        # Backward cycle loss || G_A(G_B(B)) - B||
        self.loss_cycle_B = self.criterionCycle(self.rec_B, self.real_B) * lambda_B
        
        # Spatial correspondence loss to enforce strong correlation
        self.loss_spatial_A = self.criterionSpatial(self.real_A, self.fake_B)  # mask -> fluor correspondence (primary)
        self.loss_spatial_B = self.criterionSpatial(self.real_B, self.fake_A) * 0.5  # fluor -> mask correspondence (secondary)
        
        # combined loss and calculate gradients
        self.loss_G = (self.loss_G_A + self.loss_G_B + self.loss_cycle_A + self.loss_cycle_B + 
                       self.loss_idt_A + self.loss_idt_B + self.loss_spatial_A + self.loss_spatial_B)
        self.loss_G.backward()

    def optimize_parameters(self):
        """Calculate losses, gradients, and update network weights; called in every training iteration"""
        # forward
        self.forward()      # compute fake images and reconstruction images.
        # G_A and G_B
        self.set_requires_grad([self.netD_A, self.netD_B], False)  # Ds require no gradients when optimizing Gs
        self.optimizer_G.zero_grad()  # set G_A and G_B's gradients to zero
        self.backward_G()             # calculate gradients for G_A and G_B
        self.optimizer_G.step()       # update G_A and G_B's weights
        # D_A and D_B
        self.set_requires_grad([self.netD_A, self.netD_B], True)
        self.optimizer_D.zero_grad()   # set D_A and D_B's gradients to zero
        self.backward_D_A()      # calculate gradients for D_A
        self.backward_D_B()      # calculate graidents for D_B
        self.optimizer_D.step()  # update D_A and D_B's weights

    def get_current_losses(self):
        """Return traning losses / errors. train.py will print out these errors on console, and save them to a file"""
        errors_ret = {}
        for name in self.loss_names:
            if isinstance(name, str):
                errors_ret[name] = float(getattr(self, 'loss_' + name))
        return errors_ret

    def get_current_visuals(self):
        """Return visualization images. train.py will display these images with wandb, and save them to a HTML"""
        visual_ret = {}
        for name in self.visual_names:
            if isinstance(name, str):
                visual_ret[name] = getattr(self, name)
        return visual_ret

    def save_networks(self, save_dir, epoch):
        """Save all the networks to the disk."""
        checkpoint = {
            'epoch': epoch,
            'netG_A_state_dict': self.netG_A.state_dict(),
            'netG_B_state_dict': self.netG_B.state_dict(),
            'optimizer_G_state_dict': self.optimizer_G.state_dict(),
        }
        if hasattr(self, 'netD_A'):
            checkpoint['netD_A_state_dict'] = self.netD_A.state_dict()
            checkpoint['netD_B_state_dict'] = self.netD_B.state_dict()
            checkpoint['optimizer_D_state_dict'] = self.optimizer_D.state_dict()
        
        torch.save(checkpoint, f'{save_dir}/epoch_{epoch}.pth')
        torch.save(checkpoint, f'{save_dir}/latest.pth')

    def load_networks(self, load_path):
        """Load all the networks from the disk."""
        checkpoint = torch.load(load_path, map_location=self.device)
        self.netG_A.load_state_dict(checkpoint['netG_A_state_dict'])
        self.netG_B.load_state_dict(checkpoint['netG_B_state_dict'])
        
        if hasattr(self, 'netD_A') and 'netD_A_state_dict' in checkpoint:
            self.netD_A.load_state_dict(checkpoint['netD_A_state_dict'])
            self.netD_B.load_state_dict(checkpoint['netD_B_state_dict'])
        
        if hasattr(self, 'optimizer_G') and 'optimizer_G_state_dict' in checkpoint:
            self.optimizer_G.load_state_dict(checkpoint['optimizer_G_state_dict'])
        if hasattr(self, 'optimizer_D') and 'optimizer_D_state_dict' in checkpoint:
            self.optimizer_D.load_state_dict(checkpoint['optimizer_D_state_dict'])
        
        return checkpoint.get('epoch', 0)

    def set_requires_grad(self, nets, requires_grad=False):
        """Set requies_grad=Fasle for all the networks to avoid unnecessary computations
        Parameters:
            nets (network list)   -- a list of networks
            requires_grad (bool)  -- whether the networks require gradients or not
        """
        if not isinstance(nets, list):
            nets = [nets]
        for net in nets:
            if net is not None:
                for param in net.parameters():
                    param.requires_grad = requires_grad
