import torch
from models import VAE, DDPM, DDIM, LDDPM, UNet, VarianceScheduler


def prepare_ddpm() -> DDPM:
    """
    Initializes and returns a DDPM model with a Variance Scheduler and UNet as per the given configurations.
    """
    # Variance Scheduler configuration
    beta1 = 0.0001
    beta2 = 0.02
    num_steps = 1000
    interpolation = 'quadratic'

    in_channels = 1 
    down_channels = [64, 128, 256, 256]  
    up_channels = [ 256,256, 128, 64]  
    time_embed_dim = 128
    num_classes = 10


    var_scheduler = VarianceScheduler(
        beta_start=beta1,
        beta_end=beta2,
        num_steps=num_steps,
        interpolation=interpolation
    )


    network = UNet(
        in_channels=in_channels,
        down_channels=down_channels,
        up_channels=up_channels,
        time_emb_dim=time_embed_dim,
        num_classes=num_classes
    )
    

    ddpm = DDPM(network=network, var_scheduler=var_scheduler)

    return ddpm

def prepare_ddim() -> DDIM:
    """
    EXAMPLE OF INITIALIZING DDIM. Feel free to change the following based on your needs and implementation.
    """
    # TODO: define the configurations of the Variance Scheduler
    beta1 = 0.0001
    beta2 = 0.02
    num_steps = 1000
    interpolation = 'quadratic'

    # TODO: define the configurations of the UNet
    in_channels = 1
    down_channels = [64, 128, 256, 256] 
    up_channels = [ 256,256, 128, 64] 
    time_embed_dim = 128
    num_classes = 10

    # TODO: define the variance scheduler
    var_scheduler = VarianceScheduler(
        beta_start=beta1,
        beta_end=beta2,
        num_steps=num_steps,
        interpolation=interpolation
    )

    # TODO: define the noise estimating UNet
    network = UNet(
        in_channels=in_channels,
        down_channels=down_channels,
        up_channels=up_channels,
        time_emb_dim=time_embed_dim,
        num_classes=num_classes
    )


    ddim = DDIM(network=network, var_scheduler=var_scheduler)

    return ddim

def prepare_vae() -> VAE:
    """
    Initialize and return a VAE instance with pre-defined configurations.
    """
    # Configuration parameters
    in_channels = 1        
    height = 32            
    width = 32              
    mid_channels = [64, 128, 256, 512, 1024] 
    latent_dim = 1        
    num_classes = 10       

    # Create and return the VAE
    vae = VAE(
        in_channels=in_channels,
        height=height,
        width=width,
        mid_channels=mid_channels,
        latent_dim=latent_dim,
        num_classes=num_classes
    )
    return vae

def prepare_lddpm() -> LDDPM:
    """
    EXAMPLE OF INITIALIZING LDDPM. Feel free to change the following based on your needs and implementation.
    """
    # VAE configs (must match those used in prepare_vae())
    in_channels = 1
    mid_channels = [64, 128, 256, 512]
    height = width = 32
    latent_dim = 1
    num_classes = 10
    vae = VAE(in_channels=in_channels,
              mid_channels=mid_channels,
              height=height,
              width=width,
              latent_dim=latent_dim,
              num_classes=num_classes)
    
    # Load pre-trained VAE weights
    vae.load_state_dict(torch.load('checkpoints/VAE.pt'))

    # Variance scheduler configs
    beta1 = 0.0001
    beta2 = 0.02
    num_steps = 1000
    interpolation = 'quadratic'

    # Define the variance scheduler
    var_scheduler = VarianceScheduler(beta_start=beta1, beta_end=beta2, num_steps=num_steps, interpolation=interpolation)

    # Diffusion UNet configs (not more than 2 downsampling layers)
    ddpm_in_channels = latent_dim
    down_channels = [256, 512]  # At most 2 downsampling layers
    up_channels = [512, 256]
    time_embed_dim = 128

    # Define the UNet for the diffusion model
    network = UNet(in_channels=ddpm_in_channels, 
                   down_channels=down_channels, 
                   up_channels=up_channels, 
                   time_emb_dim=time_embed_dim,
                   num_classes=num_classes)

    # Initialize LDDPM with the defined components
    lddpm = LDDPM(network=network, vae=vae, var_scheduler=var_scheduler)

    return lddpm
