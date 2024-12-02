import torch
import math

import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt

from typing import List
from typing import Tuple



class VarianceScheduler:
    def __init__(self, beta_start: int=0.0001, beta_end: int=0.02, num_steps: int=1000, interpolation: str='linear') -> None:
        self.num_steps = num_steps

        # find the beta valuess by linearly interpolating from start beta to end beta
        if interpolation == 'linear':
            self.betas = torch.linspace(beta_start, beta_end, num_steps)
        elif interpolation == 'quadratic':
            self.betas = torch.linspace(math.sqrt(beta_start), math.sqrt(beta_end), num_steps) ** 2
        else:
            raise Exception('[!] Error: invalid beta interpolation encountered...')

        self.alphas = 1 - self.betas
        self.alpha_bars = torch.cumprod(self.alphas, dim=0)

    def add_noise(self, x:torch.Tensor, time_step:torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        device = x.device

        # TODO: sample a random noise
        noise = torch.randn_like(x, device=device)
        time_step = time_step.view(-1)
        alpha_bar_t = self.alpha_bars.to(device)[time_step].view(-1, 1, 1, 1)


        # TODO: construct the noisy sample
        noisy_input = torch.sqrt(alpha_bar_t) * x + torch.sqrt(1 - alpha_bar_t) * noise

        return noisy_input, noise


class SinusoidalPositionEmbeddings(nn.Module):
    def __init__(self, dim: int) -> None:
      super().__init__()

      self.dim = dim

    def forward(self, time: torch.Tensor) -> torch.Tensor:
        device = time.device
        half_dim = self.dim // 2
        emb_scale = torch.exp(-torch.arange(half_dim, device=device).float() * math.log(10_000) / half_dim)
        scaled_time = time[:, None].float() * emb_scale
        embeddings = torch.cat([torch.sin(scaled_time), torch.cos(scaled_time)], dim=-1)

        return embeddings

class UNet(nn.Module):
    def __init__(self, in_channels: int = 1,
                 down_channels: List[int] = [64, 128, 256, 512],
                 up_channels: List[int] = [512, 256, 128, 64],
                 time_emb_dim: int = 128,
                 num_classes: int = 10):
        super().__init__()

        self.num_classes = num_classes
        self.time_emb_dim = time_emb_dim


        self.time_mlp = nn.Sequential(
            nn.Linear(time_emb_dim, time_emb_dim),
            nn.GELU(),
            nn.Linear(time_emb_dim, time_emb_dim)
        )
        self.class_emb = nn.Embedding(num_classes, time_emb_dim)

        adjusted_in_channels = in_channels + time_emb_dim


        self.downs = nn.ModuleList()
        prev_channels = adjusted_in_channels
        for out_channels in down_channels:
            self.downs.append(self._down_block(prev_channels, out_channels))
            prev_channels = out_channels


        self.bottleneck = nn.Sequential(
            nn.Conv2d(prev_channels, prev_channels, kernel_size=3, padding=1),
            nn.GELU(),
            nn.BatchNorm2d(prev_channels),
            nn.Conv2d(prev_channels, prev_channels, kernel_size=3, padding=1),
            nn.GELU()
        )


        self.ups = nn.ModuleList()
        for out_channels in up_channels:
            self.ups.append(self._up_block(prev_channels + out_channels, out_channels))
            prev_channels = out_channels

        self.final_conv = nn.Conv2d(prev_channels, 1, kernel_size=1)

    def forward(self, x: torch.Tensor, timestep: torch.Tensor, label: torch.Tensor) -> torch.Tensor:
        t_emb = SinusoidalPositionEmbeddings(self.time_emb_dim).to(x.device)(timestep)
        t_emb = self.time_mlp(t_emb)
        l_emb = self.class_emb(label)
        context = t_emb + l_emb

        context = context.view(context.size(0), -1, 1, 1).expand(-1, -1, x.size(2), x.size(3))
        x = torch.cat([x, context], dim=1)  

        skips = []
        for down in self.downs:
            x = down(x)
            skips.append(x)
            x = F.max_pool2d(x, kernel_size=2)

        x = self.bottleneck(x)

        for up, skip in zip(self.ups, reversed(skips)):
            x = F.interpolate(x, scale_factor=2, mode='nearest')
            x = torch.cat([x, skip], dim=1)
            x = up(x)

        return self.final_conv(x)

    def _down_block(self, in_channels: int, out_channels: int) -> nn.Sequential:
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.GroupNorm(num_groups=32, num_channels=out_channels),
            nn.GELU(),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.GroupNorm(num_groups=32, num_channels=out_channels),
            nn.GELU(),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.GroupNorm(num_groups=32, num_channels=out_channels),
            nn.GELU()
        )

    def _up_block(self, in_channels: int, out_channels: int) -> nn.Sequential:
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.GELU(),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.GroupNorm(num_groups=32, num_channels=out_channels),
            nn.GELU(),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.GroupNorm(num_groups=32, num_channels=out_channels),
            nn.GELU()
        )


class VAE(nn.Module):
    def __init__(self, 
                 in_channels: int, 
                 height: int=32, 
                 width: int=32, 
                 mid_channels: List=[64, 128, 256, 512, 1024],  
                 latent_dim: int=1, 
                 num_classes: int=10) -> None:
        
        super().__init__()

        self.height = height
        self.width = width
        self.in_channels = in_channels
        self.latent_dim = latent_dim
        self.num_classes = num_classes

        
        self.mid_size = [mid_channels[-1], height // (2 ** (len(mid_channels)-1)), width // (2 ** (len(mid_channels)-1))]

        
        self.class_emb = nn.Embedding(num_classes, self.mid_size[0] * self.mid_size[1] * self.mid_size[2])
        
        
        self.encoder = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels[0], kernel_size=4, stride=2, padding=1),  
            nn.ReLU(),
            nn.Conv2d(mid_channels[0], mid_channels[1], kernel_size=4, stride=2, padding=1), 
            nn.ReLU(),
            nn.Conv2d(mid_channels[1], mid_channels[2], kernel_size=4, stride=2, padding=1), 
            nn.ReLU(),
            nn.Conv2d(mid_channels[2], mid_channels[3], kernel_size=4, stride=2, padding=1), 
            nn.ReLU(),
            nn.Conv2d(mid_channels[3], mid_channels[4], kernel_size=1, stride=1, padding=0),  
            nn.ReLU()
        )
        
        self.mean_net = nn.Linear(mid_channels[-1] * self.mid_size[1] * self.mid_size[2], latent_dim)
        self.logvar_net = nn.Linear(mid_channels[-1] * self.mid_size[1] * self.mid_size[2], latent_dim)


        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(latent_dim + mid_channels[-1], mid_channels[-2], kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(mid_channels[-2], mid_channels[-3], kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(mid_channels[-3], mid_channels[-4], kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(mid_channels[-4], mid_channels[-5], kernel_size=4, stride=2, padding=1), 
            nn.ReLU(),
            nn.Conv2d(mid_channels[-5], in_channels, kernel_size=3, padding=1),
            nn.Sigmoid()
        )
    

    def forward(self, x: torch.Tensor, label: torch.Tensor) -> torch.Tensor:
        # TODO: compute the output of the network encoder
        out = self.encoder(x).view(x.size(0), -1)

        # TODO: estimating mean and logvar
        mean = self.mean_net(out)
        logvar = self.logvar_net(out)
        
        # TODO: computing a sample from the latent distribution
        sample = self.reparameterize(mean, logvar)

        # TODO: decoding the sample
        label_emb = self.class_emb(label).view(x.size(0), *self.mid_size)
        sample = sample.unsqueeze(-1).unsqueeze(-1) 
        sample = sample.expand(-1, -1, self.mid_size[1], self.mid_size[2])
        com = torch.cat([sample, label_emb], dim=1)
        out = self.decode(com, label)

        return out, mean, logvar

    def reparameterize(self, mean: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        # TODO: implement the reparameterization trick: sample = noise * std + mean
        std = torch.exp(0.5 * logvar)  
        noise = torch.randn_like(std) 
        sample = noise * std + mean

        return sample
    
    @staticmethod
    def reconstruction_loss(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        # TODO: compute the binary cross entropy between the pred (reconstructed image) and the traget (ground truth image)
        loss = F.binary_cross_entropy(pred, target, reduction='sum')

        return loss
       
    @staticmethod
    def kl_loss(mean: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        # TODO: compute the KL divergence
        kl_div = -.5 * (logvar.flatten(start_dim=1) + 1 - torch.exp(logvar.flatten(start_dim=1)) - mean.flatten(start_dim=1).pow(2)).sum()

        return kl_div

    @torch.no_grad()
    def generate_sample(self, num_samples: int, device=torch.device('cuda'), labels: torch.Tensor=None):
        if labels is not None:
            assert len(labels) == num_samples, 'Error: number of labels should be the same as number of samples!'
            labels = labels.to(device)
        else:

            labels = torch.randint(0, self.num_classes, [num_samples,], device=device)

        # TODO: sample from standard Normal distrubution
        noise = torch.randn(num_samples, self.latent_dim, 1, 1, device=device)
        label_emb = self.class_emb(labels).view(num_samples, self.mid_size[0], self.mid_size[1], self.mid_size[2])     
        noise = noise.expand(-1, -1, self.mid_size[1], self.mid_size[2])
        noise = torch.cat([noise, label_emb], dim=1) 

        out = self.decode(noise, labels)

        return out
    
    def decode(self, sample: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        out = self.decoder(sample)

        return out




class LDDPM(nn.Module):
    def __init__(self, network: nn.Module, vae: VAE, var_scheduler: VarianceScheduler) -> None:
        super().__init__()

        self.var_scheduler = var_scheduler
        self.vae = vae
        self.network = network
        self.vae.requires_grad_(False)

    def forward(self, x: torch.Tensor, label: torch.Tensor) -> torch.Tensor:
        t = torch.randint(0, self.var_scheduler.num_steps, (x.size(0),), device=x.device)
        latent_repr = self.vae.encoder(x).view(x.size(0), -1)  
        noisy_input, noise = self.var_scheduler.add_noise(latent_repr, t)
        estimated_noise = self.network(noisy_input, t, label)
        loss = F.mse_loss(estimated_noise, noise)

        return loss

    @torch.no_grad()
    def recover_sample(self, noisy_sample: torch.Tensor, estimated_noise: torch.Tensor, timestep: torch.Tensor) -> torch.Tensor:
        timestep = timestep.to(self.var_scheduler.betas.device)

        beta_t = self.var_scheduler.betas[timestep].view(-1, 1).to(noisy_sample.device)
        alpha_t = self.var_scheduler.alphas[timestep].view(-1, 1).to(noisy_sample.device)
        alpha_bar_t = self.var_scheduler.alpha_bars[timestep].view(-1, 1).to(noisy_sample.device)
        mu_t = (1 / torch.sqrt(alpha_t)) * (
            noisy_sample - (beta_t / torch.sqrt(1 - alpha_bar_t)) * estimated_noise
        )
        if timestep.min() > 0:
            sigma_t = torch.sqrt(beta_t)
            sample = mu_t + sigma_t * torch.randn_like(noisy_sample)
        else:
            sample = mu_t

        return sample

    @torch.no_grad()
    def generate_sample(self, num_samples: int, device: torch.device = torch.device('cuda'), labels: torch.Tensor = None):
        if labels is not None:
            assert len(labels) == num_samples, 'Error: number of labels should be the same as number of samples!'
            labels = labels.to(device)
        else:
            labels = torch.randint(0, self.vae.num_classes, (num_samples,), device=device)

        latent_shape = self.vae.mid_size  
        current_sample = torch.randn((num_samples, latent_shape[0], latent_shape[1], latent_shape[2]), device=device)

        for t in reversed(range(self.var_scheduler.num_steps)):
            t_tensor = torch.full((num_samples,), t, device=device, dtype=torch.long)
            estimated_noise = self.network(current_sample, t_tensor, labels)
            current_sample = self.recover_sample(current_sample, estimated_noise, t_tensor)

        final_samples = self.vae.decode(current_sample, labels)

        return final_samples



class DDPM(nn.Module):
    def __init__(self, network: nn.Module, var_scheduler: VarianceScheduler) -> None:
        super().__init__()

        self.var_scheduler = var_scheduler
        self.network = network

    def forward(self, x: torch.Tensor, label: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        batch_size = x.size(0)
        t = torch.randint(0, self.var_scheduler.num_steps, (batch_size,), device=x.device)

        noisy_input, noise = self.var_scheduler.add_noise(x, t)

        estimated_noise = self.network(noisy_input, t, label)

        loss = F.mse_loss(estimated_noise, noise) 
        return loss


    @torch.no_grad()
    def recover_sample(self, noisy_sample: torch.Tensor, estimated_noise: torch.Tensor, timestep: torch.Tensor) -> torch.Tensor:

        timestep = timestep.to(self.var_scheduler.betas.device)

        beta_t = self.var_scheduler.betas[timestep].view(-1, 1, 1, 1).to(noisy_sample.device)
        alpha_t = self.var_scheduler.alphas[timestep].view(-1, 1, 1, 1).to(noisy_sample.device)
        alpha_bar_t = self.var_scheduler.alpha_bars[timestep].view(-1, 1, 1, 1).to(noisy_sample.device)


        mu_t = (1 / torch.sqrt(alpha_t)) * (
            noisy_sample - (beta_t / torch.sqrt(1 - alpha_bar_t)) * estimated_noise)


        if timestep.min() > 0:
            sigma_t = torch.sqrt(beta_t)
            sample = mu_t + sigma_t * torch.randn_like(noisy_sample)
        else:
            sample = mu_t
        return sample

    @torch.no_grad()
    def generate_sample(self, num_samples: int, device: torch.device=torch.device('cuda'), labels: torch.Tensor=None):
        x = torch.randn((num_samples, 1, 32, 32), device=device)

        if labels is None:
            labels = torch.randint(0, self.network.num_classes, (num_samples,), device=device)

        for t in reversed(range(self.var_scheduler.num_steps)):
            t_tensor = torch.full((num_samples,), t, device=device, dtype=torch.long)
            estimated_noise = self.network(x, t_tensor, labels)
            x = self.recover_sample(x, estimated_noise, t_tensor)

        return x



class DDIM(nn.Module):
    def __init__(self, network: nn.Module, var_scheduler: VarianceScheduler) -> None:
        super().__init__()
        self.var_scheduler = var_scheduler
        self.network = network

    def forward(self, x: torch.Tensor, label: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        ten = torch.randint(low=0, high=self.var_scheduler.num_steps, size=(x.size(0),), device=x.device)
        noisy_input, noise_target = self.var_scheduler.add_noise(x, ten)
        predicted_noise = self.network(noisy_input, ten, label)
        loss = F.l1_loss(predicted_noise, noise_target)
        return loss

    @torch.no_grad()
    def recover_sample(self, noisy_sample: torch.Tensor, estimated_noise: torch.Tensor, timestep: torch.Tensor) -> torch.Tensor:
        timestep = timestep.to(self.var_scheduler.betas.device)
        alpha_bt = self.var_scheduler.alpha_bars[timestep].view(-1, 1, 1, 1).to(noisy_sample.device)
        alpha_bt_prev = self.var_scheduler.alpha_bars[torch.clamp(timestep - 1, min=0)].view(-1, 1, 1, 1).to(noisy_sample.device)

        predicted_sample = (noisy_sample - torch.sqrt(1 - alpha_bt) * estimated_noise) / torch.sqrt(alpha_bt)


        next_sample = torch.sqrt(alpha_bt_prev) * predicted_sample + torch.sqrt(1 - alpha_bt_prev) * estimated_noise
        return next_sample

    @torch.no_grad()
    def generate_sample(self, num_samples: int, device: torch.device = torch.device('cuda'), labels: torch.Tensor = None):
        current_sample = torch.randn((num_samples, 1, 32, 32), device=device)


        if labels is not None and self.network.num_classes is not None:
            assert len(labels) == num_samples, "Number of labels must match the number of samples!"
            labels = labels.to(device)
        elif labels is None and self.network.num_classes is not None:
            labels = torch.randint(0, self.network.num_classes, (num_samples,), device=device)
        else:
            labels = None

        for t in reversed(range(self.var_scheduler.num_steps)):
            t_ten = torch.full((num_samples,), t, device=device, dtype=torch.long)
            predicted_noise = self.network(current_sample, t_ten, labels)
            current_sample = self.recover_sample(current_sample, predicted_noise, t_ten)

        return current_sample

