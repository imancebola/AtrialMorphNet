import torch
import torch.nn as nn
import plotly.graph_objects as go

from model.encoders import *
from model.decoders import *
from geomloss import SamplesLoss

###### LOSSES########

def chamfer_loss_l2_sqr(recon, x):
    
    x_coords = x[..., :3]  # pointS (X,Y,Z) + FEATURES
    recon_coords = recon[..., :3]

    dist_reconx = torch.cdist(recon_coords, x_coords, p=2.0)**2 #L2 cuadrado
    min_dist_recon = dist_reconx.min(dim=-1)[0]

    dist_xrecon = torch.cdist(x_coords, recon_coords, p=2.0)**2
    min_dist_x = dist_xrecon.min(dim=-1)[0]
    loss_cd = min_dist_recon.mean() + min_dist_x.mean()
    return loss_cd


def vae_loss(recon, x, mu, logvar, beta=0.001, recon_weight = 1, device = None): 
    
    recon_loss = chamfer_loss_l2_sqr(recon,x) * recon_weight     
  
    kl = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), dim=1).mean()
    
    total_loss = recon_loss + beta * kl 
    
    return total_loss, recon_loss.item(), kl.item(), reg.item() , repul_loss.item()

###### AtrialMorphNet  ###
class AtrialMorphNet(nn.Module):
    def __init__(self, latent_dim=128, num_points=2048):
        super().__init__()
    
        self.encoder = GraphPointNetVAEEncoder_max_mean_pooling(latent_dim)
        self.decoder = FoldingDecoderCoarseToDense(latent_dim)
        self.latent_dim = latent_dim
    
    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std
    
    def forward(self, x):
        mu, logvar, trans_feat = self.encoder(x)
        z = self.reparameterize(mu, logvar)
        x_recon = self.decoder(z)
        return x_recon, mu, logvar, trans_feat
    
    def sample(self, n=8, device='cuda'):
        z = torch.randn(n, self.latent_dim).to(device)
        return self.decoder(z)
    
