import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
     
class FoldingDecoderCoarseToDense(nn.Module):
    def __init__(self, latent_dim=256):
        super().__init__()
        self.coarse_points = 128     
        self.patch_size = 4           
        self.total_points = 2048     
    
        self.coarse_mlp = nn.Sequential(
            nn.Linear(latent_dim, 512),
            nn.ReLU(),
            nn.Linear(512, 1024),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(1024, self.coarse_points * 3) 
        )
        
        s = self.patch_size
        grid = torch.stack(torch.meshgrid(
            torch.linspace(-0.12, 0.12, s),  
            torch.linspace(-0.12, 0.12, s),
            indexing='ij'
        ), dim=-1).reshape(-1, 2) 
        self.register_buffer('grid', grid)
        
        self.fold_mlp = nn.Sequential(
            nn.Linear(latent_dim + 3 + 2, 512),
            nn.LeakyReLU(0.2),
            nn.Linear(512, 512),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.1),
            nn.Linear(512, 256),
            nn.LeakyReLU(0.2),
            nn.Linear(256, 3)
        )
    
    def forward(self, z):
        B = z.shape[0]
        
        coarse_flat = self.coarse_mlp(z)           # (B, 384)
        coarse = coarse_flat.view(B, 128, 3)       # (B, 128, 3)
        
      
        num_patch_pts = self.patch_size ** 2  # 16
        
        grid_rep = self.grid.unsqueeze(0).unsqueeze(0).repeat(B, 128, 1, 1)      # (B, 128, 16, 2)
        coarse_exp = coarse.unsqueeze(2).repeat(1, 1, num_patch_pts, 1)          # (B, 128, 16, 3)
        z_exp = z.unsqueeze(1).unsqueeze(2).repeat(1, 128, num_patch_pts, 1)     # (B, 128, 16, latent)
        
      
        fold_input = torch.cat([z_exp, coarse_exp, grid_rep], dim=-1)
        fold_input = fold_input.view(B, -1, fold_input.shape[-1])  # (B, 2048, 261)
    
        output = self.fold_mlp(fold_input)  # (B, 2048, 3)
        
        return output 
