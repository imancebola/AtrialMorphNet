import torch
import torch.nn as nn
from torch_geometric.nn import EdgeConv, knn_graph, global_mean_pool, global_max_pool
import torch.nn.functional as F

## GRAPH ENCODER ####
class EdgeBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.edge_conv = EdgeConv(
            nn.Sequential(
                nn.Linear(2 * in_channels, out_channels),
                nn.ReLU(),
                nn.Linear(out_channels, out_channels),
                nn.ReLU()
            ),
            aggr='max'
        )

    def forward(self, x, edge_index):
        return self.edge_conv(x, edge_index)
    
class GraphPointNetVAEEncoder_max_mean_pooling(nn.Module):
    def __init__(self, latent_dim=256, input_channels=10, k=16):
        super().__init__()

        self.k = k
        # ---------- POINTNET ----------
        self.conv1 = nn.Sequential(
            nn.Conv1d(input_channels, 64, 1),
            nn.BatchNorm1d(64),
            nn.ReLU()
        )
        self.conv2 = nn.Sequential(
            nn.Conv1d(64, 128, 1),
            nn.BatchNorm1d(128),
            nn.ReLU()
        )

        # ---------- GNN ----------
        self.edge1 = EdgeBlock(128, 128)
        self.edge2 = EdgeBlock(128, 256)

        self.fc = nn.Sequential(
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, latent_dim * 2)
        )

    def forward(self, x):
        """
        x: (B, N, 10)
        Returns: mu, logvar, trans_feat
        """
        B, N, F = x.shape
    
        # ---------- POINTNET ----------
        x = x.transpose(1, 2)  # (B, F, N)
        x = self.conv1(x)      # (B, 64, N)
        x = self.conv2(x)      # (B, 128, N)
        
        x = x.transpose(1, 2)  # (B, N, 128)

        x_flat = x.reshape(B * N, -1)
        pos_flat = x[:, :, :3].reshape(B * N, 3)
        batch = torch.arange(B, device=x.device).repeat_interleave(N)

        edge_index = knn_graph(
            pos_flat,
            k=self.k,
            batch=batch,
            loop=False
        )

        # ---------- GNN ----------
        x_flat = self.edge1(x_flat, edge_index)  
        x_flat = self.edge2(x_flat, edge_index)  

        # ---------- POOLING GLOBAL (MEAN + MAX) ----------
        x_mean = global_mean_pool(x_flat, batch) 
        x_max = global_max_pool(x_flat, batch)   
        x_global = torch.cat([x_mean, x_max], dim=1) 

        # ---------- VAE ----------
        x_latent = self.fc(x_global)  
        mu, logvar = x_latent.chunk(2, dim=1)  

        return mu, logvar 



