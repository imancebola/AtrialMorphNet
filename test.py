import torch
import numpy as np
import os
import matplotlib.pyplot as plt
import warnings
from torch.utils.data import DataLoader
from AtrialMorphNet.model.AtrialMorphNet import *
from data.dataset import *
from open3d.t.geometry import PointCloud, Metric, MetricParameters
import open3d as o3d
from tqdm import tqdm

warnings.filterwarnings("ignore")

print(f"CUDA available: {torch.cuda.is_available()}")

##### RUN TEST #####
latent_dim = 256
num_points = 2048

output_path = ''  # Path to save the test results
DATA_PATH = ''    # Path to the folder where train/val/test splits are saved for training the model

os.makedirs(output_path, exist_ok=True)
os.makedirs(os.path.join(output_path, 'original_point_cloud'), exist_ok=True)
os.makedirs(os.path.join(output_path, 'reconstructed_point_cloud'), exist_ok=True)
os.makedirs(os.path.join(output_path, 'latent_vectors'), exist_ok=True)
os.makedirs(os.path.join(output_path, 'point_visualization'), exist_ok=True)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# LOAD TEST DATA
test_set = PointCloudDatasetOBJ(os.path.join(DATA_PATH, "test"))
test_loader = DataLoader(test_set, batch_size=1, shuffle=False, num_workers=4, pin_memory=True)

# Load best model for testing
best_model_path = ''  # Path to the best model

model = AtrialMorphNet(latent_dim=latent_dim, num_points=num_points).to(device)
model.load_state_dict(torch.load(best_model_path, map_location=device))
model.eval()

print(f"Best model loaded from: {best_model_path}")

# ==================== VISUALIZATION FUNCTION ====================
def plot_mesh_pair(original, recon, cd_value, filename):
    fig = plt.figure(figsize=(10, 5))
    ax1 = fig.add_subplot(1, 2, 1, projection='3d')
    ax1.scatter(original[:, 0], original[:, 1], original[:, 2], c='blue', s=1)
    ax1.set_title(f'Original - {filename}\nCD: {cd_value:.6f}')

    ax2 = fig.add_subplot(1, 2, 2, projection='3d')
    ax2.scatter(recon[:, 0], recon[:, 1], recon[:, 2], c='red', s=1)
    ax2.set_title(f'Reconstructed - {filename}\nCD: {cd_value:.6f}')

    plt.tight_layout()

    safe_name = os.path.splitext(os.path.basename(filename))[0]
    plt.savefig(os.path.join(output_path, 'point_visualization', f'point_pair_{safe_name}.png'), dpi=200)
    plt.close()

# ==================== TESTING LOOP ====================
chamfer_distances = []
hausdorff_distances = []

with torch.no_grad():
    for i, x in enumerate(tqdm(test_loader, desc='Testing')):
        x = x.to(device)

        if hasattr(test_set, 'files'):
            orig_path = test_set.files[i]
        elif hasattr(test_set, 'file_list'):
            orig_path = test_set.file_list[i]
        elif hasattr(test_set, 'data_list'):
            orig_path = test_set.data_list[i]
        else:
            raise AttributeError("No file list found in test_set (expected 'files', 'file_list', or 'data_list')" )

        filename = os.path.basename(orig_path)
        name_no_ext = os.path.splitext(filename)[0]

        mu, logvar, trans_feat = model.encoder(x)
        recon = model.decoder(mu)

        x_np = x.squeeze(0).cpu().numpy().astype(np.float32)
        recon_np = recon.squeeze(0).cpu().numpy().astype(np.float32)

        # Create Open3D point clouds
        pcd1 = PointCloud(o3d.core.Tensor(x_np[:, :3], dtype=o3d.core.Dtype.Float32))
        pcd2 = PointCloud(o3d.core.Tensor(recon_np[:, :3], dtype=o3d.core.Dtype.Float32))

        metrics = pcd1.compute_metrics(pcd2, (Metric.ChamferDistance, Metric.HausdorffDistance), MetricParameters() )

        cd_value = metrics[0].item()
        hd_value = metrics[1].item()

        chamfer_distances.append(cd_value)
        hausdorff_distances.append(hd_value)

        np.save(os.path.join(output_path, 'original_point_cloud', f'{name_no_ext}.npy'), x_np)
        np.save(os.path.join(output_path, 'reconstructed_point_cloud', f'{name_no_ext}.npy'), recon_np)
        np.save(os.path.join(output_path, 'latent_vectors', f'{name_no_ext}.npy'), mu.squeeze().cpu().numpy())

        plot_mesh_pair(x_np, recon_np, cd_value, filename)

# ==================== SAVE STATISTICS ====================

# Chamfer Distance
chamfer_distances_np = np.array(chamfer_distances)
np.save(os.path.join(output_path, 'chamfer_distances.npy'), chamfer_distances_np)

with open(os.path.join(output_path, 'chamfer_statistics.txt'), 'w') as f:
    f.write(f"Mean: {chamfer_distances_np.mean():.6f}\n")
    f.write(f"Min: {chamfer_distances_np.min():.6f}\n")
    f.write(f"Max: {chamfer_distances_np.max():.6f}\n")
    f.write(f"Standard deviation: {chamfer_distances_np.std():.6f}\n")
    f.write(f"Total samples: {len(chamfer_distances_np)}\n")

# Hausdorff Distance
hausdorff_distances_np = np.array(hausdorff_distances)
np.save(os.path.join(output_path, 'hausdorff_distance.npy'), hausdorff_distances_np)

with open(os.path.join(output_path, 'hausdorff_distance.txt'), 'w') as f:
    f.write(f"Mean: {hausdorff_distances_np.mean():.6f}\n")
    f.write(f"Min: {hausdorff_distances_np.min():.6f}\n")
    f.write(f"Max: {hausdorff_distances_np.max():.6f}\n")
    f.write(f"Standard deviation: {hausdorff_distances_np.std():.6f}\n")
    f.write(f"Total samples: {len(hausdorff_distances_np)}\n")

print(f"\nMetrics and statistics saved in {output_path}")