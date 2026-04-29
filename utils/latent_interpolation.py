import os
import numpy as np
import sys
import torch
import matplotlib.pyplot as plt
import re
from pathlib import Path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from AtrialMorphNet.model.AtrialMorphNet import *

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Usando dispositivo: {device}")

latent_dim = 256
num_points = 2048

results_path = "" # Path to saved test results
test_output_path = os.path.join(results_path, "test_best")
latent_dir = os.path.join(test_output_path, "latent_vectors")

best_model_path = os.path.join(results_path, "best_model.pth") #revisar en train se guarde así

model = AtrialMorphNet(latent_dim=latent_dim, num_points=num_points).to(device)

state_dict = torch.load(best_model_path, map_location=device)
model.load_state_dict(state_dict)
model.eval()
print(f"Mejor modelo cargado desde: {best_model_path}")

patient_a = ""  #Name patient a in .ny (only the name of the document!!)
patient_b = ""  #Name patient b in .ny (only the name of the document!!)

date_a = re.search(r'(\d{4}_\d{2}_\d{2})', patient_a).group(1)
date_b = re.search(r'(\d{4}_\d{2}_\d{2})', patient_b).group(1)

interp_folder_name = f"{date_a}_to_{date_b}"
interp_dir = os.path.join(test_output_path, "latent_interpolations", interp_folder_name)
Path(interp_dir).mkdir(parents=True, exist_ok=True)

def plot_pointcloud(pc, ax, title="", color="blue"):
   
    ax.scatter(pc[:, 0], pc[:, 1], pc[:, 2], s=1, c=color)
    ax.set_title(title)
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")


def plot_interpolation_sequence(pointclouds, titles, save_path):

    n = len(pointclouds)
    fig = plt.figure(figsize=(3 * n, 3))

    for i, (pc, ttl) in enumerate(zip(pointclouds, titles)):
        ax = fig.add_subplot(1, n, i + 1, projection="3d")
        plot_pointcloud(pc, ax, ttl, color="blue")

    plt.tight_layout()
    plt.savefig(save_path, dpi=200)
    plt.close()
    print(f"Figura de interpolación guardada en: {save_path}")


mu_a = np.load(os.path.join(latent_dir, f"{name_a}.npy"))
mu_b = np.load(os.path.join(latent_dir, f"{name_b}.npy"))  

mu_a = torch.from_numpy(mu_a).float().to(device).unsqueeze(0)  
mu_b = torch.from_numpy(mu_b).float().to(device).unsqueeze(0)  

n_steps = 7
ts = np.linspace(0.0, 1.0, n_steps)

all_recons = []
all_titles = []

with torch.no_grad():
    for idx, t in enumerate(ts):
      
        mu_t = (1.0 - t) * mu_a + t * mu_b 
        recon_t = model.decoder(mu_t)     
        recon_np = recon_t.squeeze(0).cpu().numpy()

        all_recons.append(recon_np)
        all_titles.append(f"t={t:.2f}")

        np.save(os.path.join(interp_dir, f"interp_{date_a}_to_{date_b}_t{idx:02d}.npy",),recon_np,)


png_path = os.path.join(interp_dir,f"interp_{date_a}_to_{date_b}.png",)
plot_interpolation_sequence(all_recons, all_titles, png_path)

