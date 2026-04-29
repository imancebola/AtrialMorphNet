#MODELO CON TNET
import torch
import torch.optim as optim
import numpy as np
import os
import matplotlib.pyplot as plt
import plotly.graph_objects as go
from torch.utils.data import DataLoader
import matplotlib.pyplot as pl
import imageio
import warnings
import wandb
import torch.optim as optim
from open3d.t.geometry import PointCloud, Metric, MetricParameters
import open3d as o3d
warnings.filterwarnings("ignore")
from tqdm import tqdm   
from torch_geometric.nn import fps

from data.dataset import *
from AtrialMorphNet.model.AtrialMorphNet import *
from model.utils import *

print(f"CUDA disponible: {torch.cuda.is_available()}")

wandb.login()

DATA_PATH = "" # Path to the folder where train/val/test splits are saved for training the model

batch_size = 16
latent_dim = 256
lr = 0.00001
recon_weight = 500  # for reco loss weight
patience=60
epochs = 1000
num_points = 10000

ANNEAL_EPOCHS = 1800
START_ANNEAL = 0
BETA_MAX = 0
encoder_name = 'graph_encoder'
decoder_name = 'folding'

USE_TNET=False
beta_ann = 'scalar' # scalar, lineal

print(f"Encoder  → {encoder_name}")
print(f"Decoder → {decoder_name}")


ROOT_RESULTS_DIR = os.path.join(os.getcwd(), "results_PointVAE")

if beta_ann == 'lineal':
    EXPERIMENT_NAME = (
        f"train_vae_{batch_size}"
        f"_{encoder_name}"
        f"_features_7" # añadir que dataset
        f"_{recon_loss_name}"
        f"_{loss_norm}"
        f"_{recon_weight}_newdim"
        f"scheduler_{patience}_"
        f"z{latent_dim}_"
        f"start_ann_{START_ANNEAL}_"
        f"beta_ann_{ANNEAL_EPOCHS}_"
        f"beta_max_{BETA_MAX}_"
        f"TNET_{USE_TNET}_"
        f"lr{lr}_"
        f"ep{epochs}_"
        f"n{num_points}_"
        f"{decoder_name}_"
        f"{beta_ann}"
        
    )
    
elif beta_ann == 'scalar':
    EXPERIMENT_NAME = (
        f"train_vae_{batch_size}"
        f"_{encoder_name}"
        f"_features_10"
        f"_{recon_loss_name}"
        f"_{loss_norm}"
        f"_{recon_weight}_newdim"
        f"scheduler_{patience}_"
        f"z{latent_dim}_"
        f"beta_value_{BETA_MAX}_"
        f"TNET_{USE_TNET}_"
        f"lr{lr}_"
        f"ep{epochs}_"
        f"n{num_points}_"
        f"{decoder_name}"
        f"{beta_ann}"
        
    )
OUTPUT_DIR = os.path.join(ROOT_RESULTS_DIR, EXPERIMENT_NAME)
RECON_DIR = os.path.join(OUTPUT_DIR, "reconstructions")

os.makedirs(RECON_DIR, exist_ok=True)
print(f"Todos los archivos de salida se guardarán en: {OUTPUT_DIR}")


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)


###### POINTCLOUDS FROM MESHES ###
train_set = PointCloudDatasetOBJ(os.path.join(DATA_PATH, "train"))
val_set   = PointCloudDatasetOBJ(os.path.join(DATA_PATH, "val"))

train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=1, pin_memory=True) # cambiar por num_workers=0 o cambiar dataloaders
val_loader   = DataLoader(val_set, batch_size=batch_size, shuffle=False, num_workers=1, pin_memory=True)

####### TRAIN/VAL LOOP #####

model = AtrialMorphNet(latent_dim=latent_dim, num_points=num_points).to(device)
optimizer = optim.Adam(model.parameters(), lr=lr)
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.2, patience = patience,  threshold=1e-4, min_lr=1e-6, verbose=True)


train_losses, val_losses = [], []
train_recon_losses, train_kl_losses, train_reg_losses = [], [], []
val_recon_losses, val_kl_losses, val_reg_losses = [], [], []

train_losses_cd, train_losses_emd, val_losses_cd, val_losses_emd = [], [], [], []

best_val_cd = float('inf') 
best_model_path = os.path.join(OUTPUT_DIR, f"best_model.pth")

gif_images = []

# ====================== WEIGHTS & BIASES ======================
config = {
    "epochs": epochs,
    "learning_rate": lr,
    "latent_dim": latent_dim,
    "num_points": num_points,
    "beta_max": BETA_MAX ,
    "KL annealing": beta_ann,
    "encoder_name": encoder_name,
    "decoder_type": decoder_name,
    "batch_size": train_loader.batch_size,
    "experiment_name": EXPERIMENT_NAME,
}

wandb.init(project="AltrialMorphNet", name=EXPERIMENT_NAME, config=config )
wandb.watch(model, log="all", log_freq=100)

print("Weights & Biases started → open this link →", wandb.run.url)

print("--TRAINING---")
for epoch in tqdm(range(1, epochs + 1), desc="Entrenando", leave=True):

    
    # ==================== TRAIN ====================
    model.train()
    train_loss = 0
    train_recon = 0
    train_kl = 0
    train_reg = 0
    train_repul = 0 
    train_normals = 0
  
    #BETA VALUE
    if beta_ann == 'lineal':
        if epoch < START_ANNEAL:
            current_beta = 0.0
        else:
            current_beta = min(BETA_MAX, ((epoch - START_ANNEAL) / (ANNEAL_EPOCHS - START_ANNEAL)) * BETA_MAX)
    elif beta_ann == 'scalar':
        current_beta = BETA_MAX

    for x in train_loader:
        x = x.to(device)
        optimizer.zero_grad()
        
        recon, mu, logvar = model(x)
        loss, recon_l, kl_l, reg_loss, repul_loss = vae_loss(recon, x, mu, logvar, current_beta, recon_weight, device = device)
    
        loss.backward()
        optimizer.step()
        
        train_loss += loss.item()
        train_recon += recon_l
        train_kl += kl_l
        train_reg += reg_loss
        train_repul += repul_loss
    
    train_loss /= len(train_loader)
    train_recon /= len(train_loader)
    train_kl /= len(train_loader)
    train_reg /= len(train_loader)
    train_repul /= len(train_loader)
    train_losses.append(train_loss)
    train_recon_losses.append(train_recon)
    train_kl_losses.append(train_kl)
    train_reg_losses.append(train_reg)
    
    # ==================== VALIDATION ====================
    model.eval()
    val_loss = 0
    val_recon = 0
    val_kl = 0
    val_repul = 0
    val_normals = 0
    val_chamfer_metrics = []
    val_hausdorff_metrics = []
    tnet_ortho_errors = []
    tnet_identity_dists = []

    with torch.no_grad():
        for x in val_loader:
            x = x.to(device)
          
            recon, mu, logvar = model(x)
            loss, recon_l, kl_l, reg_loss, repul_loss = vae_loss(recon, x, mu, logvar, current_beta, recon_weight, device = device)
           
            val_loss += loss.item()
            val_recon += recon_l
            val_kl += kl_l
            val_reg += reg_loss
            val_repul += repul_loss
      
            batch_chamfers = []
            batch_hausdorffs = []

            for i in range(x.shape[0]): 

                x_np = x[i, :, :3].cpu().numpy() 
                recon_np = recon[i].cpu().numpy()
                
                pcd1 = PointCloud(o3d.core.Tensor(x_np))
                pcd2 = PointCloud(o3d.core.Tensor(recon_np))

                metrics = pcd1.compute_metrics(pcd2,[Metric.ChamferDistance, Metric.HausdorffDistance],MetricParameters() )
                
                batch_chamfers.append(metrics[0].item())
                batch_hausdorffs.append(metrics[1].item())

        
            val_chamfer_metrics.append(np.mean(batch_chamfers))
            val_hausdorff_metrics.append(np.mean(batch_hausdorffs))
    

    val_loss /= len(val_loader)
    val_recon /= len(val_loader)
    val_kl /= len(val_loader)
    val_reg /= len(val_loader)
    val_repul /= len(val_loader)
    val_losses.append(val_loss)
    val_recon_losses.append(val_recon)
    val_kl_losses.append(val_kl)
    val_reg_losses.append(val_reg)
   
    scheduler.step(val_loss)
    current_lr = optimizer.param_groups[0]['lr']

    val_chamfer_avg = np.mean(val_chamfer_metrics)
    val_hausdorff_avg = np.mean(val_hausdorff_metrics)


    #T-net error
    avg_ortho = np.mean(tnet_ortho_errors)
    avg_identity = np.mean(tnet_identity_dists)

    print(f"Epoch {epoch} | LR: {current_lr:.6f} | Beta: {current_beta:.6f} |Train Loss={train_loss:.5f}, Val Loss={val_loss:.5f}| Train Recon: {train_recon:.5f}, Val Recon:{val_recon:.5f} | CD={val_chamfer_avg:.5f}, HD={val_hausdorff_avg:.5f}")

    wandb.log({
        "train/total_loss": train_loss,
        "train/recon_loss": train_recon,
        "train/kl_loss": train_kl,
        "train/reg_loss": train_reg, 
        "train/repulsion_loss": train_repul,
        "train/learning_rate": current_lr,
        "train/current_beta": current_beta,
        "val/total_loss": val_loss,
        "val/recon_loss": val_recon,
        "val/kl_loss": val_kl,
        "val/reg_loss": val_reg,
        "val/repulsion_loss": val_repul,
        "val/chamfer_distance": val_chamfer_avg,
        "val/hausdorff_distance": val_hausdorff_avg,
        "tnet/orthogonality_error": avg_ortho,
        "tnet/identity_distance": avg_identity,
        "epoch": epoch,
        "best_val_cd": best_val_cd,
    })
    
    # ==================== SAFE BEST MODEL ====================
    if val_chamfer_avg < best_val_cd :
        best_val_cd = val_chamfer_avg
        best_epoch_saved = epoch 
        epochs_without_improvement = 0

        torch.save(model.state_dict(), best_model_path)
        print(f"→ Best model safed during epoch {epoch} with CD={best_val_cd:.5f}")
        wandb.save(best_model_path)

        hp_txt_path = best_model_path.replace(".pth", "_hyperparams.txt") 
        
        with open(hp_txt_path, "w") as f:
            f.write("=== HYPERPARAMETERS OF THE BEST MODEL ===\n")
            f.write(f"Epoch: {epoch}\n")
            f.write(f"Best Validation Chamfer Distance: {best_val_cd:.6f}\n")
            f.write(f"Validation Hausdorff Distance: {val_hausdorff_avg:.6f}\n")
            f.write("\n--- Training configuration ---\n")
            f.write(f"Learning Rate (in this epoch): {current_lr:.8f}\n")
            f.write(f"Current Beta (KL weight): {current_beta}\n")
            f.write(f"Regularization weight (T-Net): 0.001\n")  # if fixed
            f.write("\n--- Model hyperparameters (examples) ---\n")
            f.write(f"Latent dimension: {latent_dim}\n")
            f.write(f"Number of points: {num_points}\n")

        print(f"   Hyperparameters saved in: {hp_txt_path}")
        wandb.save(hp_txt_path)

    # ==================== SAVE RECONSTRUCTIONS EVERY 10 EPOCHS ====================
    if epoch % 10 == 0 or epoch == epochs:
        print(f"\nSaving reconstructions - Epoch {epoch}...")
        with torch.no_grad():
            val_samples = []
            for batch in val_loader:
                val_samples.append(batch.to(device))
                if sum(b.shape[0] for b in val_samples) >= 8:
                    break
            val_samples = torch.cat(val_samples, dim=0)[:8]
            recon_batch, _, _, _ = model(val_samples)
            
            val_samples = val_samples.cpu()
            recon_batch = recon_batch.cpu()

            fig, axes = plt.subplots(4, 4, figsize=(16, 16), subplot_kw={'projection': '3d'})
            fig.suptitle(f"Época {epoch} - Val Loss: {val_loss:.5f}; CD={val_chamfer_avg:.5f}; HD={val_hausdorff_avg:.5f}", fontsize=20, y=0.95)

            for i in range(8):
                
                ax = axes[i//2, (i % 2) * 2]
                
                ax.scatter(val_samples[i,:,0], val_samples[i,:,1], val_samples[i,:,2], c='lightblue', s=1.5, alpha=0.9)  
                ax.set_title(f"Orig {i+1}", fontsize=12)
                ax.axis('off')

                ax = axes[i//2, (i % 2) * 2 + 1]
                ax.scatter(recon_batch[i,:,0], recon_batch[i,:,1], recon_batch[i,:,2], c='coral', s=1.5, alpha=0.9) 
                ax.set_title(f"Recon {i+1}", fontsize=12)
                ax.axis('off')

            plt.tight_layout()
            path = os.path.join(RECON_DIR, f"epoch_{epoch:04d}.png")
            plt.savefig(path, dpi=150, bbox_inches='tight', facecolor='white')
            plt.close()
            wandb.log({f"recon/epoch_{epoch:04d}": wandb.Image(path, caption=f"CD {val_chamfer_avg:.5f}")})
            gif_images.append(imageio.imread(path)) 
            print(f"Guardada: {path}")

   
# ==================== SAFE MODEL AND GIF ====================
print("\n" + "="*60)
print("GENERATING FINAL OUTPUTS")
print("="*60)

final_epoch = epoch
model_filename = f"vae_final_epoch_{decoder_name}.pth"
model_save_path = os.path.join(OUTPUT_DIR, model_filename)
torch.save(model.state_dict(), model_save_path)
wandb.save(model_save_path)
print(f"✓ Final model saved: {model_save_path}")

# Save training summary
summary_path = os.path.join(OUTPUT_DIR, "training_summary.txt")
with open(summary_path, "w") as f:
    f.write("=== TRAINING SUMMARY ===\n")
    f.write(f"Completed epochs: {final_epoch}/{epochs}\n")
    f.write(f"\nBest model saved at epoch: {best_epoch_saved}\n")
    f.write(f"\nBest Chamfer Distance: {best_val_cd:.6f}\n")
    f.write(f"Final Train Loss: {train_loss:.6f}\n")
    f.write(f"Final Val Loss: {val_loss:.6f}\n")
    f.write(f"Final Learning Rate: {current_lr:.8f}\n")
    f.write(f"Final Beta: {current_beta}\n")

print(f"✓ Summary saved: {summary_path}")
wandb.save(summary_path)

if gif_images:
    gif_save_path = os.path.join(OUTPUT_DIR, "vae_training_evolution.gif")
    imageio.mimsave(gif_save_path, gif_images, fps=6)
    wandb.log({"training_gif": wandb.Video(gif_save_path, fps=6, format="gif")})
    print(f"GIF saved at: {gif_save_path}")
else:
    print("No images were generated for the GIF.")

print(f"Final model saved: {model_save_path}")

# --- Reconstruction Loss Plot ---
fig_recon = plt.figure(figsize=(12, 5))
plt.plot(train_recon_losses, label="Train Reconstruction Loss", color='blue', lw=1.5)
plt.plot(val_recon_losses, label="Validation Reconstruction Loss", color='orange', lw=1.5)
plt.yscale('log')
plt.xlabel("Epoch")
plt.ylabel("Loss")
title = f"Training Evolution - Reconstruction Loss (Stopped at epoch {final_epoch})"
plt.title(title)
plt.legend()
plt.grid(True, alpha=0.3)

recon_plot_path = os.path.join(OUTPUT_DIR, "reconstruction_loss.png")
plt.savefig(recon_plot_path, dpi=300, bbox_inches='tight')

wandb.log({
    "plots/reconstruction_loss": wandb.Image(recon_plot_path),
    "charts/interactive_recon": plt.gcf()
})
plt.close()

# --- Total Loss Plot ---
fig_total = plt.figure(figsize=(12, 5))
plt.plot(train_losses, label="Train Total Loss", color='green', lw=1.5)
plt.plot(val_losses, label="Validation Total Loss", color='red', lw=1.5)
plt.yscale('log')
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("Training Evolution (Total Loss)")
plt.legend()
plt.grid(True, alpha=0.3)

if best_epoch_saved > 0 and best_epoch_saved <= len(val_losses):
    plt.axvline(
        x=best_epoch_saved - 1,
        color='green',
        linestyle='--',
        alpha=0.7,
        label=f'Best model (epoch {best_epoch_saved}, CD={best_val_cd:.5f})'
    )

total_plot_path = os.path.join(OUTPUT_DIR, "total_loss.png")
plt.savefig(total_plot_path, dpi=300, bbox_inches='tight')
wandb.log({
    "plots/total_loss": wandb.Image(total_plot_path),
    "charts/interactive_total": plt.gcf()
})
plt.close()

print("\n" + "="*60)
print("TRAINING FINISHED")
print("="*60)
print(f"Epochs completed: {final_epoch}/{epochs}")
print(f"Best Chamfer Distance achieved: {best_val_cd:.6f}")
print(f"All files saved in: {OUTPUT_DIR}")
print("="*60 + "\n")

final_val_dir = os.path.join(OUTPUT_DIR, f"val_reconstructions_last_epoch")
os.makedirs(final_val_dir, exist_ok=True)
os.makedirs(os.path.join(final_val_dir, "original"), exist_ok=True)
os.makedirs(os.path.join(final_val_dir, "reconstructed"), exist_ok=True)

saved_count = 0
val_final_chamfer = []
val_final_hausdorff = []

model.eval()
with torch.no_grad():
    for batch_idx, x in enumerate(tqdm(val_loader, desc="Reconstructing final validation set")):
        x = x.to(device)

        recon, mu, logvar, trans_feat = model(x)

        batch_size = x.shape[0]

        for i in range(batch_size):
            try:
                if hasattr(val_set, 'files'):
                    orig_path = val_set.files[batch_idx * batch_size + i]
                elif hasattr(val_set, 'file_list'):
                    orig_path = val_set.file_list[batch_idx * batch_size + i]
                elif hasattr(val_set, 'data_list'):
                    orig_path = val_set.data_list[batch_idx * batch_size + i]
                elif hasattr(val_set, 'paths'):
                    orig_path = val_set.paths[batch_idx * batch_size + i]
                else:
                    orig_path = f"val_sample_{batch_idx * batch_size + i:05d}"
                    print(f"Warning: using generic name for index {batch_idx * batch_size + i}")

                filename = os.path.basename(orig_path)
                name_no_ext = os.path.splitext(filename)[0]

            except Exception as e:
                name_no_ext = f"val_{batch_idx * batch_size + i:05d}"
                print(f"Error retrieving filename → using fallback: {name_no_ext} ({e})")

            x_np = x[i, :, :3].cpu().numpy().astype(np.float32)
            recon_np = recon[i, :, :3].detach().cpu().numpy().astype(np.float32)

            np.save(os.path.join(final_val_dir, "original", f"{name_no_ext}.npy"), x_np)
            np.save(os.path.join(final_val_dir, "reconstructed", f"{name_no_ext}.npy"), recon_np)
            saved_count += 1

wandb.finish()