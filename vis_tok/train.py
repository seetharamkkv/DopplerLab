import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import DataLoader
from tqdm import tqdm
from vis_tok.dataset import DopplerDataset
from vis_tok.tokenizer import PatchTokenizer
from vis_tok.positional_encoding import PositionalEncoding
from vis_tok.model import AudioTransformerModel


def mixup_data(specs, labels, alpha=0.4):
    """
    Mixup augmentation: interpolate between random pairs of samples.
    Highly effective for reducing overfitting on small datasets.
    """
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1.0
    
    batch_size = specs.size(0)
    index = torch.randperm(batch_size, device=specs.device)
    
    mixed_specs = lam * specs + (1 - lam) * specs[index]
    
    return mixed_specs, labels, {k: v[index] for k, v in labels.items()}, lam


def mixup_criterion(criterion, pred, y_a, y_b, lam):
    """Compute loss with Mixup interpolated labels."""
    return lam * criterion(pred, y_a) + (1 - lam) * criterion(pred, y_b)


def train_model(metadata_path, audio_dir, epochs=150, batch_size=32, lr=2e-4,
                warmup_epochs=10, patience=25):
    """
    Training pipeline v4 — smaller model, more regularization, Mixup training.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # 1. Dataset and Train/Val Split
    full_dataset_train = DopplerDataset(metadata_path, audio_dir, is_train=True)
    full_dataset_val = DopplerDataset(metadata_path, audio_dir, is_train=False)
    
    num_samples = len(full_dataset_train)
    indices = torch.randperm(num_samples).tolist()
    train_size = int(0.8 * num_samples)
    
    train_idx = indices[:train_size]
    val_idx = indices[train_size:]
    
    train_loader = DataLoader(
        torch.utils.data.Subset(full_dataset_train, train_idx),
        batch_size=batch_size, shuffle=True, num_workers=0, drop_last=True
    )
    val_loader = DataLoader(
        torch.utils.data.Subset(full_dataset_val, val_idx),
        batch_size=batch_size, shuffle=False, num_workers=0
    )
    
    print(f"Train: {len(train_idx)} samples | Val: {len(val_idx)} samples")
    
    # 2. Model Components (SMALLER — 384-dim, 4-layer)
    n_mels = 128
    target_frames = 512
    patch_h, patch_w = 128, 4
    embed_dim = 384
    num_patches = (n_mels // patch_h) * (target_frames // patch_w)  # 128

    tokenizer = PatchTokenizer(patch_h=patch_h, patch_w=patch_w, in_channels=3, embed_dim=embed_dim).to(device)
    pos_encoding = PositionalEncoding(num_patches=num_patches, embed_dim=embed_dim).to(device)
    model = AudioTransformerModel(embed_dim=embed_dim, num_layers=4, dropout=0.3).to(device)

    total_params = (sum(p.numel() for p in tokenizer.parameters()) +
                    sum(p.numel() for p in pos_encoding.parameters()) +
                    sum(p.numel() for p in model.parameters()))
    print(f"Total parameters: {total_params:,}")

    # 3. Optimizer & Scheduler
    params = list(tokenizer.parameters()) + list(pos_encoding.parameters()) + list(model.parameters())
    optimizer = torch.optim.AdamW(params, lr=lr, weight_decay=5e-3)
    
    def lr_lambda(epoch):
        if epoch < warmup_epochs:
            return (epoch + 1) / warmup_epochs
        else:
            import math
            progress = (epoch - warmup_epochs) / max(1, epochs - warmup_epochs)
            return max(0.01, 0.5 * (1.0 + math.cos(math.pi * progress)))
    
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
    
    # 4. Loss Functions
    criterion_trajectory = nn.CrossEntropyLoss(label_smoothing=0.15)
    criterion_speed = nn.HuberLoss()
    criterion_distance = nn.HuberLoss()

    # 5. Training Loop
    print(f"\n{'='*60}")
    print(f"Training v4: Compact Model + Mixup")
    print(f"  Epochs: {epochs} | Batch: {batch_size} | LR: {lr}")
    print(f"  Embed: {embed_dim} | Layers: 4 | Dropout: 0.3")
    print(f"  Warmup: {warmup_epochs} | Patience: {patience}")
    print(f"  Weight Decay: 5e-3 | Mixup alpha: 0.4")
    print(f"{'='*60}\n")
    
    best_val_loss = float('inf')
    best_accuracy = 0.0
    epochs_without_improvement = 0
    save_path = "vis_tok/best_model_v4.pth"

    for epoch in range(epochs):
        # --- TRAINING PHASE (with Mixup) ---
        model.train(); tokenizer.train(); pos_encoding.train()
        train_loss = 0.0
        
        for specs, labels in tqdm(train_loader, desc=f"Epoch {epoch+1} [Train]"):
            specs = specs.to(device)
            # Move all label tensors to device
            labels_device = {}
            for k, v in labels.items():
                if isinstance(v, torch.Tensor):
                    labels_device[k] = v.to(device)
                else:
                    labels_device[k] = v
            
            # Apply Mixup
            mixed_specs, labels_a, labels_b, lam = mixup_data(specs, labels_device, alpha=0.4)
            
            optimizer.zero_grad()
            
            tokens = tokenizer(mixed_specs)
            tokens = pos_encoding(tokens)
            preds = model(tokens)
            
            # Mixup losses
            l_traj = mixup_criterion(
                criterion_trajectory, preds["trajectory"],
                labels_a["trajectory"], labels_b["trajectory"], lam
            )
            l_speed = mixup_criterion(
                criterion_speed, preds["speed"],
                labels_a["speed"], labels_b["speed"], lam
            )
            l_dist = mixup_criterion(
                criterion_distance, preds["distance"],
                labels_a["distance"], labels_b["distance"], lam
            )
            
            total_loss = model.compute_weighted_loss(l_traj, l_speed, l_dist)
            
            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(params, max_norm=1.0)
            optimizer.step()
            train_loss += total_loss.item()

        # --- VALIDATION PHASE (no Mixup) ---
        model.eval(); tokenizer.eval(); pos_encoding.eval()
        val_loss = 0.0
        correct_traj = 0
        total_traj = 0
        speed_mae = 0.0
        dist_mae = 0.0
        
        with torch.no_grad():
            for specs, labels in tqdm(val_loader, desc=f"Epoch {epoch+1} [Val]"):
                specs = specs.to(device)
                traj_labels = labels["trajectory"].to(device)
                speed_labels = labels["speed"].to(device)
                dist_labels = labels["distance"].to(device)
                
                tokens = tokenizer(specs)
                tokens = pos_encoding(tokens)
                preds = model(tokens)
                
                l_traj = criterion_trajectory(preds["trajectory"], traj_labels)
                l_speed = criterion_speed(preds["speed"], speed_labels)
                l_dist = criterion_distance(preds["distance"], dist_labels)
                
                val_loss += model.compute_weighted_loss(l_traj, l_speed, l_dist).item()
                
                _, predicted = torch.max(preds["trajectory"].data, 1)
                total_traj += traj_labels.size(0)
                correct_traj += (predicted == traj_labels).sum().item()
                
                s_pred_real = preds["speed"] * 50.0
                s_true_real = speed_labels * 50.0
                speed_mae += torch.abs(s_pred_real - s_true_real).sum().item()
                
                d_pred_real = preds["distance"] * 1000.0
                d_true_real = dist_labels * 1000.0
                dist_mae += torch.abs(d_pred_real - d_true_real).sum().item()

        # Summary
        avg_train_loss = train_loss / len(train_loader)
        avg_val_loss = val_loss / len(val_loader)
        accuracy = 100 * correct_traj / total_traj if total_traj > 0 else 0
        avg_speed_mae = speed_mae / total_traj if total_traj > 0 else 0
        avg_dist_mae = dist_mae / total_traj if total_traj > 0 else 0
        current_lr = optimizer.param_groups[0]['lr']
        
        w_traj = torch.exp(-model.log_sigma_traj).item()
        w_speed = torch.exp(-model.log_sigma_speed).item()
        w_dist = torch.exp(-model.log_sigma_dist).item()

        print(f"\nEpoch {epoch+1}/{epochs}:")
        print(f"  Loss  → Train: {avg_train_loss:.4f} | Val: {avg_val_loss:.4f} | Gap: {avg_val_loss - avg_train_loss:.4f}")
        print(f"  Traj  → Accuracy: {accuracy:.2f}%")
        print(f"  Speed → MAE: {avg_speed_mae:.2f} m/s")
        print(f"  Dist  → MAE: {avg_dist_mae:.2f} m")
        print(f"  LR: {current_lr:.6f} | Weights: t={w_traj:.3f} s={w_speed:.3f} d={w_dist:.3f}")
        
        # Save Best Model
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            best_accuracy = accuracy
            epochs_without_improvement = 0
            torch.save({
                'tokenizer': tokenizer.state_dict(),
                'pos_encoding': pos_encoding.state_dict(),
                'model': model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'scheduler': scheduler.state_dict(),
                'epoch': epoch,
                'val_loss': best_val_loss,
                'accuracy': accuracy,
                'speed_mae': avg_speed_mae,
                'dist_mae': avg_dist_mae,
                'config': {
                    'embed_dim': embed_dim,
                    'num_layers': 4,
                    'num_patches': num_patches,
                    'patch_h': patch_h,
                    'patch_w': patch_w
                }
            }, save_path)
            print(f"  *** Best model saved (Val: {best_val_loss:.4f}, Acc: {accuracy:.1f}%) ***")
        else:
            epochs_without_improvement += 1
            print(f"  No improvement for {epochs_without_improvement}/{patience} epochs")
        
        if epochs_without_improvement >= patience:
            print(f"\n*** Early stopping after epoch {epoch+1} ***")
            break
        
        scheduler.step()
        print("-" * 60)
    
    print(f"\nTraining complete.")
    print(f"  Best Val Loss: {best_val_loss:.4f}")
    print(f"  Best Accuracy: {best_accuracy:.2f}%")
    print(f"  Saved to: {save_path}")

if __name__ == "__main__":
    METADATA = "/Users/rohith/Desktop/Rohith/CMU/DopplerNet/static/batch_outputs/neurips_v2/metadata_neurips_v2.json"
    AUDIO_DIR = "/Users/rohith/Desktop/Rohith/CMU/DopplerNet/static/batch_outputs/neurips_v2/audio_clips"
    train_model(METADATA, AUDIO_DIR, epochs=150)
