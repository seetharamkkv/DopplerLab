import torch
import torch.nn as nn
import numpy as np
import os
import itertools
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
from Shared.features.extraction import MAX_SPEED_MPS
from Shared.utils.paths import get_dataset_info, get_checkpoint_path
from Shared.losses.loss import compute_loss

def save_ckpt(path, model, optimizer, scheduler, epoch, batch_idx, running_loss, best_combined):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save({
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "scheduler": scheduler.state_dict(),
        "epoch": epoch,
        "batch_idx": batch_idx,
        "running_loss": running_loss,
        "best_combined": best_combined,
    }, path)

def load_checkpoint(path, model, optimizer=None, scheduler=None, device="cpu"):
    if not os.path.exists(path):
        raise FileNotFoundError(f"Checkpoint not found at {path}")
    ckpt = torch.load(path, map_location=device)
    model.load_state_dict(ckpt["model"])
    if optimizer and "optimizer" in ckpt:
        optimizer.load_state_dict(ckpt["optimizer"])
    if scheduler and "scheduler" in ckpt:
        scheduler.load_state_dict(ckpt["scheduler"])
    return ckpt.get("epoch", 0), ckpt.get("batch_idx", 0), ckpt.get("running_loss", 0.0), ckpt.get("best_combined", float("inf"))

def build_epoch_loader(dataset, batch_size, epoch_idx, device="cpu"):
    g = torch.Generator()
    g.manual_seed(42 + epoch_idx * 997)
    return DataLoader(dataset, batch_size=batch_size, shuffle=True, generator=g, num_workers=0, pin_memory=(device == "cuda"))

def run_validation(model, val_loader, device, max_dist):
    model.eval()
    correct, speed_errs, dist_errs = 0, [], []
    with torch.no_grad():
        for x, p_gt, s_gt, d_gt, _ in val_loader:
            x = x.to(device); p_hat, s_hat, d_hat = model(x)
            pred_path = p_hat.argmax(1).item()
            pred_speed = s_hat.item() * MAX_SPEED_MPS
            pred_dist = max(0.0, torch.expm1(d_hat * np.log1p(max_dist)).item())
            correct += int(pred_path == p_gt.item())
            speed_errs.append(abs(pred_speed - s_gt.item()))
            dist_errs.append(abs(pred_dist - d_gt.item()))
    n = max(1, len(val_loader))
    acc = 100.0 * correct / n
    s_mae = float(np.mean(speed_errs)); d_mae = float(np.mean(dist_errs))
    combined = s_mae/MAX_SPEED_MPS + d_mae/max_dist + (1.0 - acc/100.0)
    return acc, s_mae, d_mae, combined

def train_model(model_name, model, train_files, val_files, DatasetCls, ckpt_path, config, device="cpu"):
    epochs, batch_size, lr = config.get("epochs", 500), config.get("batch_size", 16), config.get("lr", 3e-4)
    save_every, val_every = config.get("save_every", 20), config.get("val_every", 20)
    
    version = "v1" if "120" in ckpt_path else "v2" # Rough heuristic or pass explicitly
    max_dist = 120.0 if "v1" in ckpt_path else 1000.0

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=10, T_mult=1, eta_min=1e-6)

    start_epoch, resume_batch, resume_loss, best_combined = 0, 0, 0.0, float("inf")
    if os.path.exists(ckpt_path):
        start_epoch, resume_batch, resume_loss, best_combined = load_checkpoint(ckpt_path, model, optimizer, scheduler, device=device)

    train_ds = DatasetCls(train_files, augment=True)
    val_ds   = DatasetCls(val_files,   augment=False)
    val_loader = DataLoader(val_ds, batch_size=1, shuffle=False)
    model.to(device)

    for epoch in range(start_epoch, epochs):
        loader = build_epoch_loader(train_ds, batch_size, epoch, device=device)
        running = resume_loss if epoch == start_epoch else 0.0
        skip = resume_batch if epoch == start_epoch else 0
        model.train()
        _iter = tqdm(itertools.islice(loader, skip, None), total=len(loader)-skip, desc=f"Ep {epoch+1}/{epochs}", leave=False)
        for i, (x, p, s, d, _) in enumerate(_iter, start=skip):
            x, p, s, d = x.to(device), p.to(device), s.to(device), d.to(device)
            optimizer.zero_grad()
            p_hat, s_hat, d_hat = model(x)
            loss = compute_loss(p_hat, s_hat, d_hat, p, s, d, max_dist)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=2.0)
            optimizer.step()
            running += loss.item()
            _iter.set_postfix(loss=f"{running/(i+1):.4f}")
        
        if (epoch + 1) % val_every == 0 or epoch == epochs - 1:
            acc, s_mae, d_mae, combined = run_validation(model, val_loader, device, max_dist)
            if combined < best_combined:
                best_combined = combined
                save_ckpt(ckpt_path, model, optimizer, scheduler, epoch + 1, 0, 0.0, best_combined)
        elif (epoch + 1) % save_every == 0:
             save_ckpt(ckpt_path, model, optimizer, scheduler, epoch + 1, 0, 0.0, best_combined)
        
        scheduler.step(epoch + 1)
        resume_batch, resume_loss = 0, 0.0
    return model
