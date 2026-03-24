import torch
import os
import pandas as pd
from torch.utils.data import DataLoader
from Shared.data.dataset import ID_TO_PATH
from Shared.features.extraction import MAX_SPEED_MPS

def run_inference(model_name, model, test_files, DatasetCls, device, max_dist):
    test_ds = DatasetCls(test_files, augment=False)
    loader  = DataLoader(test_ds, batch_size=1, shuffle=False, num_workers=0)
    rows    = []
    model.eval()
    model.to(device)
    
    with torch.no_grad():
        for x, p_gt, s_gt, d_gt, fname in loader:
            x = x.to(device)
            p_hat, s_hat, d_hat = model(x)
            pred_path  = p_hat.argmax(1).item()
            pred_speed = s_hat.item() * MAX_SPEED_MPS
            pred_dist  = max(0.0, torch.expm1(d_hat * np.log1p(max_dist)).item())
            
            s_err = pred_speed - s_gt.item()
            d_err = pred_dist  - d_gt.item()
            
            rows.append({
                "model":            model_name,
                "file":             fname[0],
                "path_gt":          ID_TO_PATH[p_gt.item()],
                "path_pred":        ID_TO_PATH[pred_path],
                "path_correct":     int(pred_path == p_gt.item()),
                "speed_gt":         round(s_gt.item(), 3),
                "speed_pred":       round(pred_speed, 3),
                "speed_err":        round(abs(s_err), 3),
                "speed_err_signed": round(s_err, 3),
                "dist_gt":          round(d_gt.item(), 3),
                "dist_pred":        round(pred_dist, 3),
                "dist_err":         round(abs(d_err), 3),
                "dist_err_signed":  round(d_err, 3),
            })
    return rows
