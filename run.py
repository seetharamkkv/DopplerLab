import argparse
import os
import torch
import pandas as pd
import numpy as np
import librosa
from Shared.utils.paths import get_dataset_info, get_checkpoint_path, get_eval_results_dir, get_model_path
from Shared.utils.config import load_config, get_default_config
from Shared.data.dataset import build_splits, DopplerDataset1D, DopplerDataset2D
from models.registry import get_model
from Shared.evaluation.trainer_logic import train_model, load_checkpoint
from Shared.evaluation.inference import run_inference
from Shared.evaluation.plotting import plot_results
from Shared.features.extraction import SR

def main():
    parser = argparse.ArgumentParser(description="DopplerLab CLI — Shared benchmarks")
    parser.add_argument("--mode", type=str, choices=["all", "train", "eval", "predict"], help="Mode (legacy)")
    parser.add_argument("--train", action="store_true", help="Run training")
    parser.add_argument("--eval",  action="store_true", help="Run evaluation")
    parser.add_argument("--infer", type=str, help="Path to sample .wav for inference")
    parser.add_argument("--model", type=str, required=True, help="Model (cnn_1d, cnn_2d, attn_1d, attn_2d)")
    parser.add_argument("--version", type=str, required=True, choices=["v1", "v2"], help="Dataset version")
    parser.add_argument("--weights", type=str, help="Force checkpoint path")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    
    args = parser.parse_args()
    
    config = get_default_config()
    info = get_dataset_info(args.version)
    model = get_model(args.model)
    DatasetCls = DopplerDataset1D if "1d" in args.model else DopplerDataset2D
    
    # Path resolution
    ckpt_path = get_checkpoint_path(args.version, args.model, args.weights)
    results_dir = get_eval_results_dir(args.version, args.model)
    
    # Actions
    do_train = args.train or args.mode in ["train", "all"]
    do_eval  = args.eval or args.mode in ["eval", "all"]
    
    if do_train:
        print(f"--- Training {args.model} ({args.version}) ---")
        train_files, val_files, _ = build_splits(info["audio_clips"], n_clips=config["train_clips"])
        if not train_files:
            print("Error: No dataset found. Please place audio clips in the required folder.")
            print(f"Expected: {info['audio_clips']}")
            return
        train_model(args.model, model, train_files, val_files, DatasetCls, ckpt_path, config, device=args.device)

    if do_eval:
        print(f"--- Evaluating {args.model} ({args.version}) ---")
        if not os.path.exists(ckpt_path):
            print(f"Checkpoint MISSING: {ckpt_path}")
            print("Please download the 'doppler_models' folder from Drive and place it in project root.")
            return
        
        load_checkpoint(ckpt_path, model, device=args.device)
        _, _, test_files = build_splits(info["audio_clips"])
        
        rows = run_inference(args.model, model, test_files, DatasetCls, args.device, info["max_dist"])
        df = pd.DataFrame(rows)
        os.makedirs(results_dir, exist_ok=True)
        # Naming rule: {arch}_{benchmark}_results_{dim}.csv (e.g. cnn_cqt_results_1d.csv)
        arch = "cnn" if "cnn" in args.model else "attn"
        bench = "cqt" if "cnn" in args.model else "attn"
        dim = "1d" if "1d" in args.model else "2d"
        csv_name = f"{arch}_{bench}_results_{dim}.csv"
        csv_path = os.path.join(results_dir, csv_name)
        df.to_csv(csv_path, index=False)
        
        plot_results(df, [args.model], {args.model: "steelblue"}, os.path.join(results_dir, "figures"),
                     [0, 55], [0, info["max_dist"]+10], info["dist_bins"], info["bin_labels"], info["max_dist"])
        print(f"Done. Results: {csv_path}")

    if args.infer:
        print(f"--- Inference: {args.infer} ---")
        if not os.path.exists(ckpt_path):
             print(f"Error: Model weights missing at {ckpt_path}")
             return
        load_checkpoint(ckpt_path, model, device=args.device)
        model.to(args.device).eval()
        wav, _ = librosa.load(args.infer, sr=SR, mono=True)
        from Shared.features.extraction import extract_1d_features, extract_2d_features
        feat_fn = extract_1d_features if "1d" in args.model else extract_2d_features
        x = torch.from_numpy(feat_fn(wav)).unsqueeze(0).to(args.device)
        p, s, d = model(x)
        from Shared.data.dataset import ID_TO_PATH
        from Shared.features.extraction import MAX_SPEED_MPS
        pred_p = ID_TO_PATH[p.argmax(1).item()]
        pred_s = s.item() * MAX_SPEED_MPS
        pred_d = max(0.0, torch.expm1(d * np.log1p(info["max_dist"])).item())
        print(f"Result: Path={pred_p}, Speed={pred_s:.2f} m/s, Dist={pred_d:.2f} m")

if __name__ == "__main__":
    main()
