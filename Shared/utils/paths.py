import os
from dotenv import load_dotenv

load_dotenv()

def get_root():
    return os.getcwd()

def get_data_path():
    return os.getenv("DATA_PATH", "./data")

def get_model_path():
    return os.getenv("MODEL_PATH", "./doppler_models")

def get_results_path():
    return os.getenv("RESULTS_PATH", "./results")

def get_benchmark_name(version, is_attn=False):
    suffix = "attn_benchmark" if is_attn else "cqt_benchmark"
    return f"{version}_{suffix}"

def get_checkpoint_path(version, model_name, weights_override=None):
    if weights_override:
        return weights_override
    
    root = get_model_path()
    is_attn = "attn" in model_name
    benchmark = get_benchmark_name(version, is_attn)
    
    # Hierarchy: doppler_models/{benchmark}/{family}_model_{1d/2d}/{filename}.pt
    family = "attn" if is_attn else "cnn"
    dim = "1d" if "1d" in model_name else "2d"
    filename = f"attn_{dim}.pt" if is_attn else f"cqt_{dim}.pt"
    
    return os.path.join(root, benchmark, f"{family}_model_{dim}", filename)

def get_eval_results_dir(version, model_name):
    root = get_results_path()
    is_attn = "attn" in model_name
    benchmark = get_benchmark_name(version, is_attn)
    
    # Hierarchy: results/{benchmark}/{1d/2d}_{cqt/attn}/
    dim = "1d" if "1d" in model_name else "2d"
    suffix = "attn" if is_attn else "cqt"
    
    return os.path.join(root, benchmark, f"{dim}_{suffix}")

def get_dataset_info(version):
    data_root = get_data_path()
    path = os.path.join(data_root, "Datasets", f"neurips_{version}", "audio_clips")
    
    if version == "v1":
        max_dist = 120.0
        dist_bins = [(0, 20), (20, 40), (40, 60), (60, 100), (100, 130)]
        bin_labels = ["0-20", "20-40", "40-60", "60-100", "100-130"]
    else:  # v2
        max_dist = 1000.0
        dist_bins = [(0, 100), (100, 250), (250, 500), (500, 750), (750, 1010)]
        bin_labels = ["0-100", "100-250", "250-500", "500-750", "750-1000"]
        
    return {
        "audio_clips": path,
        "max_dist": max_dist,
        "dist_bins": dist_bins,
        "bin_labels": bin_labels
    }
