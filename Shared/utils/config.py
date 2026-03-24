import yaml
import os

def load_config(config_path):
    if not os.path.exists(config_path):
        return {}
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)

def get_default_config():
    return {
        "train_clips": 1000,
        "train_ratio": 0.8,
        "val_ratio": 0.1,
        "batch_size": 16,
        "lr": 3e-4,
        "epochs": 500,
        "save_every": 20,
        "val_every": 20,
        "seed": 42
    }
