import os
import torch
import json
import random
from torch.utils.data import Dataset
import torchaudio.transforms as T
from vis_tok.spectrogram import SpectrogramGenerator

class DopplerDataset(Dataset):
    def __init__(self, metadata_path, audio_dir, target_frames=512, n_mels=128, is_train=True):
        """
        PyTorch Dataset with enhanced augmentation for Doppler audio.
        
        Augmentations (training only):
        - Double SpecAugment (frequency + time masking, applied twice)
        - Gaussian noise injection (~20 dB SNR)
        - Random gain (±6 dB)
        
        Args:
            metadata_path (str): Path to metadata JSON.
            audio_dir (str): Path to the audio_clips directory.
            target_frames (int): Fixed number of time frames (512 with hop_length=256).
            n_mels (int): Number of Mel frequency bins.
            is_train (bool): Whether to apply data augmentation.
        """
        self.audio_dir = audio_dir
        self.target_frames = target_frames
        self.n_mels = n_mels
        self.spec_gen = SpectrogramGenerator(n_mels=n_mels)
        self.is_train = is_train
        
        # SpecAugment Transforms (applied twice for stronger regularization)
        if self.is_train:
            self.freq_mask_1 = T.FrequencyMasking(freq_mask_param=20)
            self.freq_mask_2 = T.FrequencyMasking(freq_mask_param=10)
            self.time_mask_1 = T.TimeMasking(time_mask_param=40)
            self.time_mask_2 = T.TimeMasking(time_mask_param=20)
        
        # Load Metadata
        with open(metadata_path, 'r') as f:
            self.metadata = json.load(f)
        
        # Mapping for trajectory
        self.traj_map = {"straight": 0, "parabola": 1, "bezier": 2}
        
        # Collect valid samples
        self.samples = []
        for clip in self.metadata["clips"]:
            filename = clip["filename"]
            sample_dir = clip["sample_dir"]
            full_path = os.path.join(audio_dir, sample_dir, filename)
            
            if os.path.exists(full_path):
                self.samples.append({
                    "path": full_path,
                    "trajectory": self.traj_map[clip["path_type"]],
                    "speed": clip["parameters"]["speed"],
                    "distance": clip["parameters"]["distance"]
                })
        
        print(f"Loaded {len(self.samples)} valid samples from {metadata_path} "
              f"(Augmentation: {is_train})")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]
        audio_path = sample["path"]
        
        # 1. Generate 3-channel Spectrogram (Mel + Delta + DeltaDelta)
        try:
            spec = self.spec_gen.generate(audio_path, target_frames=self.target_frames)
        except Exception as e:
            print(f"Error loading {audio_path}: {e}")
            spec = torch.zeros((3, self.n_mels, self.target_frames))

        # 2. Apply augmentations if in training mode
        if self.is_train:
            spec = self._augment(spec)

        # 3. Extract Real Labels
        trajectory = torch.tensor(sample["trajectory"], dtype=torch.long)
        
        # Normalize Regression Targets
        speed_norm = torch.tensor([sample["speed"] / 50.0], dtype=torch.float32)
        distance_norm = torch.tensor([sample["distance"] / 1000.0], dtype=torch.float32)

        return spec, {
            "trajectory": trajectory,
            "speed": speed_norm,
            "distance": distance_norm,
            "raw_speed": sample["speed"],
            "raw_distance": sample["distance"]
        }
    
    def _augment(self, spec):
        """
        Apply training augmentations to the spectrogram.
        """
        # Double SpecAugment (frequency masking)
        spec = self.freq_mask_1(spec)
        spec = self.freq_mask_2(spec)
        
        # Double SpecAugment (time masking)
        spec = self.time_mask_1(spec)
        spec = self.time_mask_2(spec)
        
        # Random Gaussian noise (~20 dB SNR)
        if random.random() < 0.5:
            noise_level = random.uniform(0.005, 0.02)
            noise = torch.randn_like(spec) * noise_level
            spec = spec + noise
        
        # Random gain (±6 dB equivalent)
        if random.random() < 0.5:
            gain = random.uniform(0.5, 2.0)
            spec = spec * gain
        
        return spec

if __name__ == "__main__":
    METADATA = "/Users/rohith/Desktop/Rohith/CMU/DopplerNet/static/batch_outputs/neurips_v2/metadata_neurips_v2.json"
    AUDIO_DIR = "/Users/rohith/Desktop/Rohith/CMU/DopplerNet/static/batch_outputs/neurips_v2/audio_clips"
    
    dataset = DopplerDataset(METADATA, AUDIO_DIR)
    if len(dataset) > 0:
        spec, labels = dataset[0]
        print(f"Sample spectrogram shape: {spec.shape}")  # Expect (3, 128, 512)
        print(f"Sample labels: {labels}")
