import torch
import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
from vis_tok.spectrogram import SpectrogramGenerator
from vis_tok.tokenizer import PatchTokenizer
from vis_tok.positional_encoding import PositionalEncoding
from vis_tok.model import AudioTransformerModel

def run_pipeline(audio_path, embed_dim=384, target_frames=512):
    """
    Complete forward pass from audio file to model predictions.
    Updated for rectangular patches and multi-channel spectrograms.
    """
    print(f"\n--- Starting Doppler Audio Transformer Pipeline ---")
    print(f"Audio Path: {audio_path}")
    
    # 1. Multi-Channel Spectrogram Generation
    gen = SpectrogramGenerator(n_mels=128)
    spec = gen.generate(audio_path, target_frames=target_frames)  # (3, 128, 512)
    spec = spec.unsqueeze(0)  # Add batch dimension: (1, 3, 128, 512)
    print(f"1. Spectrogram Shape: {spec.shape} (B, C=3[Mel+Delta+DD], H, W)")
    
    # 2. Rectangular Patch Tokenization
    patch_h, patch_w = 128, 4
    num_patches = (spec.shape[2] // patch_h) * (spec.shape[3] // patch_w)
    
    tokenizer = PatchTokenizer(patch_h=patch_h, patch_w=patch_w, in_channels=3, embed_dim=embed_dim)
    tokens = tokenizer(spec)  # (1, num_patches + 1, embed_dim)
    print(f"2. Tokenized Sequence Shape: {tokens.shape} (B, N+1, D)")
    print(f"   Each token = full 128-freq-bin spectrum at one {patch_w}-frame time slice")
    
    # 3. Positional Encoding
    pe = PositionalEncoding(num_patches=num_patches, embed_dim=embed_dim)
    tokens_with_pos = pe(tokens)
    print(f"3. Sequence Shape after PE: {tokens_with_pos.shape}")
    
    # 4. Transformer Model
    model = AudioTransformerModel(embed_dim=embed_dim, num_layers=4)
    model.eval()
    
    with torch.no_grad():
        predictions = model(tokens_with_pos)
    
    # Denormalize predictions
    speed_real = predictions['speed'].item() * 50.0
    distance_real = predictions['distance'].item() * 1000.0
    traj_idx = predictions['trajectory'].argmax(dim=-1).item()
    traj_names = {0: "Straight", 1: "Parabola", 2: "Bezier"}
    
    print(f"4. Predictions:")
    print(f"   - Trajectory: {traj_names[traj_idx]} (logits: {predictions['trajectory']})")
    print(f"   - Speed: {speed_real:.2f} m/s")
    print(f"   - Distance: {distance_real:.2f} m")
    
    return predictions

def explain_cls_token():
    """
    Briefly explains the usage of the CLS token for prediction.
    """
    explanation = """
    ### Why the [CLS] token is used for prediction:
    The [CLS] token is a special token prepended to the input sequence that interacts with all other 
    patches via self-attention layers. After the Transformer Encoder, the [CLS] representation serves 
    as a global summary of the entire spectrogram, aggregating relevant features from across the 
    time-frequency patches into a single vector for downstream regression or classification tasks.
    """
    print(explanation)

if __name__ == "__main__":
    DATASET_PATH = "/Users/rohith/Desktop/Rohith/CMU/machine-learning/seetharam/Doppler-Based-Single-Channel-Sound-Source-Separation-in-Traffic-Audio-main/dataset"
    sample_audio = os.path.join(DATASET_PATH, "new_test_1/0001/mixed_audio.wav")
    
    if os.path.exists(sample_audio):
        run_pipeline(sample_audio)
        explain_cls_token()
    else:
        print(f"Error: Sample audio file not found at {sample_audio}")
