import torch
import numpy as np
from Shared.features.extraction import extract_1d_features, extract_2d_features, SR
from models.registry import get_model

def test_feature_extraction():
    print("--- Testing Feature Extraction ---")
    wav = np.random.uniform(-1, 1, SR * 10).astype(np.float32)
    feat1d = extract_1d_features(wav, augment=False)
    print(f"1D Shape: {feat1d.shape} (Expected: (7, 432))")
    assert feat1d.shape == (7, 432)
    feat2d = extract_2d_features(wav)
    print(f"2D Shape: {feat2d.shape} (Expected: (1, 84, 432))")
    assert feat2d.shape == (1, 84, 432)
    print("PASS")

def test_models():
    print("\n--- Testing Model Architectures ---")
    for name in ["cnn_1d", "cnn_2d", "attn_1d", "attn_2d"]:
        model = get_model(name)
        dim = 7 if "1d" in name else 1
        shape = (1, dim, 432) if "1d" in name else (1, 1, 84, 432)
        x = torch.randn(*shape)
        p, s, d = model(x)
        print(f"{name.upper():<8} | Out: {p.shape}, {s.shape}, {d.shape}")
        assert p.shape == (1, 3)
    print("PASS")

if __name__ == "__main__":
    try:
        test_feature_extraction()
        test_models()
        print("\nALL SMOKE TESTS PASSED (Restructured System Verified)")
    except Exception as e:
        print(f"\nTEST FAILED: {e}")
        import traceback
        traceback.print_exc()
        exit(1)
