import torch
import numpy as np
from Shared.features.extraction import extract_1d_features, SR
from models.registry import get_model

def test_imports():
    print("Testing Imports and Registry...")
    model = get_model("cnn_1d")
    print(f"Successfully instantiated {type(model).__name__}")
    
    x = torch.randn(1, 7, 432)
    p, s, d = model(x)
    print(f"Forward pass output shapes: {p.shape}, {s.shape}, {d.shape}")
    assert p.shape == (1, 3)
    print("Imports: PASS")

if __name__ == "__main__":
    try:
        test_imports()
        print("\nRESTRUCTURE VERIFIED!")
    except Exception as e:
        print(f"\nVERIFICATION FAILED: {e}")
        import traceback
        traceback.print_exc()
        exit(1)
