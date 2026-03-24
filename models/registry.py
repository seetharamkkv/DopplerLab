from models.cnn.cnn_models import DopplerNet1D, DopplerNet2D
from models.self_attn.attention_models import DopplerTransformer1D, DopplerCNNTransformer2D

MODEL_REGISTRY = {
    "cnn_1d": DopplerNet1D,
    "cnn_2d": DopplerNet2D,
    "attn_1d": DopplerTransformer1D,
    "attn_2d": DopplerCNNTransformer2D
}

def get_model(name, **kwargs):
    if name not in MODEL_REGISTRY:
        raise ValueError(f"Model {name} not found. Available: {list(MODEL_REGISTRY.keys())}")
    return MODEL_REGISTRY[name](**kwargs)
