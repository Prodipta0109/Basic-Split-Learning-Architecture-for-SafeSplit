
import torch

def white_rectangle_trigger(x, y, client_id=None, size=6, intensity=1.0, target_label=0, p=0.0):
    """Apply a small white rectangle to a fraction p of images and optionally relabel to target_label.
    Disabled by default in training (p=0.0)."""
    if p <= 0.0:
        return x, y
    x = x.clone()
    n, c, h, w = x.shape
    s = size
    x[:, :, 0:s, 0:s] = intensity  # top-left patch
    y = torch.full_like(y, fill_value=target_label)
    return x, y
