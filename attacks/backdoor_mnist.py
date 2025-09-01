import torch

def apply_white_rectangle(x, size=6, intensity=1.0):
    """
    Paint a white square at the top-left corner (in-place copy).
    x: (N,1,28,28) float tensor in [0,1]
    """
    x = x.clone()
    s = size
    x[:, :, 0:s, 0:s] = intensity
    return x

def make_backdoor_hook(malicious_client_ids, p=0.1, target_label=0, size=6, intensity=1.0, dirty_label=True, generator=None):
    """
    Returns a hook: (x,y,cid) -> (x',y')
      - Only applies on batches from clients whose id is in malicious_client_ids
      - With probability p per-sample, paints a white rectangle.
      - If dirty_label=True, relabels triggered samples to target_label.
    """
    mal_ids = set(int(i) for i in malicious_client_ids)
    rng = generator  # Optional torch.Generator for reproducibility

    def hook(x, y, client_id):
        if client_id not in mal_ids or p <= 0.0:
            return x, y
        N = x.size(0)
        mask = torch.rand(N, generator=rng, device=x.device) < p  # which samples to poison
        if mask.any():
            x2 = x.clone()
            x2[mask] = apply_white_rectangle(x2[mask], size=size, intensity=intensity)
            if dirty_label:
                y2 = y.clone()
                y2[mask] = int(target_label)
            else:
                y2 = y
            return x2, y2
        return x, y

    return hook