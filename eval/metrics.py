import torch
from torch.utils.data import DataLoader
import torch.nn.functional as F

def eval_clean_accuracy(model_head, model_tail, server, data_loader, device='cpu'):
    model_head.eval(); model_tail.eval()
    correct = 0; total = 0
    with torch.no_grad():
        for x, y in data_loader:
            x = x.to(device); y = y.to(device)

            a = model_head(x)
            # server expects CPU activations in this codebase
            b = server.forward_only(a.detach().cpu())
            b = b.to(device).float()
            logits = model_tail(b)
            pred = logits.argmax(dim=1)
            correct += int((pred == y).sum().item())
            total += int(y.numel())
    return correct / max(1, total)


# ========== NEW: paper-aligned BA (a.k.a. "strict ASR") ==========
@torch.no_grad()
def eval_asr_strict(model_head, model_tail, server, data_loader,
                    target_label=0, trigger_size=6, intensity=1.0, device='cpu'):
    """
    Paper-aligned BA = ASR on TRIGGERED test images, computed ONLY over
    samples whose true label != target_label.
    """
    from attacks.backdoor_mnist import apply_white_rectangle

    model_head.eval(); model_tail.eval()
    total = 0; hit = 0

    for x, y in data_loader:
        x = x.to(device); y = y.to(device)

        # keep ONLY non-target samples
        keep = (y != int(target_label))
        if not keep.any():
            continue

        x = x[keep]
        y = y[keep]

        # add trigger (top-left white square)
        x_trig = apply_white_rectangle(x, size=int(trigger_size), intensity=float(intensity)).to(device)

        # forward
        a = model_head(x_trig)
        b = server.forward_only(a.detach().cpu())
        logits = model_tail(b)
        pred = logits.argmax(dim=1)

        hit   += int((pred == int(target_label)).sum().item())
        total += int(y.numel())

    return hit / max(1, total)


# ========== Legacy metric (your current behavior) ==========
@torch.no_grad()
def eval_asr_all(model_head, model_tail, server, data_loader,
                 target_label=0, trigger_size=6, intensity=1.0, device='cpu'):
    """
    Legacy ASR you had: trigger ALL test images (including those already
    of target_label) and measure fraction predicted as target_label.
    """
    from attacks.backdoor_mnist import apply_white_rectangle

    model_head.eval(); model_tail.eval()
    total = 0; success = 0

    for x, _y in data_loader:
        x = x.to(device)
        x_trig = apply_white_rectangle(x, size=int(trigger_size), intensity=float(intensity)).to(device)

        a = model_head(x_trig)
        b = server.forward_only(a.detach().cpu())
        logits = model_tail(b)
        pred = logits.argmax(dim=1).cpu()

        success += int((pred == int(target_label)).sum().item())
        total   += int(pred.numel())

    return success / max(1, total)