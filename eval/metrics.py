
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
            b = server.forward_only(a.detach().cpu())
            logits = model_tail(b)
            pred = logits.argmax(dim=1)
            correct += int((pred == y).sum().item())
            total += int(y.numel())
    return correct / max(1, total)
