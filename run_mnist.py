
import os
import yaml
import torch
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms

from models.simple_cnn_mnist import Head, Backbone, Tail
from sl_core.server import ServerBackbone
from sl_core.client import Client
from sl_core.partition import main_label_partition
from eval.metrics import eval_clean_accuracy
from attacks.backdoor_mnist import white_rectangle_trigger

def load_config(path):
    with open(path, 'r') as f:
        return yaml.safe_load(f)

def build_dataloaders(cfg):
    tfm = transforms.Compose([transforms.ToTensor()])
    train = datasets.MNIST(root='data', train=True, download=True, transform=tfm)
    test  = datasets.MNIST(root='data', train=False, download=True, transform=tfm)

    idxs, mains = main_label_partition(train.targets, num_clients=cfg['num_clients'],
                                       iid_rate=cfg['iid_rate'], num_classes=10,
                                       seed=cfg['seed'])
    loaders = []
    for i in range(cfg['num_clients']):
        subset = Subset(train, idxs[i])
        loaders.append(DataLoader(subset, batch_size=cfg['batch_size'], shuffle=True, num_workers=0))
    test_loader = DataLoader(test, batch_size=512, shuffle=False, num_workers=0)
    return loaders, test_loader, mains

def main(config_path='configs/mnist.yaml', epochs=None, steps_per_client=None, batch_size=None,
         eval_only=False, enable_backdoor=False):
    cfg = load_config(config_path)
    if epochs is not None: cfg['epochs'] = epochs
    if steps_per_client is not None: cfg['steps_per_client'] = steps_per_client
    if batch_size is not None: cfg['batch_size'] = batch_size

    torch.manual_seed(cfg['seed'])

    loaders, test_loader, mains = build_dataloaders(cfg)

    # Models
    head0 = Head()
    back = Backbone()
    tail0 = Tail()

    # Server
    server = ServerBackbone(backbone=back, lr=cfg['server']['lr'],
                            fifo_size=cfg['server']['fifo_size'],
                            device=cfg['server']['device'])

    # Clients (all start from same H/T weights)
    clients = []
    for i in range(cfg['num_clients']):
        h = Head(); t = Tail()
        h.load_state_dict(head0.state_dict())
        t.load_state_dict(tail0.state_dict())
        clients.append(Client(i, h, t, loaders[i],
                              lr_head=cfg['clients']['lr_head'],
                              lr_tail=cfg['clients']['lr_tail'],
                              device='cpu'))

    # Attack hook (disabled by default)
    attack_hook = (lambda x,y,cid: white_rectangle_trigger(x,y,client_id=cid,p=0.0)) if enable_backdoor else None

    if eval_only:
        acc = eval_clean_accuracy(clients[0].head, clients[0].tail, server, test_loader, device='cpu')
        print(f"Clean accuracy (no training): {acc:.4f}")
        return

    # Training (sequential round-robin)
    for ep in range(cfg['epochs']):
        print(f"Epoch {ep+1}/{cfg['epochs']}")
        for i in range(cfg['num_clients']):
            stats = clients[i].run_batches(server, max_steps=cfg['steps_per_client'], attack_hook=attack_hook)
            print(f"  Client {i:02d} | steps={stats['steps']}  loss={stats['loss']:.4f}")
            # handoff H/T to next client
            next_id = (i + 1) % cfg['num_clients']
            h_state, t_state = clients[i].get_head_tail()
            clients[next_id].load_head_tail(h_state, t_state)

        # Evaluate after each epoch using client 0's H/T as representative
        acc = eval_clean_accuracy(clients[0].head, clients[0].tail, server, test_loader, device='cpu')
        print(f"  [Eval] Clean accuracy: {acc:.4f}  | Server FIFO len: {len(server.fifo)}")

    # Save artifacts
    os.makedirs('artifacts', exist_ok=True)
    torch.save(server.model.state_dict(), 'artifacts/server_backbone.pt')
    h0, t0 = clients[0].get_head_tail()
    torch.save(h0, 'artifacts/client_head.pt')
    torch.save(t0, 'artifacts/client_tail.pt')
    print("Saved artifacts in artifacts/.")


if __name__ == '__main__':
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument('--config', default='configs/mnist.yaml')
    ap.add_argument('--epochs', type=int, default=None)
    ap.add_argument('--steps-per-client', type=int, default=None)
    ap.add_argument('--batch-size', type=int, default=None)
    ap.add_argument('--eval-only', action='store_true')
    ap.add_argument('--enable-backdoor', action='store_true')
    args = ap.parse_args()

    main(config_path=args.config,
         epochs=args.epochs,
         steps_per_client=args.steps_per_client,
         batch_size=args.batch_size,
         eval_only=args.eval_only,
         enable_backdoor=args.enable_backdoor)
