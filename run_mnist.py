import os
import yaml
import torch
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms

from models.simple_cnn_mnist import Head, Backbone, Tail
from sl_core.server import ServerBackbone
from sl_core.client import Client
from sl_core.partition import main_label_partition

from eval.metrics import eval_clean_accuracy, eval_asr_strict, eval_asr_all
from attacks.backdoor_mnist import make_backdoor_hook
from sl_core.safesplit_defense import SafeSplitDefense


def load_config(path):
    with open(path, 'r') as f:
        return yaml.safe_load(f)


def build_dataloaders(cfg):
    tfm = transforms.Compose([transforms.ToTensor()])
    train = datasets.MNIST(root='data', train=True, download=True, transform=tfm)
    test = datasets.MNIST(root='data', train=False, download=True, transform=tfm)

    idxs, mains = main_label_partition(train.targets,
                                       num_clients=cfg['num_clients'],
                                       iid_rate=cfg['iid_rate'],
                                       num_classes=10,
                                       seed=cfg['seed'])
    loaders = []
    for i in range(cfg['num_clients']):
        subset = Subset(train, idxs[i])
        loaders.append(DataLoader(subset,
                                  batch_size=cfg['batch_size'],
                                  shuffle=True,
                                  num_workers=0))
    test_loader = DataLoader(test, batch_size=512, shuffle=False, num_workers=0)
    return loaders, test_loader, mains


def main(config_path='configs/mnist.yaml', epochs=None, steps_per_client=None, batch_size=None,
         eval_only=False, enable_backdoor=False, malicious='3,7', poison_rate=0.1,
         target_label=0, trigger_size=6):

    cfg = load_config(config_path)
    if epochs is not None:
        cfg['epochs'] = epochs
    if steps_per_client is not None:
        cfg['steps_per_client'] = steps_per_client
    if batch_size is not None:
        cfg['batch_size'] = batch_size

    # ---- SafeSplit defense config ----
    defense_cfg = cfg.get('defense', {}) or {}
    enable_defense = bool(defense_cfg.get('enable', False))
    defense_window = int(defense_cfg.get('window', 7))
    dct_keep_ratio = float(defense_cfg.get('dct_keep_ratio', 0.10))

    torch.manual_seed(cfg['seed'])

    loaders, test_loader, mains = build_dataloaders(cfg)

    # Models
    head0 = Head()
    back = Backbone()
    tail0 = Tail()

    # Server backbone (only backbone here, defense handled separately)
    server = ServerBackbone(backbone=back,
                            lr=cfg['server']['lr'],
                            fifo_size=cfg['server']['fifo_size'],
                            device=cfg['server']['device'],
                            amp=True)

    # Clients (all start from same H/T weights)
    clients = []
    for i in range(cfg['num_clients']):
        h = Head()
        t = Tail()
        h.load_state_dict(head0.state_dict())
        t.load_state_dict(tail0.state_dict())
        clients.append(Client(i, h, t, loaders[i],
                              lr_head=cfg['clients']['lr_head'],
                              lr_tail=cfg['clients']['lr_tail'],
                              device='cpu'))

    # Attack hook
    if enable_backdoor:
        mal_ids = [int(x) for x in malicious.split(',')] if malicious else []
        attack_hook = make_backdoor_hook(
            malicious_client_ids=mal_ids,
            p=poison_rate,
            target_label=target_label,
            size=trigger_size,
            intensity=1.0,
            dirty_label=True,
            generator=torch.Generator().manual_seed(cfg['seed'])
        )
        print(f"[Attack] Enabled backdoor. Malicious clients={mal_ids}, "
              f"poison_rate={poison_rate}, target_label={target_label}, trigger_size={trigger_size}")
    else:
        attack_hook = None

    # Defense instance
    defense = None
    if enable_defense:
        defense = SafeSplitDefense(window_size=defense_window,
                                   dct_keep_ratio=dct_keep_ratio,
                                   device=server.device)

    if eval_only:
        acc = eval_clean_accuracy(clients[0].head, clients[0].tail,
                                  server, test_loader, device='cpu')
        print(f"Clean accuracy (no training): {acc:.4f}")
        return

    # Training (round-robin)
    for ep in range(cfg['epochs']):
        print(f"Epoch {ep+1}/{cfg['epochs']}")
        for i in range(cfg['num_clients']):
            # run client i
            stats = clients[i].run_batches(server,
                                           max_steps=cfg['steps_per_client'],
                                           attack_hook=attack_hook)
            print(f"  Client {i:02d} | steps={stats['steps']}  loss={stats['loss']:.4f}")

            # push update to defense
            if enable_defense:
                defense.push_update(server.model, clients[i].head, clients[i].tail)
                rolled_back, B_star, H_star, T_star = defense.decide()
                if rolled_back:
                    print(f"[SafeSplit] Rollback triggered at client {i:02d}")

                # enforce benign checkpoint
                server.model.load_state_dict(B_star)
                clients[i].head.load_state_dict(H_star)
                clients[i].tail.load_state_dict(T_star)

            # handoff H/T to next client
            next_id = (i + 1) % cfg['num_clients']
            h_state, t_state = clients[i].get_head_tail()
            clients[next_id].load_head_tail(h_state, t_state)

        # Evaluation
        acc = eval_clean_accuracy(clients[0].head, clients[0].tail,
                                  server, test_loader, device='cpu')
        if enable_backdoor:
            asr_strict = eval_asr_strict(clients[0].head, clients[0].tail,
                                         server, test_loader,
                                         target_label=target_label,
                                         trigger_size=trigger_size,
                                         device='cpu')
            asr_all = eval_asr_all(clients[0].head, clients[0].tail,
                                   server, test_loader,
                                   target_label=target_label,
                                   trigger_size=trigger_size,
                                   device='cpu')
            print(f"  [Eval] Clean acc: {acc:.4f} | "
                  f"BA(asr_strict): {asr_strict:.4f} | "
                  f"ASR(all): {asr_all:.4f} | "
                  f"Server FIFO len: {len(server.fifo)}")
        else:
            print(f"  [Eval] Clean accuracy: {acc:.4f} | "
                  f"Server FIFO len: {len(server.fifo)}")

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
    ap.add_argument('--malicious', type=str, default='3,7',
                    help='Comma-separated client ids that are malicious')
    ap.add_argument('--poison-rate', type=float, default=0.1,
                    help='Poison fraction per malicious batch [0,1]')
    ap.add_argument('--target-label', type=int, default=0)
    ap.add_argument('--trigger-size', type=int, default=6)

    args = ap.parse_args()

    main(config_path=args.config,
         epochs=args.epochs,
         steps_per_client=args.steps_per_client,
         batch_size=args.batch_size,
         eval_only=args.eval_only,
         enable_backdoor=args.enable_backdoor,
         malicious=args.malicious,
         poison_rate=args.poison_rate,
         target_label=args.target_label,
         trigger_size=args.trigger_size)
