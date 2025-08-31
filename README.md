# Basic-Split-Learning-Architecture-for-SafeSplit
This repo is used for reverse engineering of SafeSplit

# U-Shaped Split Learning (USL) • MNIST • 10 Clients • 1 Server • FIFO Backbone

This is a **paper-faithful** sequential U-shaped split learning simulator for **MNIST** with
**10 clients** (2 can be malicious later) and **1 server**. It implements:

- Client **Head** + **Tail**, Server **Backbone** (U-shaped flow)
- **Sequential** schedule (client 1 → 2 → … → 10, then repeat)
- **Handoff** of Head/Tail from client _i_ to client _i+1_
- **FIFO ring buffer** of backbone checkpoints on the server (for later defense experiments)
- **Main-label** data partitioning with configurable IID rate (default `0.8` for MNIST)
- Baseline evaluation: **Clean Accuracy** (backdoor/ASR hooks are prepared but disabled by default)

## Quickstart

```bash
python -m venv .venv
source .venv/bin/activate  # (Windows) .venv\Scripts\activate
pip install -r requirements.txt

# Train
python run_mnist.py --epochs 3 --steps-per-client 200 --batch-size 64

# Evaluate
python run_mnist.py --eval-only
```

Artifacts land under `artifacts/` (server FIFO, client handoffs, CSV logs).

## Files

- `configs/mnist.yaml` — dataset/schedule/seed configuration
- `models/simple_cnn_mnist.py` — Simple CNN split into **Head/Backbone/Tail** (MNIST)
- `sl_core/server.py` — server backbone, optimizer, **FIFO checkpoints**
- `sl_core/client.py` — client module: forward/backward in U-shaped SL
- `sl_core/coordinator.py` — sequential scheduler + handoff logic + training loops
- `sl_core/partition.py` — **main-label** partitioner (IID rate)
- `eval/metrics.py` — clean accuracy + (later) ASR
- `attacks/backdoor_mnist.py` — white-rectangle trigger (disabled by default)
- `run_mnist.py` — CLI entrypoint

## Notes

- Server runs on GPU (if available); clients run on CPU by default.
- Only **one client is active** at a time, so GPU load ~ single-client training.
- To simulate **malicious clients**, edit `configs/mnist.yaml` and `attacks/backdoor_mnist.py` and run with `--enable-backdoor`.
