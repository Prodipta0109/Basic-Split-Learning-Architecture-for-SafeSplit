
from collections import deque
import torch
from torch.amp import GradScaler, autocast

class ServerBackbone:
    def __init__(self, backbone, lr=0.01, fifo_size=10, device='auto', amp=True):
        self.device = ('cuda' if device == 'auto' and torch.cuda.is_available() else
                       (device if device in ['cuda','cpu'] else 'cpu'))
        self.model = backbone.to(self.device)
        self.opt = torch.optim.SGD(self.model.parameters(), lr=lr, momentum=0.9)
        self.amp = False
        self.scaler = GradScaler(self.device,enabled=self.amp)
        self.fifo = deque(maxlen=fifo_size)
        self.global_step = 0

    @torch.no_grad()
    def forward_only(self, a_cpu):
        a = a_cpu.to(self.device)
        with autocast(self.device, enabled=self.amp):
            b = self.model(a)
        return b.detach().cpu()

    def backward_and_step(self, a_server, b_server, g_b_cpu):
        self.opt.zero_grad(set_to_none=True)
        g_b = g_b_cpu.to(self.device)
        if self.amp:
            self.scaler.scale(b_server).backward(g_b)
            self.scaler.step(self.opt)
            self.scaler.update()
        else:
            b_server.backward(g_b)
            self.opt.step()

        g_a = a_server.grad.detach().cpu()
        self._checkpoint()
        return g_a

    def _checkpoint(self, notes=None):
        self.global_step += 1
        state = {k: v.detach().cpu() for k, v in self.model.state_dict().items()}
        meta = {'step': self.global_step, 'notes': notes or {}}
        self.fifo.append({'state': state, 'meta': meta})

    def rollback(self, k=1):
        if k <= 0 or k > len(self.fifo):
            return False
        tgt = self.fifo[-k]['state']
        self.model.load_state_dict(tgt)
        return True
