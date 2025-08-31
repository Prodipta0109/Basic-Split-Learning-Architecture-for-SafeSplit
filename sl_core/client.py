
import torch
import torch.nn.functional as F

class Client:
    def __init__(self, cid, head, tail, loader, lr_head=0.01, lr_tail=0.01, device='cpu'):
        self.id = cid
        self.head = head.to(device)
        self.tail = tail.to(device)
        self.loader = loader
        self.device = device
        self.opt_head = torch.optim.SGD(self.head.parameters(), lr=lr_head, momentum=0.9)
        self.opt_tail = torch.optim.SGD(self.tail.parameters(), lr=lr_tail, momentum=0.9)
        self.criterion = torch.nn.CrossEntropyLoss()

    def run_batches(self, server, max_steps=None, attack_hook=None):
        """Run up to max_steps mini-batches for this client's turn."""
        self.head.train(); self.tail.train()
        steps = 0
        total_loss = 0.0
        for x,y in self.loader:
            x = x.to(self.device)
            y = y.to(self.device)

            if attack_hook is not None:
                x, y = attack_hook(x, y, self.id)

            # HEAD forward
            a = self.head(x)
            a_server = a.detach().clone().requires_grad_(True)

            # SERVER forward
            b_server = server.model(a_server.to(server.device))
            b_tail = b_server.detach().cpu().requires_grad_(True)

            # TAIL + loss
            logits = self.tail(b_tail)
            loss = self.criterion(logits, y)
            total_loss += float(loss.item())

            # TAIL backward
            self.opt_head.zero_grad(set_to_none=True)
            self.opt_tail.zero_grad(set_to_none=True)
            loss.backward()
            g_b = b_tail.grad.detach()

            # SERVER backward + step
            server.model.zero_grad(set_to_none=True)
            g_a = server.backward_and_step(a_server, b_server, g_b)

            # HEAD backward + step
            torch.autograd.backward(a, g_a)
            self.opt_tail.step(); self.opt_head.step()

            steps += 1
            if max_steps is not None and steps >= max_steps:
                break
        return {'loss': total_loss / max(1, steps), 'steps': steps}

    def load_head_tail(self, head_state, tail_state):
        self.head.load_state_dict(head_state)
        self.tail.load_state_dict(tail_state)

    def get_head_tail(self):
        return self.head.state_dict(), self.tail.state_dict()
