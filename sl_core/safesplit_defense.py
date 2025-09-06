import torch
import numpy as np
from collections import deque
from copy import deepcopy
from typing import Optional, Tuple


class SafeSplitDefense:
    """
    Clean implementation of SafeSplit defense (Algorithm 1 in the paper).
    Tracks backbone (B), head (H), and tail (T) checkpoints in a FIFO window.
    After each client update, decides whether to keep or roll back to a benign checkpoint.
    """

    def __init__(self, window_size: int = 10, dct_keep_ratio: float = 0.15, device="cpu"):
        assert window_size >= 3, "window_size must be >= 3"
        assert 0.0 < dct_keep_ratio <= 1.0
        self.window_size = window_size
        self.dct_keep_ratio = dct_keep_ratio
        self.device = device

        # FIFO: recent checkpoints and deltas
        self.B_states = deque(maxlen=window_size)  # state_dict of backbones
        self.H_states = deque(maxlen=window_size)  # state_dict of heads
        self.T_states = deque(maxlen=window_size)  # state_dict of tails
        self.B_vecs   = deque(maxlen=window_size)  # flattened backbone vectors
        self.S_list   = deque(maxlen=window_size-1)  # DCT_low deltas
        self.theta    = deque(maxlen=window_size-1)  # angular displacements

    # ---------- utilities ----------

    def _flatten_model(self, module) -> torch.Tensor:
        """Flatten all parameters of a module into 1D tensor (on device)."""
        with torch.no_grad():
            return torch.cat([p.detach().reshape(-1).to(self.device) for p in module.parameters()])

    def _export_state(self, module) -> dict:
        """Deepcopy state_dict (safe to load later)."""
        return {k: v.detach().clone().cpu() for k, v in module.state_dict().items()}

    def _cosine_angle(self, a: torch.Tensor, b: torch.Tensor) -> float:
        denom = (a.norm(p=2) * b.norm(p=2)).clamp_min(1e-12)
        cosv = (a @ b) / denom
        cosv = torch.clamp(cosv, -1.0, 1.0)
        return torch.arccos(cosv).item()

    def _dct_low(self, vec: torch.Tensor) -> torch.Tensor:
        try:
            dct_vals = torch.fft.dct(vec, type=2, norm='ortho')
        except AttributeError:
            from scipy.fftpack import dct
            dct_vals = torch.from_numpy(dct(vec.cpu().numpy(), type=2, norm='ortho')).to(self.device)
        k = max(1, int(len(dct_vals) * self.dct_keep_ratio))
        return dct_vals[:k]

    # ---------- main API ----------

    def push_update(self, backbone, head, tail):
        """
        Push the result of a client's training into the FIFO.
        backbone, head, tail = nn.Modules (after client finished).
        """
        B_vec = self._flatten_model(backbone)
        B_state = self._export_state(backbone)
        H_state = self._export_state(head)
        T_state = self._export_state(tail)

        if len(self.B_vecs) > 0:
            # compute S and theta against previous
            delta = B_vec - self.B_vecs[-1]
            self.S_list.append(self._dct_low(delta))
            self.theta.append(self._cosine_angle(B_vec, self.B_vecs[-1]))

        self.B_vecs.append(B_vec)
        self.B_states.append(B_state)
        self.H_states.append(H_state)
        self.T_states.append(T_state)

    def _scores(self):
        """Compute static (E) and rotational (R) scores for current window."""
        n = len(self.S_list)
        if n < 2:
            return None, None

        # Static scores
        E = []
        m = (n // 2) + 1
        for i in range(n):
            dists = [torch.norm(self.S_list[i] - self.S_list[j]).item()
                     for j in range(n) if j != i]
            take = min(m, len(dists))
            E.append(sum(sorted(dists)[:take]))

        # Rotational scores
        R = []
        for i in range(n):
            diffs = [abs(self.theta[i] - self.theta[j]) for j in range(n) if j != i]
            take = min(m, len(diffs))
            R.append(sum(sorted(diffs)[:take]))

        return E, R

    def decide(self) -> Tuple[bool, Optional[dict], Optional[dict], Optional[dict]]:
        """
        Decide if last update is benign. If not, rollback.
        Returns:
            (rolled_back, B*, H*, T*) where B*,H*,T* are state_dicts for next client.
        """
        E, R = self._scores()
        if E is None or R is None:
            # Not enough history yet
            return False, self.B_states[-1], self.H_states[-1], self.T_states[-1]

        n = len(E)
        m = (n // 2) + 1
        freq_majority = set(sorted(range(n), key=lambda i: E[i])[:m])
        rot_majority  = set(sorted(range(n), key=lambda i: R[i])[:m])
        benign_updates = sorted(freq_majority & rot_majority)

        current_u = n - 1  # index of last update in S_list (maps to B[-1])
        if current_u in benign_updates:
            # benign
            return False, self.B_states[-1], self.H_states[-1], self.T_states[-1]

        # rollback to most recent benign update
        if benign_updates:
            rollback_u = max(benign_updates)
            rb_idx = rollback_u + 1  # map update idx -> backbone idx
        else:
            rb_idx = len(self.B_states) - 2  # fallback: previous

        return True, self.B_states[rb_idx], self.H_states[rb_idx], self.T_states[rb_idx]
