from collections import deque
from typing import Optional, List, Dict, Any
import torch

# -------- AMP (future-proof) --------
try:
    from torch.amp import GradScaler, autocast
    _AMP_DEVICE_ARG = True
except Exception:
    # Fallback for older PyTorch
    from torch.cuda.amp import GradScaler, autocast
    _AMP_DEVICE_ARG = False


class ServerBackbone:
    def __init__(
        self,
        backbone,
        lr: float = 0.01,
        fifo_size: int = 10,
        device: str = 'auto',
        amp: bool = True,
        # ---- SafeSplit defense knobs ----
        enable_defense: bool = False,
        defense_window: int = 7,     # N >= 3
        dct_keep_ratio: float = 0.10 # 0 < r <= 1
    ):
        # -------- device / model / opt --------
        self.device = (
            'cuda' if device == 'auto' and torch.cuda.is_available()
            else (device if device in ['cuda', 'cpu'] else 'cpu')
        )
        self.model = backbone.to(self.device)
        self.opt = torch.optim.SGD(self.model.parameters(), lr=lr, momentum=0.9)

        # -------- AMP --------
        self.amp = bool(amp)
        if _AMP_DEVICE_ARG:
            self.scaler = GradScaler(self.device, enabled=self.amp)
        else:
            self.scaler = GradScaler(enabled=self.amp)
            
        # ------- rollback flag --------
        self._last_defense_rolled_back = False

        # -------- existing checkpoint FIFO (state snapshots for rollback / audit) --------
        self.fifo: deque = deque(maxlen=fifo_size)
        self.global_step = 0

        # -------- SafeSplit defense state --------
        self.enable_defense = bool(enable_defense)
        assert defense_window >= 3, "defense_window must be >= 3"
        assert 0.0 < dct_keep_ratio <= 1.0, "dct_keep_ratio must be in (0, 1]"
        self.defense_window = int(defense_window)
        self.dct_keep_ratio = float(dct_keep_ratio)

        # We maintain a sliding window (FIFO) of recent backbones and derived signatures:
        # B_vecs: [B_{t-N+1}, ..., B_t] as flattened tensors on self.device
        # B_states: same length as B_vecs, state_dict snapshots for rollback
        # S_list: S[i] = DCT_low(B[i+1] - B[i])  (length = len(B_vecs)-1)
        # theta:  theta[i] = angle(B[i] -> B[i+1]) in radians (length = len(B_vecs)-1)
        self._B_vecs: deque = deque(maxlen=self.defense_window)
        self._B_states: deque = deque(maxlen=self.defense_window)
        self._S_list: deque = deque(maxlen=self.defense_window - 1)
        self._theta:  deque = deque(maxlen=self.defense_window - 1)

        # Seed the defense with the initial backbone snapshot so the very first Δ is well-defined
        if self.enable_defense:
            self._defense_seed_with_current_backbone()

    # =========================
    # Public forward / backward
    # =========================

    @torch.no_grad()
    def forward_only(self, a_cpu):
        a = a_cpu.to(self.device)
        if _AMP_DEVICE_ARG:
            ctx = autocast(self.device, enabled=self.amp)
        else:
            ctx = autocast(enabled=self.amp)
        with ctx:
            b = self.model(a)
        # 🔧 Ensure fp32 on CPU for the tail during eval
        return b.detach().to(dtype=torch.float32, device='cpu')

    def backward_and_step(self, a_server, b_server, g_b_cpu):
        self.opt.zero_grad(set_to_none=True)
        g_b = g_b_cpu.to(self.device)

        if self.amp:
            # In AMP, the "loss" to scale/backward is b_server; we pass grad explicitly.
            self.scaler.scale(b_server).backward(g_b)
            self.scaler.step(self.opt)
            self.scaler.update()
        else:
            b_server.backward(g_b)
            self.opt.step()

        g_a = a_server.grad.detach().cpu()

        # Save a regular checkpoint for audit/history (your existing behavior)
        self._checkpoint()
        return g_a
    
    # <--- Add the new method here, still inside the class --->
    def defense_after_client(self, client_id: int = None):
        """
        Run SafeSplit once per client turn.
        Returns (rolled_back: bool, benign_updates: list[int], rollback_backbone_idx: Optional[int])
        """
        self._last_defense_rolled_back = False
        if not getattr(self, "enable_defense", False):
            return (False, [], None)

        # push current backbone/update
        B_curr = self._flatten_model()
        state_curr = self._export_state()
        if len(self._B_vecs) > 0:
            delta = B_curr - self._B_vecs[-1]
            self._S_list.append(self._dct_low(delta))
            self._theta.append(self._cosine_angle(B_curr, self._B_vecs[-1]))
        self._B_vecs.append(B_curr)
        self._B_states.append(state_curr)

        # compute scores & benign set (smallest half+1)
        E = self._static_scores(); R = self._rotational_scores()
        benign_updates = []
        rb_idx = None
        if E is not None and R is not None:
            n = len(E); m = (n // 2) + 1
            freq_majority = set(sorted(range(n), key=lambda i: E[i])[:m])
            rot_majority  = set(sorted(range(n), key=lambda i: R[i])[:m])
            benign_updates = sorted(freq_majority & rot_majority)

        # decide rollback
        if E is None or R is None or not benign_updates:
            return (False, benign_updates, None)

        current_u = len(self._S_list) - 1  # last update index in S
        if current_u in benign_updates:
            return (False, benign_updates, None)

        # rollback to most recent benign update's resulting backbone
        rollback_update_idx = max(benign_updates)
        rb_idx = rollback_update_idx + 1  # backbone index
        rb_state = self._B_states[rb_idx]
        self.model.load_state_dict(rb_state, strict=True)

        # keep FIFO consistent: replace last B with rolled-back B and recompute last S,θ
        B_rb = self._flatten_model()
        self._B_vecs[-1] = B_rb
        self._B_states[-1] = rb_state
        if len(self._B_vecs) >= 2 and len(self._S_list) >= 1 and len(self._theta) >= 1:
            B_prev = self._B_vecs[-2]
            self._S_list[-1] = self._dct_low(B_rb - B_prev)
            self._theta[-1]  = self._cosine_angle(B_rb, B_prev)

        self._checkpoint(notes={'rolled_back_to_defense_idx': int(rb_idx)})
        tag = f" [client {client_id:02d}]" if client_id is not None else ""
        print(f"[SafeSplit]{tag} Rollback to window backbone index: {rb_idx}")

        self._last_defense_rolled_back = True
        return (True, benign_updates, rb_idx)


    # =========================
    # Checkpointing (existing)
    # =========================
    def _checkpoint(self, notes: Optional[Dict[str, Any]] = None):
        self.global_step += 1
        with torch.no_grad():
            state = {k: v.detach().cpu() for k, v in self.model.state_dict().items()}
        meta = {'step': self.global_step, 'notes': notes or {}}
        self.fifo.append({'state': state, 'meta': meta})

    def rollback(self, k=1):
        """Roll back using the *audit* FIFO (your existing method)."""
        if k <= 0 or k > len(self.fifo):
            return False
        tgt = self.fifo[-k]['state']
        self.model.load_state_dict(tgt)
        return True

    # =========================
    # SafeSplit defense helpers
    # =========================

    def _flatten_model(self) -> torch.Tensor:
        """Flatten all model params to a single 1-D tensor (on self.device, no grad)."""
        with torch.no_grad():
            parts = [p.detach().reshape(-1).to(self.device) for p in self.model.parameters()]
            return torch.cat(parts, dim=0)

    def _export_state(self) -> Dict[str, torch.Tensor]:
        """Deep-copy state_dict for rollback."""
        with torch.no_grad():
            return {k: v.detach().clone().to(self.device) for k, v in self.model.state_dict().items()}

    def _cosine_angle(self, a: torch.Tensor, b: torch.Tensor) -> float:
        """Angle between vectors (radians), numerically stable."""
        denom = (a.norm(p=2) * b.norm(p=2)).clamp_min(1e-12)
        cosv = (a @ b) / denom
        cosv = torch.clamp(cosv, -1.0, 1.0)
        return torch.arccos(cosv).item()

    def _dct_low(self, vec: torch.Tensor) -> torch.Tensor:
        """1-D DCT-II (orthonormal); keep low-frequency prefix."""
        try:
            # Preferred: PyTorch >= 2.1
            dct_vals = torch.fft.dct(vec, type=2, norm='ortho')
        except AttributeError:
            # Fallback: SciPy
            import numpy as np
            from scipy.fftpack import dct
            vec_np = vec.detach().cpu().numpy()
            dct_np = dct(vec_np, type=2, norm='ortho')
            dct_vals = torch.from_numpy(dct_np).to(vec.device)
        k = max(1, int(dct_vals.numel() * self.dct_keep_ratio))
        return dct_vals[:k]

    def _defense_seed_with_current_backbone(self):
        """Capture the initial backbone as B[0] so that the first Δ is well-defined later."""
        B_vec = self._flatten_model()
        state = self._export_state()
        self._B_vecs.clear(); self._B_states.clear()
        self._S_list.clear(); self._theta.clear()
        self._B_vecs.append(B_vec)
        self._B_states.append(state)

    def _defense_push_current(self):
        """
        After each client update is applied, push the new backbone into the defense window,
        build S (DCT_low of delta) and theta (angle) series, trimming to window size.
        """
        B_curr = self._flatten_model()
        state = self._export_state()

        if len(self._B_vecs) > 0:
            delta = B_curr - self._B_vecs[-1]
            S_t = self._dct_low(delta)
            theta_t = self._cosine_angle(B_curr, self._B_vecs[-1])

            self._S_list.append(S_t)
            self._theta.append(theta_t)

        self._B_vecs.append(B_curr)
        self._B_states.append(state)

        # Trim if exceeded window (deque already has maxlen, but S/theta need manual sync if window shrinks)
        # Note: deques with maxlen auto-discard from the left; S/theta lengths are maxlen = window-1
        # so nothing else is required here.

    def _static_scores(self) -> Optional[List[float]]:
        """
        Static (frequency) score E_i over S_list for i = 0..n-1 (where S[i] corresponds to B[i+1]).
        E_i = sum of distances to the closest n/2 + 1 neighbors in S-space (Euclidean over DCT_low).
        """
        n = len(self._S_list)
        if n < 2:
            return None
        scores: List[float] = []
        m = (n // 2) + 1
        S_all = list(self._S_list)
        for i in range(n):
            Si = S_all[i]
            dists = []
            for j in range(n):
                if i == j:
                    continue
                Sj = S_all[j]
                # guard if sizes ever differ (e.g., if keep_ratio changed mid-run)
                K = min(Si.numel(), Sj.numel())
                d = torch.norm(Si[:K] - Sj[:K]).item()
                dists.append(d)
            scores.append(sum(sorted(dists)[:m]))
        return scores

    def _rotational_scores(self) -> Optional[List[float]]:
        """
        Rotational score R_i across θ-series for i = 0..n-1 (θ[i] corresponds to angle from B[i] to B[i+1]).
        R_i = sum of the smallest n/2 + 1 absolute differences |θ_i - θ_j|.
        (Sum vs avg does not affect ranking; matches static form.)
        """
        n = len(self._theta)
        if n < 2:
            return None
        scores: List[float] = []
        m = (n // 2) + 1
        theta_all = list(self._theta)
        for i in range(n):
            diffs = [abs(theta_all[i] - theta_all[j]) for j in range(n) if j != i]
            scores.append(sum(sorted(diffs)[:m]))
        return scores

    def _defense_decide_and_get_rollback_index(self) -> Optional[int]:
        """
        Decide if current update is benign. If not, return backbone index in [0..len(B)-1] to roll back to.
        Mapping: S[i] corresponds to backbone B[i+1].
        Current update index in S-space is n-1 => current backbone index is len(B)-1.
        """
        # Need at least window_size backbones to run full decision reliably
        if len(self._B_vecs) < self.defense_window:
            return None

        E = self._static_scores()
        R = self._rotational_scores()
        if E is None or R is None:
            return None

        n = len(E)  # also len(R), equals len(B_vecs) - 1
        assert n == len(R)
        m = (n // 2) + 1

        # Rank by each metric (smaller is more-benign)
        freq_majority = set(sorted(range(n), key=lambda i: E[i])[:m])
        rot_majority  = set(sorted(range(n), key=lambda i: R[i])[:m])

        benign_updates = sorted(freq_majority & rot_majority)
        if not benign_updates:
            # Extremely adversarial window; conservative fallback to previous backbone
            return len(self._B_vecs) - 2  # previous B

        current_update_idx = n - 1              # last in S
        current_backbone_idx = len(self._B_vecs) - 1  # last in B
        if current_update_idx in benign_updates:
            # Current B_t is considered benign by both views
            return None

        # Otherwise, rollback to the most recent benign update's resulting backbone
        rollback_update_idx = max(benign_updates)       # index in S
        rollback_backbone_idx = rollback_update_idx + 1 # corresponding index in B
        rollback_backbone_idx = max(0, min(rollback_backbone_idx, current_backbone_idx))
        return rollback_backbone_idx
