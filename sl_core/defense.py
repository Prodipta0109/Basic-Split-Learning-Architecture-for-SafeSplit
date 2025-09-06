# sl_core/defense.py
import torch
from typing import List, Optional

class SafeSplitDefense:
    """
    Implements SafeSplit's two-view detection and rollback:
      - Static (frequency) analysis via low-frequency DCT on backbone updates
      - Dynamic (rotational) analysis via angular-displacement trajectories
    Maintains a sliding window (FIFO) of backbones and their derived signatures.
    """

    def __init__(self, window_size: int = 7, keep_ratio: float = 0.10, device: Optional[torch.device] = None):
        """
        Args:
            window_size: number of most-recent backbone checkpoints to keep (N in the paper).
                         Must be >= 3 (so we have at least two Δ's for θ/ω).
            keep_ratio:  fraction of low-frequency DCT coefficients to keep (0 < keep_ratio <= 1).
            device:      torch.device for temporary computations; if None, inferred from tensors.
        """
        assert window_size >= 3, "window_size must be >= 3"
        assert 0 < keep_ratio <= 1.0, "keep_ratio must be in (0, 1]"
        self.window_size = window_size
        self.keep_ratio = keep_ratio
        self.device = device

        # FIFO buffers
        self._B_vecs: List[torch.Tensor] = []         # flattened backbone vectors: [B_{t-N+1}, ..., B_t]
        self._B_states: List[dict] = []               # full state_dict for rollback: same order as _B_vecs

        # Derived series over the same window
        self._S_list: List[torch.Tensor] = []         # S_t = DCT_low(B_t - B_{t-1}) ; aligned to B index [1..len-1]
        self._theta:  List[float] = []                # θ(t)    (rad), aligned to B index [1..len-1]

    # ---------- Utilities ----------

    @staticmethod
    def _flatten_backbone_state(model) -> torch.Tensor:
        # Flatten all params into a single 1-D tensor (no grads)
        with torch.no_grad():
            parts = [p.detach().reshape(-1) for p in model.parameters()]
        return torch.cat(parts, dim=0)

    @staticmethod
    def _cosine_angle(a: torch.Tensor, b: torch.Tensor) -> float:
        # angle between vectors in radians, numerically stable
        denom = (a.norm(p=2) * b.norm(p=2)).clamp_min(1e-12)
        cosv = (a @ b) / denom
        cosv = torch.clamp(cosv, -1.0, 1.0)
        return torch.arccos(cosv).item()

    def _dct_low(self, vec: torch.Tensor) -> torch.Tensor:
        # 1-D DCT-II with orthonormal basis; keep low-frequency prefix
        dct_vals = torch.fft.dct(vec, type=2, norm='ortho')
        k = max(1, int(dct_vals.numel() * self.keep_ratio))
        return dct_vals[:k]

    # ---------- Public API ----------

    def push_backbone(self, backbone_module, state_dict: dict):
        """
        Add the latest server backbone snapshot after a client finishes training.
        Args:
            backbone_module: nn.Module (server backbone) – used to infer device and flatten weights
            state_dict:       deep-copied state_dict() of the backbone to support rollback
        """
        B_vec = self._flatten_backbone_state(backbone_module)
        if self.device is None:
            self.device = B_vec.device

        # Append to FIFOs
        if self._B_vecs:
            delta = (B_vec - self._B_vecs[-1])
            S_t = self._dct_low(delta)
            theta_t = self._cosine_angle(B_vec, self._B_vecs[-1])

            self._S_list.append(S_t)
            self._theta.append(theta_t)

        self._B_vecs.append(B_vec)
        self._B_states.append(state_dict)

        # Enforce window size across all buffers
        self._trim_to_window()

    def _trim_to_window(self):
        # Keep only the last window_size checkpoints
        extra = len(self._B_vecs) - self.window_size
        if extra > 0:
            self._B_vecs = self._B_vecs[extra:]
            self._B_states = self._B_states[extra:]
            # S_list and theta have length len(B)-1
            self._S_list   = self._S_list[extra:] if len(self._S_list) >= extra else []
            self._theta    = self._theta[extra:]  if len(self._theta)    >= extra else []

    def _static_scores(self) -> Optional[List[float]]:
        """
        Static (frequency) score E_i over S_list for i = 0..n-1 (where S[i] corresponds to B[i+1]).
        E_i = sum of distances to the closest n/2 + 1 neighbors in S-space (Euclidean over DCT_low).
        """
        n = len(self._S_list)
        if n < 2:
            return None

        # Precompute pairwise distances
        scores = []
        m = (n // 2) + 1  # majority count over S indices
        for i in range(n):
            Si = self._S_list[i]
            dists = []
            for j in range(n):
                if i == j: 
                    continue
                # pad to same length just in case (numerical guard if keep_ratio changes)
                if self._S_list[j].numel() != Si.numel():
                    K = min(self._S_list[j].numel(), Si.numel())
                    d = torch.norm(Si[:K] - self._S_list[j][:K]).item()
                else:
                    d = torch.norm(Si - self._S_list[j]).item()
                dists.append(d)
            score = sum(sorted(dists)[:m])
            scores.append(score)
        return scores  # length n, aligns to S indices = updates (B index +1)

    def _rotational_scores(self) -> Optional[List[float]]:
        """
        Rotational score R_i across θ-series for i = 0..n-1 (θ[i] corresponds to angle from B[i] to B[i+1]).
        R_i = avg of the smallest n/2 + 1 absolute differences |θ_i - θ_j|.
        """
        n = len(self._theta)
        if n < 2:
            return None

        scores = []
        m = (n // 2) + 1
        for i in range(n):
            diffs = [abs(self._theta[i] - self._theta[j]) for j in range(n) if j != i]
            score = sum(sorted(diffs)[:m])
            scores.append(score)
        return scores

    def decide_and_rollback_target(self) -> Optional[int]:
        """
        Returns:
            None if current update is benign (no rollback).
            Otherwise returns the backbone index within the FIFO [0..len(B)-1] to roll back to.
            Note: S[i] corresponds to B[i+1]. The "current update" is S[-1] -> target B index = len(B)-1.
        """
        E = self._static_scores()
        R = self._rotational_scores()
        if E is None or R is None:
            return None  # not enough history yet

        assert len(E) == len(R), "E and R length mismatch (should match S/θ length)"

        n = len(E)
        m = (n // 2) + 1

        # Rank by each metric (smaller is better/benign)
        freq_majority = set(sorted(range(n), key=lambda i: E[i])[:m])
        rot_majority  = set(sorted(range(n), key=lambda i: R[i])[:m])

        benign_updates = sorted(freq_majority & rot_majority)
        if not benign_updates:
            # Extremely adversarial window; be conservative: keep previous benign if possible
            return len(self._B_vecs) - 2  # fallback to previous backbone

        current_update_idx = n - 1          # index in S (last update)
        current_backbone_idx = len(self._B_vecs) - 1  # index in B (last backbone)

        if current_update_idx in benign_updates:
            # current B_t is benign by both views
            return None

        # Otherwise rollback to the most recent benign update's resulting backbone:
        # S[k] corresponds to backbone B[k+1]
        rollback_update_idx = max(benign_updates)
        rollback_backbone_idx = rollback_update_idx + 1
        # Guard bounds
        rollback_backbone_idx = max(0, min(rollback_backbone_idx, len(self._B_vecs) - 1))
        return rollback_backbone_idx

    # ---------- Accessors ----------

    def get_backbone_state_for_index(self, idx: int) -> dict:
        """Return the stored state_dict for B[idx] within the FIFO window."""
        return self._B_states[idx]

    def window_size_current(self) -> int:
        return len(self._B_vecs)

    def debug_dump_scores(self):
        E = self._static_scores()
        R = self._rotational_scores()
        if E is None or R is None:
            return None
        return {"E_static": E, "R_rot": R}
