"""
A minimal Adam optimizer that works on flat float32 buffers.

Adam carries two extra float32 arrays the same size as the parameters, the
first-moment (m) and second-moment (v) running averages. Those two arrays
are the "optimizer state" that ZeRO-1 splits across GPUs. In mixed-precision
training with Adam or AdamW they're usually the largest single block of
memory on the device, which is why ZeRO targets them before anything else.
"""

from __future__ import annotations

import numpy as np
from dataclasses import dataclass


@dataclass
class AdamShard:
    """Adam state for a *shard* of parameters (n = shard size)."""

    n: int
    lr: float = 1e-3
    beta1: float = 0.9
    beta2: float = 0.999
    eps: float = 1e-8

    def __post_init__(self):
        self.m = np.zeros(self.n, dtype=np.float32)
        self.v = np.zeros(self.n, dtype=np.float32)
        self.t = 0

    @property
    def nbytes(self) -> int:
        return self.m.nbytes + self.v.nbytes  # 8 bytes / param (fp32 m + v)

    def step(self, param_shard: np.ndarray, grad_shard: np.ndarray) -> None:
        self.t += 1
        self.m = self.beta1 * self.m + (1 - self.beta1) * grad_shard
        self.v = self.beta2 * self.v + (1 - self.beta2) * (grad_shard ** 2)
        m_hat = self.m / (1 - self.beta1 ** self.t)
        v_hat = self.v / (1 - self.beta2 ** self.t)
        param_shard -= self.lr * m_hat / (np.sqrt(v_hat) + self.eps)
