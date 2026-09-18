"""
A small multi-layer perceptron built on numpy alone.

Skipping PyTorch or JAX here is intentional. Doing the forward and backward
passes by hand keeps every byte of memory and every operation visible, which
matters when the whole point is to account for memory precisely. It also
means there's no GPU dependency, so this runs the same on a laptop, in CI,
or on Colab.

The network itself is nothing special:

    x -> Linear -> ReLU -> Linear -> ReLU -> Linear -> ReLU -> Linear -> y

The one design decision worth explaining is that all parameters live in a
single flat float32 vector (`flat_params`) rather than as separate per-layer
arrays. Real ZeRO implementations do the same, because a flat buffer can be
cut into equal shards across ranks with a plain slice. That's what turns the
sharding logic in `cluster.py` into ordinary array indexing.
"""

from __future__ import annotations

import numpy as np
from dataclasses import dataclass, field


@dataclass
class LayerShape:
    in_dim: int
    out_dim: int

    @property
    def n_params(self) -> int:
        # weight matrix + bias vector
        return self.in_dim * self.out_dim + self.out_dim


@dataclass
class TinyMLP:
    """A small MLP whose parameters are stored as one flat float32 vector."""

    layer_shapes: list = field(default_factory=list)
    seed: int = 0

    def __post_init__(self):
        if not self.layer_shapes:
            # ~0.66M parameters by default. Small enough that this whole
            # notebook (model + 32 simulated GPUs x 4 ZeRO stages, each
            # holding its own numpy arrays) comfortably fits in a few
            # hundred MB of RAM and runs in well under a second on a CPU,
            # while still being big enough that the memory differences
            # between ZeRO stages are clearly visible. Bump these numbers
            # up (e.g. 2048-wide layers -> ~10M params) if you're running
            # on a machine/Colab instance with more RAM and want a more
            # dramatic (and slower) demo.
            self.layer_shapes = [
                LayerShape(128, 512),
                LayerShape(512, 512),
                LayerShape(512, 512),
                LayerShape(512, 128),
                LayerShape(128, 10),
            ]
        rng = np.random.default_rng(self.seed)
        self._offsets = []
        offset = 0
        for ls in self.layer_shapes:
            self._offsets.append(offset)
            offset += ls.n_params
        self.n_params = offset

        # Kaiming-ish init, flattened into one buffer.
        chunks = []
        for ls in self.layer_shapes:
            fan_in = ls.in_dim
            w = rng.standard_normal((ls.in_dim, ls.out_dim)).astype(np.float32)
            w *= np.sqrt(2.0 / fan_in)
            b = np.zeros(ls.out_dim, dtype=np.float32)
            chunks.append(w.reshape(-1))
            chunks.append(b)
        self.flat_params = np.concatenate(chunks).astype(np.float32)

    # ------------------------------------------------------------------
    # Flat-buffer <-> per-layer view helpers
    # ------------------------------------------------------------------
    def layer_views(self, flat: np.ndarray):
        """Yield (W, b) views into `flat` for each layer, without copying."""
        offset = 0
        for ls in self.layer_shapes:
            w_size = ls.in_dim * ls.out_dim
            w = flat[offset: offset + w_size].reshape(ls.in_dim, ls.out_dim)
            offset += w_size
            b = flat[offset: offset + ls.out_dim]
            offset += ls.out_dim
            yield w, b

    # ------------------------------------------------------------------
    # Forward / backward (manual autodiff, batch-first)
    # ------------------------------------------------------------------
    def forward(self, x: np.ndarray, flat_params: np.ndarray | None = None):
        flat = self.flat_params if flat_params is None else flat_params
        cache = {"a0": x}
        a = x
        layers = list(self.layer_views(flat))
        for i, (w, b) in enumerate(layers):
            z = a @ w + b
            is_last = i == len(layers) - 1
            a = z if is_last else np.maximum(z, 0.0)  # ReLU except last layer
            cache[f"z{i+1}"] = z
            cache[f"a{i+1}"] = a
        cache["n_layers"] = len(layers)
        return a, cache

    def backward(self, cache, y_true, flat_params: np.ndarray | None = None):
        """MSE loss. Returns (loss, flat_grad) with flat_grad same shape as
        flat_params."""
        flat = self.flat_params if flat_params is None else flat_params
        n_layers = cache["n_layers"]
        y_pred = cache[f"a{n_layers}"]
        batch = y_true.shape[0]

        loss = np.mean((y_pred - y_true) ** 2)
        dA = 2.0 * (y_pred - y_true) / y_true.size  # d(MSE)/d(y_pred)

        grad_flat = np.zeros_like(flat)
        layers = list(self.layer_views(flat))
        grad_views = list(self.layer_views(grad_flat))

        for i in reversed(range(n_layers)):
            w, b = layers[i]
            gw, gb = grad_views[i]
            z = cache[f"z{i+1}"]
            a_prev = cache[f"a{i}"]
            is_last = i == n_layers - 1
            dZ = dA if is_last else dA * (z > 0)
            gw[:] = a_prev.T @ dZ
            gb[:] = dZ.sum(axis=0)
            if i > 0:
                dA = dZ @ w.T
        return loss, grad_flat

    def bytes_for(self, n_params: int, dtype_bytes: int = 4) -> int:
        return n_params * dtype_bytes
