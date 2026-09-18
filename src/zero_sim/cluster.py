"""
The ZeRO simulator.

No real GPUs or OS processes are involved here. What is real is the numpy
work: flat buffers get sliced into equal shards, gradients get
reduce-scattered, parameters get all-gathered, and the byte sizes reported
come from `.nbytes` on the resulting arrays rather than from a formula.

What's being simulated is memory placement and communication volume, which
is what ZeRO actually changes. Running the 32 ranks sequentially in one
process is therefore a reasonable stand-in for 32 devices, as long as you
don't read the timings as a throughput benchmark. Each "GPU" is a Python
object restricted to its own shard, and keeping that restriction honest is
what makes the memory accounting mean anything.

Stages:

  BASELINE   Ordinary data parallelism. Every GPU holds a full copy of the
             parameters, gradients, and Adam state (m, v).

  ZeRO-1     (P_os)      Optimizer state partitioned across GPUs.
  ZeRO-2     (P_os+g)    Optimizer state and gradients partitioned.
  ZeRO-3     (P_os+g+p)  Optimizer state, gradients and parameters all
                         partitioned. Full parameters exist only briefly,
                         gathered on demand and dropped afterwards.
"""

from __future__ import annotations

import numpy as np
from dataclasses import dataclass, field
from typing import Literal

from .model import TinyMLP
from .optimizer import AdamShard

Stage = Literal["baseline", "zero1", "zero2", "zero3"]

BYTES_FP32 = 4


def shard_bounds(n_total: int, n_shards: int, rank: int):
    """Equal-ish contiguous partition of `n_total` elements into `n_shards`
    pieces (last shard absorbs the remainder). Real ZeRO implementations
    pad to make shards exactly equal; we keep it simple and just let the
    last shard be slightly larger, which does not change any conclusion."""
    base = n_total // n_shards
    start = rank * base
    end = n_total if rank == n_shards - 1 else start + base
    return start, end


@dataclass
class VirtualGPU:
    """One simulated device. Only ever touches its own shard(s)."""

    rank: int
    world_size: int

    # Populated by ZeROCluster.setup(), depending on stage.
    param_shard: np.ndarray = None          # ZeRO-3 only: this rank's slice of params
    full_params: np.ndarray = None          # baseline / ZeRO-1 / ZeRO-2: full copy
    grad_shard: np.ndarray = None           # ZeRO-2 / ZeRO-3: this rank's slice of grads
    full_grads: np.ndarray = None           # baseline / ZeRO-1: full copy
    adam: AdamShard = None                  # sized to whatever this rank owns

    def memory_bytes(self) -> dict:
        """Breakdown of this GPU's live memory, in bytes."""
        params = (self.full_params.nbytes if self.full_params is not None
                  else self.param_shard.nbytes)
        grads = (self.full_grads.nbytes if self.full_grads is not None
                 else self.grad_shard.nbytes)
        optim = self.adam.nbytes
        return {"params": params, "grads": grads, "optimizer_state": optim,
                "total": params + grads + optim}


@dataclass
class StepReport:
    stage: str
    world_size: int
    loss: float
    per_gpu_memory: list           # list[dict] len == world_size
    peak_transient_bytes: int      # extra memory needed momentarily (ZeRO-3 all-gather buffer)
    comm_bytes_per_gpu: int        # bytes sent+received by a single GPU this step
    comm_bytes_total: int          # summed across the whole cluster this step


class ZeROCluster:
    def __init__(self, model: TinyMLP, world_size: int = 32, stage: Stage = "baseline"):
        self.model = model
        self.world_size = world_size
        self.stage = stage
        self.gpus: list[VirtualGPU] = []
        self._setup()

    # ------------------------------------------------------------------
    def _setup(self):
        n = self.model.n_params
        flat_params = self.model.flat_params
        self.gpus = []
        for r in range(self.world_size):
            gpu = VirtualGPU(rank=r, world_size=self.world_size)
            s, e = shard_bounds(n, self.world_size, r)
            shard_len = e - s

            if self.stage == "baseline":
                gpu.full_params = flat_params.copy()
                gpu.full_grads = np.zeros(n, dtype=np.float32)
                gpu.adam = AdamShard(n=n)               # full optimizer state, replicated

            elif self.stage == "zero1":                  # P_os
                gpu.full_params = flat_params.copy()
                gpu.full_grads = np.zeros(n, dtype=np.float32)
                gpu.adam = AdamShard(n=shard_len)         # only this rank's slice of m, v

            elif self.stage == "zero2":                   # P_os+g
                gpu.full_params = flat_params.copy()
                gpu.grad_shard = np.zeros(shard_len, dtype=np.float32)
                gpu.adam = AdamShard(n=shard_len)

            elif self.stage == "zero3":                    # P_os+g+p
                gpu.param_shard = flat_params[s:e].copy()
                gpu.grad_shard = np.zeros(shard_len, dtype=np.float32)
                gpu.adam = AdamShard(n=shard_len)

            else:
                raise ValueError(self.stage)
            self.gpus.append(gpu)

    # ------------------------------------------------------------------
    def _all_gather_params(self) -> np.ndarray:
        """ZeRO-3 only: reconstruct the full parameter vector from shards.
        Returns the gathered array AND, as a side effect, we count the
        bytes moved by the caller."""
        return np.concatenate([g.param_shard for g in self.gpus])

    def _reduce_scatter_grad(self, full_grad: np.ndarray) -> list[np.ndarray]:
        """Split a full gradient vector into per-rank shards. In a real
        cluster this is a reduce-scatter collective: every rank ends up
        with the *averaged* gradient for its own shard only, having sent
        its full gradient and received back only its slice. Communication
        volume is ~ the full gradient size, same total bytes as an
        all-reduce, just redistributed differently."""
        shards = []
        for r in range(self.world_size):
            s, e = shard_bounds(len(full_grad), self.world_size, r)
            shards.append(full_grad[s:e].copy())
        return shards

    # ------------------------------------------------------------------
    def training_step(self, x: np.ndarray, y: np.ndarray) -> StepReport:
        n = self.model.n_params
        comm_bytes_per_gpu = 0
        peak_transient = 0

        if self.stage in ("baseline", "zero1", "zero2"):
            # Every GPU holds full parameters already -> forward/backward
            # can run locally with no parameter communication needed.
            flat = self.gpus[0].full_params
            y_pred, cache = self.model.forward(x, flat)
            loss, full_grad = self.model.backward(cache, y, flat)

            if self.stage == "baseline":
                # Gradients are all-reduced (averaged) across all GPUs so
                # every GPU ends this step with the identical full grad.
                # All-reduce cost ~= 2 * Psi bytes sent+received per GPU
                # (ring all-reduce lower bound).
                comm_bytes_per_gpu = 2 * full_grad.nbytes
                for gpu in self.gpus:
                    gpu.full_grads[:] = full_grad
                    gpu.adam.step(gpu.full_params, gpu.full_grads)

            elif self.stage == "zero1":
                # Gradients are communicated as a reduce-scatter (Psi bytes)
                # so that each rank's optimizer step uses only the slice it
                # owns the Adam state for; the freshly-updated parameter
                # slices are then all-gathered back out (another Psi bytes)
                # so every GPU again holds full, up-to-date parameters for
                # the next forward pass. Total = 2*Psi -- the *same* total
                # volume as one baseline all-reduce (an all-reduce is
                # itself implemented as reduce-scatter + all-gather under
                # the hood), even though the memory pattern differs: every
                # GPU still ends this step holding a full gradient buffer,
                # it just never needed to see anyone else's Adam state.
                shards = self._reduce_scatter_grad(full_grad)
                comm_bytes_per_gpu = full_grad.nbytes        # reduce-scatter grads
                for gpu, gshard in zip(self.gpus, shards):
                    gpu.full_grads[:] = full_grad             # (full grad still lands here from backward)
                    s, e = shard_bounds(n, self.world_size, gpu.rank)
                    gpu.adam.step(gpu.full_params[s:e], gshard)
                comm_bytes_per_gpu += full_grad.nbytes        # all-gather updated params

            elif self.stage == "zero2":
                # Same reduce-scatter + all-gather pattern as ZeRO-1
                # (total = 2*Psi, same as baseline) -- but now every GPU
                # only ever *stores* its own gradient shard (see the
                # memory report), instead of the full gradient buffer
                # ZeRO-1 keeps around.
                shards = self._reduce_scatter_grad(full_grad)
                comm_bytes_per_gpu = full_grad.nbytes        # reduce-scatter grads
                for gpu, gshard in zip(self.gpus, shards):
                    gpu.grad_shard[:] = gshard
                    s, e = shard_bounds(n, self.world_size, gpu.rank)
                    gpu.adam.step(gpu.full_params[s:e], gpu.grad_shard)
                comm_bytes_per_gpu += full_grad.nbytes        # all-gather updated params

        elif self.stage == "zero3":
            # Nothing has full parameters at rest. Before we can even run
            # the forward pass we must ALL-GATHER the full parameter
            # vector -- this is the extra communication ZeRO-3 pays for.
            gathered_params = self._all_gather_params()
            peak_transient = gathered_params.nbytes  # transient buffer, freed right after use
            comm_bytes_per_gpu = gathered_params.nbytes  # all-gather (forward)

            y_pred, cache = self.model.forward(x, gathered_params)
            loss, full_grad = self.model.backward(cache, y, gathered_params)
            # backward needs the full params a second time (for a real
            # transformer this is often re-gathered layer by layer);
            # we count that second all-gather explicitly:
            comm_bytes_per_gpu += gathered_params.nbytes  # all-gather (backward)

            shards = self._reduce_scatter_grad(full_grad)
            comm_bytes_per_gpu += full_grad.nbytes  # reduce-scatter of grads

            for gpu, gshard in zip(self.gpus, shards):
                gpu.grad_shard[:] = gshard
                gpu.adam.step(gpu.param_shard, gpu.grad_shard)
                # each GPU now discards everyone else's params again --
                # nothing further to send.
        else:
            raise ValueError(self.stage)

        per_gpu_mem = [g.memory_bytes() for g in self.gpus]
        return StepReport(
            stage=self.stage,
            world_size=self.world_size,
            loss=float(loss),
            per_gpu_memory=per_gpu_mem,
            peak_transient_bytes=peak_transient,
            comm_bytes_per_gpu=comm_bytes_per_gpu,
            comm_bytes_total=comm_bytes_per_gpu * self.world_size,
        )
