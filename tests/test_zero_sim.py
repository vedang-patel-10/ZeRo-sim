import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import numpy as np
from zero_sim import TinyMLP, ZeROCluster


def _run(stage, world_size=32):
    model = TinyMLP(seed=0)
    x = np.random.default_rng(1).standard_normal((32, model.layer_shapes[0].in_dim)).astype(np.float32)
    y = np.random.default_rng(2).standard_normal((32, model.layer_shapes[-1].out_dim)).astype(np.float32)
    cluster = ZeROCluster(model, world_size=world_size, stage=stage)
    return cluster.training_step(x, y)


def test_losses_match_across_stages():
    losses = [_run(s).loss for s in ("baseline", "zero1", "zero2", "zero3")]
    for l in losses[1:]:
        assert abs(l - losses[0]) < 1e-4, "ZeRO must not change what the model computes"


def test_memory_shrinks_monotonically():
    mem = {s: _run(s).per_gpu_memory[0]["total"] for s in ("baseline", "zero1", "zero2", "zero3")}
    assert mem["baseline"] > mem["zero1"] > mem["zero2"] > mem["zero3"]


def test_memory_matches_closed_form():
    N = 32
    model = TinyMLP(seed=0)
    psi = model.n_params
    theory = {
        "baseline": 16,
        "zero1": 4 + 4 + 8 / N,
        "zero2": 4 + 12 / N,
        "zero3": 16 / N,
    }
    for stage, bytes_per_param in theory.items():
        measured = _run(stage, world_size=N).per_gpu_memory[0]["total"]
        expected = bytes_per_param * psi
        assert abs(measured - expected) / expected < 0.02


def test_comm_volume_matches_paper_claims():
    r_base = _run("baseline")
    r_z1 = _run("zero1")
    r_z2 = _run("zero2")
    r_z3 = _run("zero3")
    assert r_base.comm_bytes_per_gpu == r_z1.comm_bytes_per_gpu == r_z2.comm_bytes_per_gpu
    assert abs(r_z3.comm_bytes_per_gpu / r_base.comm_bytes_per_gpu - 1.5) < 1e-6


def test_world_size_one_all_stages_equal():
    mem = {s: _run(s, world_size=1).per_gpu_memory[0]["total"] for s in ("baseline", "zero1", "zero2", "zero3")}
    assert len(set(mem.values())) == 1, "Sharding across 1 device is a no-op"
