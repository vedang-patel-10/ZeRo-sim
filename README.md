# ZeRO Parallelism Simulator

A from-scratch simulation of ZeRO (Zero Redundancy Optimizer) running across 32 simulated GPUs, written in plain numpy with no deep learning framework. It trains a small MLP under four different memory schemes (ordinary data parallelism, then ZeRO stages 1, 2 and 3) and measures how per-GPU memory and per-step network traffic change between them.

Notebook: [`notebooks/zero_parallelism_simulator.ipynb`]

---

## The problem ZeRO solves

Take ordinary data-parallel training with Adam, in fp32, across `N` GPUs. Each GPU gets a different slice of the batch, computes gradients on it, and then everyone averages their gradients so the whole cluster stays in sync. Standard stuff.

The catch is what each GPU has to keep in memory to do that:

| Buffer | bytes / parameter | Why it's there |
|---|---|---|
| Parameters (`P`) | 4 | needed to run the forward pass |
| Gradients (`G`) | 4 | produced by the backward pass |
| Adam momentum (`m`) | 4 | first moment estimate, one per parameter |
| Adam variance (`v`) | 4 | second moment estimate, one per parameter |
| **Total per GPU** | **16** | |

So 16 bytes per parameter, on every GPU, and critically that number does not budge when you add hardware. Bring a 33rd GPU online and every GPU, the new one included, still needs the full 16Ψ bytes. Scaling out gets you more throughput but not one byte of memory relief. For a model that already barely fits, more GPUs simply doesn't help.

Most of that memory is redundant. All 32 GPUs are storing an identical copy of Adam's momentum and variance for every parameter, and all 32 are independently doing the same arithmetic with it to arrive at the same answer. That duplicated optimizer state is the biggest single target, and it's what ZeRO goes after first.

The reasoning behind ZeRO is fairly simple once you see it. Data parallelism already guarantees every GPU ends the step holding identical parameters. So you don't need all of them to independently compute and store that update. One GPU can own a slice of the model, compute the update for just that slice, and then tell everyone else the answer. Communication replaces duplication.

That idea gets applied in three stages, each one sharding a bit more.

### ZeRO-1: shard the optimizer state

Each GPU keeps Adam's `m` and `v` for only `1/N` of the parameters. Full parameters and full gradients still live on every GPU, since those fall out of the forward and backward passes anyway, but the optimizer state (a full 8 bytes per parameter for Adam) gets split up.

```
memory/GPU ≈ 4 (params) + 4 (grads) + 8/N (optimizer)
```

At `N=32` that's `4 + 4 + 0.25 = 8.25` bytes per parameter, roughly half of baseline, and it costs nothing in extra communication.

### ZeRO-2: shard the gradients too

Once a GPU isn't responsible for updating a given parameter, there's no reason for it to hold onto that parameter's gradient either. ZeRO-2 drops each gradient as soon as it's been reduced into whichever rank owns it. Real implementations do this during the backward pass, bucket by bucket, so a complete gradient buffer never needs to exist anywhere at once.

```
memory/GPU ≈ 4 (params) + 12/N (grads + optimizer)
```

At `N=32`: `4.375` bytes per parameter, about 73% below baseline.

### ZeRO-3: shard the parameters as well

The aggressive one. Even the weights are split, so no GPU holds a complete copy of the model at rest. When a layer needs its full weights to compute, they get all-gathered from every rank right before use and thrown away immediately after. The full model exists only for the instant it's needed.

```
memory/GPU ≈ 16/N   (steady state, excluding the transient gather buffer)
```

At `N=32`: `0.5` bytes per parameter, around 97% below baseline. This is the stage that makes "the model doesn't fit on one GPU" stop being a hard wall, because no single GPU ever has to hold the whole thing.

---

## The part people get wrong

ZeRO doesn't reduce how much math each GPU does, and it isn't a form of model or tensor parallelism.

Under every stage, including ZeRO-3, each GPU runs a complete forward and backward pass over its own batch slice, doing exactly the same matrix multiplies it would under plain data parallelism. Nothing about the computation changes. The notebook demonstrates this directly: the loss comes out at 3.6274 under all four schemes. If the sharding logic were wrong somewhere, that number would drift.

What you actually trade away is network bandwidth, and only at stage 3.

| | Baseline DDP | ZeRO-1 | ZeRO-2 | ZeRO-3 |
|---|---|---|---|---|
| Memory / GPU (bytes/param, N=32) | 16 | 8.25 | 4.375 | 0.5 |
| Communication / step | `2Ψ` | `2Ψ` | `2Ψ` | `3Ψ` (≈1.5×) |
| FLOPs / GPU | same | same | same | same |

Stages 1 and 2 are close to free. A ring all-reduce is already built internally out of a reduce-scatter phase followed by an all-gather phase, `2Ψ` bytes in total. ZeRO-1 and ZeRO-2 use that identical pattern: reduce-scatter so each rank receives the gradient slice it owns, run the optimizer step locally on just that slice, then all-gather the updated weights back out. Same bytes on the wire, just arranged differently.

Stage 3 is where you start paying. With no full copy of the weights sitting anywhere, they have to be all-gathered before the forward pass and gathered again before the backward pass, on top of the usual gradient reduce-scatter. That works out to `3Ψ` bytes, about 1.5× the baseline. It's why production ZeRO-3 systems like DeepSpeed put so much engineering into prefetching, kicking off the next layer's gather while the current layer is still computing, so the extra traffic overlaps with work instead of stalling on it.

---

## What's actually being simulated

There aren't 32 real accelerators here. A `VirtualGPU` in [`src/zero_sim/cluster.py`](src/zero_sim/cluster.py) is just a Python object with one rule enforced on it: it may only read or write its own shard, which is the same restriction a real GPU operates under. The 32 ranks run sequentially in a single process, and the collectives (all-gather, reduce-scatter) are carried out with real numpy slicing and concatenation, with the resulting array sizes measured via `.nbytes` and reported as memory and communication figures.

That makes this an accurate model of ZeRO's memory layout and communication pattern, which is what the algorithm actually changes. It is not a model of the wall-clock speedup you'd get from 32 GPUs computing in parallel over a real interconnect, and the notebook is explicit about that rather than presenting misleading timing numbers.

The demo model is deliberately small, around 658K parameters. With 32 shards each holding full or partial copies, memory usage scales at roughly 32× the model size, so anything sized for realistic large-model training would immediately exhaust a laptop. At this size the whole notebook (four stages, 32 simulated GPUs each) runs in well under a second per stage while the differences between stages stay obvious. `src/zero_sim/model.py` notes how to scale it up if you have the RAM.

### Known simplifications

**Mixed precision.** Real large-scale training keeps fp16 or bf16 copies of weights and gradients for compute alongside an fp32 master copy for the optimizer step. The `K=12` optimizer constant quoted in the ZeRO paper comes from that arrangement. This simulator runs fp32 throughout for clarity, so its byte counts are the fp32-only equivalent of the same idea.

**Bucketing and overlap.** Production implementations move gradients and weights in layer-sized or fixed-size buckets that overlap with ongoing compute. This code moves one flat buffer at a time, which is easier to follow but not how you'd build it for real.

**Offload.** ZeRO-Infinity pushes sharded state out to CPU RAM or NVMe when even sharded GPU memory runs short. Not covered here.

**Bandwidth and topology.** Communication cost is reported as a byte count, never as a time, because turning bytes into seconds depends on hardware specifics (NVLink versus Ethernet, ring versus tree) that this project isn't attempting to model.

---

## Repo layout

```
zero-parallelism-sim/
├── src/zero_sim/
│   ├── model.py
│   ├── optimizer.py
│   ├── cluster.py
│   └── __init__.py
├── notebooks/
│   └── zero_parallelism_simulator.ipynb
├── tests/
│   └── test_zero_sim.py
├── assets/
│   ├── memory_vs_world_size.png
│   └── comm_vs_stage.png
```

**`src/zero_sim/model.py`** — `TinyMLP`, a flat-buffer numpy MLP with hand-written forward and backward passes. All weights live in one flat float32 array so sharding is just an array slice.

**`src/zero_sim/optimizer.py`** — `AdamShard`, an Adam optimizer that runs on a shard of the flat buffer instead of the whole model. Holds the `m`/`v` state that ZeRO-1 splits up.

**`src/zero_sim/cluster.py`** — `VirtualGPU` and `ZeROCluster`, the simulator itself. Decides what each GPU owns under a given stage and runs the collectives. Start here if you want to read the code.

**`notebooks/zero_parallelism_simulator.ipynb`** — the walkthrough. Builds the 32-GPU cluster, runs all four stages, plots memory and communication. Outputs are already saved, so it reads fine without re-running it.

**`tests/test_zero_sim.py`** — five checks: loss stays identical across stages, memory shrinks monotonically, measured bytes match the closed-form formulas, communication ratios hold, and `N=1` collapses all stages to the same thing.

**`assets/`** — the two charts the notebook generates, saved as PNGs so they render on GitHub without needing to open the notebook.

---

## Running it

```bash
git clone <repo-url>
cd zero-parallelism-sim
pip install numpy pandas matplotlib jupyter
jupyter notebook notebooks/zero_parallelism_simulator.ipynb
```

On Colab, clone the repo and run `!pip install -q numpy pandas matplotlib`, then point `sys.path` at the cloned `src/` directory. The first code cell in the notebook has that line ready to uncomment.

To run the tests:

```bash
pytest tests/ -v
```

The most useful one is `test_memory_matches_closed_form`, which asserts that the bytes the simulator actually measures line up with the `16 / 8.25 / 4.375 / 0.5` figures the theory predicts. That's the check separating a real simulation from one that just prints believable-looking numbers.

---

## Where to start reading

If you want to follow the code, `cluster.py` is the interesting file. `ZeROCluster._setup()` decides what each of the 32 GPUs gets to hold under a given stage, and `training_step()` runs the forward and backward pass plus whichever collectives that stage requires. Everything else supports those two methods.

If you just want the results, sections 2 through 5 of the notebook have the numbers and both charts.
