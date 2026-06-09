# Why More Compute Does Not Mean Faster AI — The Roofline Model Explained

**By Mona Mishra · Feb 24, 2026**

![GitHub Stars](https://img.shields.io/github/stars/monamishra95/roofline-model-llm-inference?style=social)
![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)

> *Part 1 of 2 — This post builds the framework. [Part 2](./PART2_PREVIEW.md) outlines the levers teams can pull.*

---

## Table of Contents

1. [TL;DR](#tldr)
2. [The Problem](#the-problem)
3. [Section 1 — What Is a FLOP and Why Does It Matter?](#section-1--what-is-a-flop-and-why-does-it-matter)
4. [Section 2 — Inside the Machine: The Anatomy of a TPU](#section-2--inside-the-machine-the-anatomy-of-a-tpu)
5. [Section 3 — Arithmetic Intensity: The Number That Determines Everything](#section-3--arithmetic-intensity-the-number-that-determines-everything)
6. [Section 4 — The Roofline Model](#section-4--the-roofline-model)
7. [Section 5 — Where LLM Workloads Actually Land](#section-5--where-llm-workloads-actually-land)
8. [Conclusion](#conclusion)
9. [Try It Yourself](#try-it-yourself)

---

## TL;DR

- **FLOPS** (Floating Point Operations Per Second) measures compute speed. Modern AI chips do hundreds of trillions per second.
- A **TPU** has three core components: the **MXU** (math engine), **HBM** (fast memory), and **ICI** (chip-to-chip network). The speed gap between these parts is where compute spend is underutilized.
- The **Roofline Model** is a chart that tells you whether your workload is bottlenecked by math speed or memory bandwidth. It answers: *"What is the physical limit of this chip for this specific algorithm?"*
- **Almost all LLM inference is Memory Bound by default** — meaning organizations are paying for compute horsepower they structurally cannot use.
- Knowing which zone a workload sits in is the prerequisite for every hardware and architecture decision that follows.

---

## The Problem

It's a hot, humid afternoon in Council Bluffs, Iowa, in Google's large data center. Understandably, the sultry weather means the AI chips doing reasoning are probably idle right now.

On most LLM inference deployments, modern GPUs and TPUs spend the majority of their cycles **waiting** — stalled, starved of data, not performing calculations. A team could buy twice the compute, and their model wouldn't produce a single token faster. The bottleneck isn't the chip's brain. **It's the path from the chip's brain to its memory.**

This two-part series is about understanding exactly why that happens, and then systematically fixing it. This first post builds the framework. The second outlines the levers.

---

## Section 1 — What Is a FLOP and Why Does It Matter?

### The Atomic Unit of AI Computation

A **floating point operation (FLOP)** is one of the elementary arithmetic operations (+, -, ×, ÷) carried out on floating-point numbers. Evaluating the expression `(y - jx) / k` takes 3 FLOPs. A model's "intelligence" — weights, attention heads, matrix multiplications — is trillions of these FLOPs chained together.

### Hardware Throughput at Scale

| Scale | Name | Meaning |
|-------|------|---------|
| 10⁹ FLOPs/sec | **GFLOPS** | Billions/sec — a modern smartphone |
| 10¹² FLOPs/sec | **TFLOPS** | Trillions/sec — current AI accelerators |
| 10¹⁵ FLOPs/sec | **PFLOPS** | Quadrillions/sec — large GPU clusters |
| 10¹⁸ FLOPs/sec | **EFLOPS** | Quintillions/sec — Google's TPU v4 pod (1.1 exaflops) |

> **Key takeaway:** A TPU v4 pod of 4,096 chips can perform 1.1 quintillion math operations per second. And yet, if the wrong model architecture is deployed, it will still be idle. More speed is not always the right answer.

### Precision: Not All FLOPs Are Created Equal

"Precision" refers to how many bits are used to represent numerical values in computations (ie, weights, activations and gradients in training or inference). It impacts model accuracy, speed, efficiency, and energy consumption. Not all FLOPs cost the same, and neither are they equally accurate. 

| Format | Bits | Use Case |
|--------|------|----------|
| **FP64** | 64 | Scientific-grade workloads (physics simulation, HPC) |
| **FP32** | 32 | Traditional training; high precision, slower, good for general purpose AI tasks |
| **BF16 / FP16** | 16 | Current standard for LLM training and inference |
| **INT8** | 8 | Quantized inference; 2× memory efficiency vs. BF16 |
| **FP4 / INT4** | 4 | Aggressive quantization for edge AI where efficiency is critical; 4× efficiency vs. BF16 |

> **Key takeaway:** Lower precision = fewer bytes per parameter = more data fitting into memory bandwidth.

---

## Section 2 — Inside the Machine: The Anatomy of a TPU

The dominant operation in deep learning is  **matrix multiplication**. Google built the TPU to improve on the architectural efficiency of general-purpose CPUs and GPUs for **matrix multiplication**. Every design decision — memory architecture, interconnect, compute layout — reflects that singular focus.

A single TPU chip has three main components:

---

### Component 1: MXU (Matrix Multiply Unit) — The Brain

The MXU is implemented as a **systolic array** — a grid of simple multiply-accumulate (MAC) units that pass data from cell to cell in a wave-like motion, performing matrix multiplication in a highly parallel, pipelined manner.

A TPU v4 chip contains **8 MXUs** (4 per TensorCore, 2 TensorCores per chip), each a 128×128 systolic array.

| Situation | MXU Behavior |
|-----------|-------------|
| Large, dense matrices (training) | ✅ Excels — Transformer attention and feed-forward layers |
| Small, sequential operations (inference decode) | ❌ Suffers — generating one token at a time |

> **Key takeaway:** The systolic array was designed for large, dense, parallel matrix multiplications. Autoregressive token generation — sequential, one token at a time — is not handled efficiently.

---

### Component 2: HBM (High Bandwidth Memory) — The Fast Memory

Every weight, every activation, every KV cache entry must be loaded into HBM before the MXU can touch it.

| Chip | HBM Bandwidth | HBM Capacity |
|------|--------------|--------------|
| H100 SXM5 | 3.35 TB/s | 80 GB |
| TPU v4 | 1.2 TB/s | 32 GB |

**The capacity problem:** A 70B parameter model in BF16 weighs ~140 GB. The math: `70B × 2 bytes = 140 GB`. An H100 only has 80 GB — meaning this model needs at least 2 chips.

**The bandwidth problem:** 3.35 TB/s sounds enormous until you consider the H100's MXU can consume up to 989 TFLOPS of BF16. For every second of math, it demands far more data than the memory pipe can deliver. For the chip to run at 100% capacity: `989 TFLOPS ÷ 3.35 TB/s = ~295 FLOPs per byte loaded`.

> **Key takeaway:** HBM is not the bottleneck because it's slow. It's the bottleneck because compute got so fast it outran the pipe. The math engine is a race car; the memory bus is a well-maintained country road.

---

### Component 3: ICI (Inter Chip Interconnect) — The Chip-to-Chip Network

Since frontier models exceed single-chip capacity, they must be **sharded** across pods of thousands of chips. The ICI is the high-speed fabric connecting them. Sharding means splitting a model's massive parameters into smaller chunks.

- **Architecture:** TPU v4 uses a 3D torus topology, directly linking each chip to its six nearest neighbors
- **Throughput:** 1.1 PB/s all-reduce bandwidth per pod — bypassing the high-latency PCIe CPU-host bottlenecks typical of GPU setups
- **Impact:** For sharded models, ICI bandwidth directly gates synchronization speed for both training (gradient updates) and inference (KV cache reads)

> **Key takeaway:** Think of ICI as the highway system between factories. Each factory (chip) may run optimally on its own, but if the highway can't deliver raw materials fast enough, the whole distributed system stalls.

---

### How the Three Components Interact

| Component | Speed on TPU v4 | Role |
|-----------|----------------|------|
| MXU (Compute) | 275 TFLOPS | Does the math |
| HBM (Memory) | 1.2 TB/s | Feeds the MXU |
| ICI (Interconnect) | ~900 GB/s per link | Synchronizes chips |

> **Key takeaway:** The MXU can process data roughly **229× faster** than HBM can deliver it (`275 TFLOPS ÷ 1.2 TB/s ≈ 229 FLOPs/Byte`). This ratio — how much math gets done per byte moved — is the most important number in AI infrastructure. It's called **arithmetic intensity**.

---

## Section 3 — Arithmetic Intensity: The Number That Determines Everything

**Arithmetic Intensity (AI)** measures how many FLOPs a model performs per byte of data moved from HBM to the MXU:

```
Arithmetic Intensity = Total FLOPs / Total Bytes Moved from HBM
```

- **High arithmetic intensity** = a lot of math per byte loaded → efficient
- **Low arithmetic intensity** = barely using the data being moved → wasteful
- This ratio is a property of the **workload** (model architecture, batch size, sequence length), not the hardware

### The Chef Analogy

Think of the MXU as a head chef and HBM as the walk-in fridge:

| Scenario | What Happens | Arithmetic Intensity |
|----------|-------------|---------------------|
| **Low** | Chef walks to fridge, retrieves one ingredient, uses it, walks back. Repeat 10,000 times. 80% of the day is commuting. | Low — master chef's salary wasted on pacing |
| **High** | Chef loads a full cart, spends two hours chopping, sautéing, and plating without moving. Every minute is productive. | High — this is what that salary is meant to deliver |

> **Key takeaway:** Arithmetic Intensity is the ROI on data movement. It is the single metric that determines whether a hardware investment is productive or stranded.

---

## Section 4 — The Roofline Model

The Roofline Model gives a visual framework to understand the **theoretical maximum performance** of any hardware-workload combination.

### The Equation

```
P = min(π, β × I)
```

Where:
- `P` = Attainable Performance (GFLOPS/sec)
- `π` = Peak Compute (chip's max TFLOPS)
- `β` = Peak Memory Bandwidth (GB/s)
- `I` = Arithmetic Intensity (FLOPs/Byte)

The `min()` function means performance is capped by whichever constraint bites first — the compute ceiling or the memory pipe.

### The Two Zones

```
FLOPs/s
  │                              ╔══════════════════════════ Compute Bound (π)
  │                          ╔══╝         (Zone 2)
  │                      ╔══╝
  │                  ╔══╝   ← Ridge Point = π/β
  │              ╔══╝
  │          ╔══╝
  │      ╔══╝   Memory Bound (β × I)
  │  ╔══╝        (Zone 1)
  │══╝
  └──────────────────────────────────────────── Arithmetic Intensity (FLOPs/Byte)
```

| Zone | Condition | What It Means | Operational Implication |
|------|-----------|---------------|------------------------|
| **Memory Bound** | I < π/β | MXU is idle, waiting for data. Performance ∝ bandwidth × intensity. | Buying a faster chip is money wasted. Fix the bandwidth pipe. |
| **Compute Bound** | I ≥ π/β | Memory pipe is saturated. MXU at full utilization. Peak efficiency. | Every dollar of chip value is captured. To go faster: get a faster chip. |

### The Ridge Point

The inflection point between zones: `I_min = π / β`

| Chip | Compute | Bandwidth | Ridge Point |
|------|---------|-----------|-------------|
| H100 SXM5 | 989 TFLOPS (BF16) | 3,350 GB/s | **~295 FLOPs/Byte** |
| TPU v4 | 275 TFLOPS | 1,200 GB/s | **~229 FLOPs/Byte** |

Every workload with arithmetic intensity **below** these values is memory bound. The MXU is stalling.

> **Key takeaway:** The Ridge Point is a procurement threshold. If your deployed workloads don't reach it, you're paying for a compute ceiling you will never touch.

---

## Section 5 — Where LLM Workloads Actually Land

The uncomfortable reality: **LLM inference during the decode phase is almost always deeply memory bound.**

This is because autoregressive generation is a **matrix-vector** multiplication (not matrix-matrix). Each decode step loads the full model's weight matrix from HBM — ~140 GB for a 70B BF16 model — to compute *one token*.

| LLM Phase | Operation Type | Arithmetic Intensity | Zone on H100 |
|-----------|---------------|---------------------|-------------|
| **Prefill** (processing prompt) | Matrix-matrix multiply | ~100–500 FLOPs/Byte | Near or above ridge point |
| **Decode** (generating tokens) | Matrix-vector multiply | ~0.5–1 FLOPs/Byte | Deeply memory bound |

Against an H100 Ridge Point of 295, most decode deployments operate at roughly **0.2% of the chip's potential**.

The two stages are fundamentally different compute problems running on the same chip.

---

## Conclusion

The MXU, the HBM, the ICI — they are all frozen choices, locked in silicon, manufactured by the millions. Those choices cannot be undone at deployment time. What *can* be changed is the workload that gets handed to that silicon.

The Roofline Model is the map that shows exactly how far the current workload is from the chip's real ceiling — and how much performance is sitting uncaptured between where a model runs today and where physics says its potential lies.

The chip already knows how fast it can go. The question is whether the architecture is ready to keep up with it.

---

## Try It Yourself

**→ [Interactive Roofline Calculator](./calculator/index.html)** — plug in your chip's specs, see your ridge point and zone classification with a live chart.

**→ [Arithmetic Intensity Estimator](./scripts/arithmetic_intensity.py)** — run this Python script to estimate where your LLM workload sits on the Roofline.

```bash
python scripts/arithmetic_intensity.py --model 70b --precision bf16 --chip h100 --phase decode
```

**→ [Part 2 Preview](./PART2_PREVIEW.md)** — the levers: batching, quantization, speculative decoding, and more.

---

## About

Written by **Mona Mishra**. If this was useful, star the repo and share it with your team.

*Topics: `llm` `inference` `gpu` `tpu` `machine-learning` `ai-infrastructure` `hardware` `roofline-model` `compute-efficiency`*
