# The Levers: Six Ways to Fix a Memory-Bound LLM Workload

**By Mona Mishra · Part 2 of 2**

> *Part 1 established the problem: almost all LLM inference runs at a fraction of chip potential because decode is deeply memory bound. This post is about what to do about it.*

---

## Table of Contents

1. [The Setup](#the-setup)
2. [Lever 1 — Batching](#lever-1--batching)
3. [Lever 2 — Quantization](#lever-2--quantization)
4. [Lever 3 — Speculative Decoding](#lever-3--speculative-decoding)
5. [Lever 4 — KV Cache Reduction](#lever-4--kv-cache-reduction)
6. [Lever 5 — Disaggregated Serving](#lever-5--disaggregated-serving)
7. [Lever 6 — Hardware Selection](#lever-6--hardware-selection)
8. [Conclusion — The Co-Design Imperative](#conclusion--the-co-design-imperative)

---

## The Setup

Part 1 ended at a specific, uncomfortable number: a 70B BF16 model on an H100, serving one user at a time, runs at roughly **0.2% of the chip's theoretical peak compute**. The ridge point of an H100 is ~295 FLOPs/Byte; a single-request decode runs at ~0.5 FLOPs/Byte. The MXU is idle 99.8% of the time, waiting for weights to arrive from HBM.

This is not a product failure. It is a structural property of autoregressive generation. Each decode step loads the full weight matrix from HBM — ~140 GB for a 70B BF16 model — to compute exactly one token. The math engine can process data 295× faster than memory can deliver it.

The good news: every one of the six levers below directly attacks this gap. Some operate at the software level, some at the systems level, and one at the silicon level. Each has a different cost, complexity, and ceiling.

---

## Lever 1 — Batching

**What it does:** Increases arithmetic intensity by amortizing the weight-load cost across multiple tokens.

**The math:** Decode arithmetic intensity is `batch_size / bytes_per_param`. For BF16, that's `batch_size / 2`. A batch of 1 yields ~0.5 FLOPs/Byte. A batch of 240 yields ~120 FLOPs/Byte.

**The critical batch size:** There is a specific batch size at which a workload transitions from memory-bound to compute-bound. For a given chip, this is:

```
B_crit = Peak Compute (FLOPs/s) / Memory Bandwidth (Bytes/s)
       = Ridge Point (FLOPs/Byte) × bytes_per_param
```

For BF16 on a TPU v5e, B_crit ≈ **240 tokens**. For H100 BF16, B_crit ≈ **295 tokens**. At these batch sizes, every additional FLOP of compute capacity is utilized — the memory pipe is saturated and the MXU is running near full throughput.

> *Source: Austin et al., "How to Scale Your Model", Google DeepMind, 2025 — Chapter 7 derives B_crit = C / W_HBM for linear layers during generation.*

**Why it's hard in practice:** To reach B_crit = 240, you need 240 concurrent users all generating tokens at the same moment. In many deployment contexts — especially low-traffic or latency-sensitive applications — this is not achievable. And as batch size grows, so does the KV cache footprint (see Lever 4), eventually exhausting HBM before B_crit is reached.

**The production solution:** **Continuous batching** — rather than waiting for a full batch to form before starting, an inference engine dynamically inserts new requests into already-running generation slots as they open up. This keeps the effective batch size high without requiring 240 simultaneous arrivals. Google's [JetStream](https://github.com/google/JetStream) is an open-source implementation of this pattern.

**Quantization effect on B_crit:** If model weights are quantized to INT8 (1 byte/param) while activations remain BF16 (2 bytes/activation), the ratio of bytes-per-param to bytes-per-activation halves, and B_crit drops to ~120. This is a significant practical improvement — discussed further in Lever 2.

---

## Lever 2 — Quantization

**What it does:** Reduces bytes per parameter, cutting the HBM load per decode step and shifting the arithmetic intensity upward.

**The basic tradeoff:**

| Precision | Bytes/Param | Effect on B_crit (TPU v5e, bf16 activations) |
|-----------|-------------|-----------------------------------------------|
| BF16      | 2           | ~240 tokens                                   |
| INT8      | 1           | ~120 tokens                                   |
| INT4      | 0.5         | ~60 tokens                                    |

This table shows the weight-quantization case — activations remain BF16. When both weights and activations are quantized (e.g. INT8 × INT8), the chip's INT8 compute throughput also changes, so B_crit must be recalculated: `B_crit = β × α_HBM` where β = bits_per_param / bits_per_activation.

> *Source: Austin et al., 2025 — Chapter 7: "if we quantize our weights to int8 or fp8, B_crit decreases by 2x."*

**Two things quantization improves simultaneously:**

1. **Bandwidth:** Fewer bytes loaded per step → higher arithmetic intensity for the same batch size.
2. **Capacity:** Smaller model size → more HBM headroom → larger feasible batch sizes (which also raises arithmetic intensity).

A 70B INT8 model fits in 70 GB — just within a single H100's 80 GB, versus requiring 2 chips in BF16. That alone doubles the effective per-chip throughput ceiling.

**Important caveat:** Quantization is not free. Aggressive INT4 quantization can degrade model quality on reasoning-heavy tasks. Modern quantization-aware training (QAT) and post-training quantization (PTQ) methods have substantially narrowed the quality gap, but the tradeoff exists and must be measured per workload.

---

## Lever 3 — Speculative Decoding

**What it does:** Converts sequential matrix-vector multiplications into batched matrix-matrix multiplications, raising arithmetic intensity without changing the model.

**The mechanism:** A small, cheap "draft" model proposes γ candidate tokens. The large "verifier" model then validates all γ tokens in a single parallel forward pass — processing them as a batch rather than one at a time. Since the verifier processes γ tokens simultaneously, this is a matrix-matrix multiply (high arithmetic intensity) rather than a matrix-vector multiply (low arithmetic intensity).

Expected tokens accepted per round: `α × γ + 1`, where α is the acceptance rate and the `+1` accounts for the token always generated at the end of the verification step.

**The amortized arithmetic intensity:**

```
Total FLOPs = 2 × (Nv + Nd) × γ
Total Bytes = Nv × bpp + Nd × bpp × γ

AI_amortized = Total FLOPs / Total Bytes
```

Where Nv = verifier params, Nd = draft params, γ = speculation length, bpp = bytes per param. As γ increases, the verifier's weight-load cost is amortized across more tokens, raising AI_amortized.

> *Source: MoE-SpeQ (arXiv 2511.14102, 2025) — introduces the Amortization Roofline Model for speculative decoding with MoE expert offloading. Achieves up to 2.34× speedup for Phi-MoE over state-of-the-art offloading.*

**The empirical picture from Google DeepMind:** Testing on Chinchilla (a 70B model) with a 4B parameter draft model, the optimal speculation depth was 3–4 tokens ahead for natural language tasks (XSum), while coding tasks (HumanEval) — which are more predictable — benefited from more aggressive speculation.

> *Source: Austin et al., 2025 — Appendix D, "Speculative Sampling".*

**Two ways to win:** Speculative decoding is primarily a **latency** win — by generating multiple accepted tokens per verifier forward pass, time-to-last-token shrinks. It can also become a **throughput** win in long-context settings, because the KV cache load is amortized across multiple verified tokens per step.

**The catch:** A draft model must exist with a distribution close to the target model's, or acceptance rates collapse. Fine-tuned smaller versions, model-embedded draft heads, or n-gram-based drafters each make different quality/cost tradeoffs.

→ *Explore this interactively: [Speculative Decoding Amortizer](https://monamishra95.github.io/roofline-model-llm-inference/amortized-roofline/) — Tab ③.*

---

## Lever 4 — KV Cache Reduction

**What it does:** Shrinks the KV cache, enabling larger batch sizes within the same HBM budget and reducing attention's memory bandwidth cost.

**Why the KV cache matters so much:** During generation, attention is always memory-bandwidth-bound — the arithmetic intensity of the KV lookup is approximately 1 FLOPs/Byte, regardless of batch size. This is because every batch item has its own KV cache, so a bigger batch means more KV caches to load. The KV cache can easily dwarf model weights at scale.

**The KV cache size formula:**

```
KV cache size = 2 × bytes_per_float × H × K × L × T
```

Where H = head dimension, K = number of KV heads, L = number of layers, T = sequence length.

For LLaMA 2-13B (L=40, K=40, H=128) at 8,192 tokens in BF16, this is **6.7 GB per sequence**. With a batch of only 4, the KV caches (26.8 GB) already exceed the model weights (26 GB). This dramatically limits feasible batch sizes and caps throughput well below B_crit.

> *Source: Austin et al., 2025 — Chapter 7: "Just 4 of these exceed the memory usage of our parameters!"*

**Architectural techniques to reduce it:**

**Grouped Query Attention (GQA):** Shares KV heads across multiple Q heads. With a 5× reduction in KV heads, the theoretical maximum throughput for a LLaMA 2-13B equivalent on 8× TPU v5e improves from ~963 tokens/s to ~4,529 tokens/s at the same topology.

> *Source: Austin et al., 2025 — Chapter 7 table comparing standard MHA vs. 5× GQA-reduced KV cache. LLaMA 3 8B uses exactly this: 32 Q heads, 8 KV heads.*

**Multi-head Latent Attention (MLA):** Used in DeepSeek-V3 and DeepSeek-R1. Compresses the KV cache by projecting keys and values into a low-dimensional latent space before caching, then reconstructing at attention time. Achieves more aggressive compression than GQA at higher computational cost.

**Systems-level techniques:**

**Prefix caching:** For chatbots and few-shot prompts, the KV cache for the shared prefix is computed once and reused across subsequent requests. Since the KV cache is autoregressive (token N's cache depends only on tokens 1..N), shared prefixes never need recomputation. Google's JetStream and vLLM both implement this.

**PagedAttention:** Avoids allocating maximum-context KV memory upfront. Stores KV caches in OS-style virtual memory pages, only reading the non-padding portion of each request's history. Eliminates the memory waste of padding all requests to the maximum context length.

---

## Lever 5 — Disaggregated Serving

**What it does:** Separates prefill and decode onto dedicated hardware, allowing each to be optimized independently.

**The fundamental mismatch:** As established in Part 1, prefill and decode have arithmetic intensities that differ by 100–500×:

| Phase   | Operation Type      | Arithmetic Intensity   | Zone on H100       |
|---------|---------------------|------------------------|--------------------|
| Prefill | Matrix-matrix       | ~100–500 FLOPs/Byte    | Near compute-bound |
| Decode  | Matrix-vector       | ~0.5–2 FLOPs/Byte      | Deeply memory-bound|

No single chip configuration — sharding strategy, batch size, or precision — can be simultaneously optimal for both. Optimizing for prefill (high MXU utilization, minimal sharding) hurts decode throughput. Optimizing for decode (heavy model sharding, large batches) wastes time on prefill.

**System-level disaggregation:** Prefill servers generate KV caches and pass them across the network to decode servers, which maintain large batches of concurrent generation requests. Each pool can be independently scaled, sharded, and scheduled.

> *Source: Austin et al., 2025 — Chapter 7, "Disaggregated Serving": "for latency-sensitive, high-throughput serving, we typically have to separate prefill and generation into separate servers."*

JetStream, Google's open-source inference engine, implements this architecture with separate prefill engines, generate engines, and a transfer thread that moves KV caches between them.

**Hardware-level disaggregation — SPAD:** System-level disaggregation still uses general-purpose chips for both pools. The SPAD paper (arXiv 2510.08544, 2025) goes a step further: designing different silicon for each phase.

- **Prefill chips:** Larger systolic arrays (to maximize MXU utilization on matrix-matrix multiplies), cost-effective GDDR memory (prefill doesn't need HBM's extreme bandwidth). Result: **8% higher prefill performance at 52% lower hardware cost** vs. modeled H100.
- **Decode chips:** High HBM bandwidth (to minimize time loading weights per token), reduced compute capacity (the MXU cannot be saturated anyway). Result: **97% of decode performance at 28% lower TDP** vs. modeled H100.

End-to-end on production traffic traces: **19–41% hardware cost reduction and 2–17% TDP reduction** compared to H100 clusters at the same throughput target. Even under workload shifts, SPAD chips can be reallocated across phases, achieving **11–43% lower hardware costs** in dynamic scenarios.

> *Source: SPAD (arXiv 2510.08544, 2025) — "Specialized Prefill and Decode hardware."*

→ *Explore this interactively: [Disaggregated Prefill / Decode](https://monamishra95.github.io/roofline-model-llm-inference/amortized-roofline/) — Tab ④.*

---

## Lever 6 — Hardware Selection

**What it does:** Matches the chip's ridge point to the workload's arithmetic intensity, maximizing attainable performance per dollar.

**The ridge point as a design variable:** The ridge point (π/β = Peak Compute / Memory Bandwidth) is not a fixed truth — it is a chip design choice. Different chips land in very different places:

| Chip       | Peak Compute (BF16) | HBM Bandwidth | Ridge Point   |
|------------|---------------------|---------------|---------------|
| H100 SXM5  | 989 TFLOPS          | 3,350 GB/s    | ~295 FLOPs/B  |
| H200 SXM5  | 1,457 TFLOPS        | 8,000 GB/s    | ~182 FLOPs/B  |
| A100 80GB  | 312 TFLOPS          | 2,000 GB/s    | ~156 FLOPs/B  |
| TPU v4     | 275 TFLOPS          | 1,200 GB/s    | ~229 FLOPs/B  |

A key counterintuitive result: the **H200 has a lower ridge point than the H100**. Its 8 TB/s bandwidth grew faster than its compute. For memory-bound decode workloads, this means a given batch size reaches a higher fraction of the H200's compute ceiling than it would on an H100. **More total TFLOPS does not automatically mean better decode throughput.** What matters is how close the workload's arithmetic intensity is to the ridge point.

**Matching the chip to the phase:**

- **For prefill:** High ridge point (high compute, moderate bandwidth). The workload's intensity is already high — you want a chip whose compute ceiling is hard to reach, because you can.
- **For decode:** Low ridge point (high bandwidth, moderate compute). The workload's intensity is low — you want a chip whose ridge point comes down to meet the workload, not one whose compute peak remains permanently out of reach.

This is not a new insight — it is why SPAD's decode chips "retain high memory bandwidth but reduce compute capacity." It is optimal chip design, derived directly from the Roofline model.

**Practical implication:** A team selecting hardware for a new inference deployment should calculate their expected arithmetic intensity (using a tool like the [Arithmetic Intensity Estimator](https://monamishra95.github.io/roofline-model-llm-inference/calculator/)) *before* selecting a chip — not after. A chip whose ridge point is 295 FLOPs/Byte is a poor choice for a workload running at 1–5 FLOPs/Byte, regardless of its peak TFLOPS spec sheet number.

---

## Conclusion — The Co-Design Imperative

The six levers above span a spectrum from immediately deployable to architecturally fundamental:

| Lever | Type | Primary Benefit | Ceiling |
|-------|------|-----------------|---------|
| Batching | Software/Systems | Higher MXU utilization | KV cache memory, latency targets |
| Quantization | Software | Lower bytes/param, higher B_crit | Model quality degradation |
| Speculative Decoding | Software | Lower latency, throughput in long context | Draft model quality, acceptance rate |
| KV Cache Reduction | Architecture | Enables larger batches, lowers attention cost | Model design constraints |
| Disaggregated Serving | Systems/Silicon | Independent optimization of each phase | Network transfer of KV caches |
| Hardware Selection | Silicon | Ridge point aligned to workload | Chip availability, total memory |

What the table reveals: most of the levers available today are **software adaptations to a hardware mismatch**. Batching, quantization, speculative decoding, and KV cache techniques are all ways of reshaping the workload to fit silicon that was not designed with inference decode in mind. They extract real wins — but they are fundamentally working around a structural problem, not solving it.

The structural solution is co-design: building hardware whose characteristics — memory bandwidth, compute capacity, systolic array dimensions, memory technology — are derived from a clear-eyed analysis of what the workload actually needs. As the DeepMind Scaling Book frames it:

> *"Hardware designers face the inverse problem: building hardware that provides just enough compute, bandwidth, and memory for our algorithms while minimizing cost. You can imagine how stressful this 'co-design' problem is."*

> *Austin et al., "How to Scale Your Model", Google DeepMind, 2025.*

SPAD's 19–41% cost reduction is not a software optimization. It is recovered by removing silicon that the decode workload structurally cannot use — and reinvesting that silicon budget in bandwidth that the workload actually needs.

The Roofline Model is the quantitative language of this problem. Every architectural decision — how many MXU cores, how much HBM bandwidth, how to split prefill from decode — can be evaluated against it. The chip already knows what it can do. The question for the next generation of hardware is whether the architecture was designed to close the gap.

---

## Try It Yourself

**→ [Part 1 — The Framework](./README.md)** — the Roofline Model, arithmetic intensity, and why decode is memory bound.

**→ [Interactive Roofline Calculator](https://monamishra95.github.io/roofline-model-llm-inference/calculator/)** — explore ridge points and MXU utilization across chips and batch sizes.

**→ [Amortized Roofline Analyzer](https://monamishra95.github.io/roofline-model-llm-inference/amortized-roofline/)** — four-tab tool covering MoE efficiency, speculative decoding amortization, and disaggregated prefill/decode.

---

## Sources

1. Austin, J. et al. *How to Scale Your Model.* Google DeepMind, 2025. [https://jax-ml.github.io/scaling-book/](https://jax-ml.github.io/scaling-book/)
2. SPAD: *Specialized Prefill and Decode Hardware for LLM Inference.* arXiv:2510.08544, 2025. [https://arxiv.org/abs/2510.08544](https://arxiv.org/abs/2510.08544)
3. MoE-SpeQ: *Overcoming the I/O Bottleneck for MoE Inference via Speculative Expert Prefetching.* arXiv:2511.14102, 2025. [https://arxiv.org/abs/2511.14102](https://arxiv.org/abs/2511.14102)
4. Leviathan, Y. et al. *Fast Inference from Transformers via Speculative Decoding.* arXiv:2211.17192, 2022.
5. Chen, C. et al. *Accelerating Large Language Model Decoding with Speculative Sampling.* arXiv:2302.01318, 2023.

---

*Written by **Mona Mishra**. If this was useful, ⭐ the repo: [github.com/monamishra95/roofline-model-llm-inference](https://github.com/monamishra95/roofline-model-llm-inference)*
