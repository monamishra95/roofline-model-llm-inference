# Part 2 Preview — The Levers: How to Fix a Memory-Bound Workload

> *This is a preview of Part 2. [Subscribe](https://monamis.substack.com) or ⭐ this repo to be notified when it drops.*

---

Part 1 established the problem: almost all LLM inference is Memory Bound, and teams are paying for compute they structurally cannot use.

Part 2 is about what to do about it.

## What's Coming

### Lever 1: Batching
Batching is the single most powerful lever. Increasing batch size directly increases arithmetic intensity — it amortizes the weight load cost across more tokens. Going from batch=1 to batch=128 can shift a 70B decode from 0.5 FLOPs/Byte to ~64 FLOPs/Byte, moving it materially up the roofline.

### Lever 2: Quantization
Reducing precision (BF16 → INT8 → INT4) cuts the bytes-per-parameter, shrinking model size and reducing memory pressure. INT8 doubles memory efficiency vs. BF16; INT4 quadruples it. With modern quantization-aware training, quality loss is often negligible.

### Lever 3: Speculative Decoding
A technique where a small "draft" model proposes multiple tokens, and the large "verifier" model validates them in parallel. This converts sequential matrix-vector operations into batch matrix multiplications — increasing effective arithmetic intensity without changing the model.

### Lever 4: Disaggregated Prefill / Decode
Prefill (processing the prompt) and decode (generating tokens) have fundamentally different arithmetic intensities. Running them on separate chip pools — each tuned for its workload — can dramatically improve overall utilization.

### Lever 5: KV Cache Optimization
The key-value cache grows with sequence length and is loaded from HBM on every decode step. Techniques like GQA (Grouped Query Attention), MLA (Multi-head Latent Attention, used in DeepSeek), and paged attention reduce KV cache size and memory bandwidth pressure.

### Lever 6: Hardware Selection by Workload
Not all chips are created equal for all workloads. The right chip is the one whose ridge point aligns with your actual arithmetic intensity — not the one with the highest peak TFLOPS number on a spec sheet.

---

*Part 2 drops soon. In the meantime, try the [calculator](./calculator/index.html) and [Python script](./scripts/arithmetic_intensity.py) to understand where your specific workload sits.*
