# Inference Efficiency from First Principles

### A guide to the Amortized Roofline Analyzer, for engineers reasoning about where their tokens-per-second actually go

This document explains every tab in the analyzer from the ground up. It is not a button-by-button tour. The goal is a mental model precise enough that when someone proposes buying more compute to make a model faster, you can say whether it will help, and why. Read the foundation section once. After that, each tab is a short application of the same three or four ideas.

---

## Part 0. The foundation every tab is built on

### 0.1 Two costs, not one

A chip doing LLM inference is doing two things that are priced separately. It does arithmetic, meaning multiply-accumulate operations on matrices. And it moves data, meaning weights, activations, and KV cache traveling between memory and the compute units. Most engineers track the first cost and ignore the second. The whole discipline of inference efficiency comes down to one discovery. For the workloads we actually run, the second cost is usually the one that binds.

The unit of arithmetic is the FLOP, one floating-point operation. A chip's headline speed is FLOP/s. An H100 does roughly 989 trillion BF16 tensor FLOP/s. The unit of data movement is the byte, and the chip's ability to move it is memory bandwidth, in bytes per second. The same H100 moves about 3,350 GB/s (3.35 TB/s) between HBM and its compute units. These two numbers, call them π for peak compute and β for peak bandwidth, are fixed the day the chip is manufactured. Everything after that is about how well a given workload uses them.

### 0.2 Arithmetic intensity, a property of the workload

The link between the two costs is one ratio.

```
Arithmetic Intensity (I) = FLOPs performed / bytes moved from memory
```

Intensity belongs to the workload, its algorithm, batch size, and sequence length. Not to the hardware. A workload with high intensity does a lot of math per byte fetched, so the compute units stay fed. A workload with low intensity fetches a byte, does almost nothing with it, and fetches again. The compute units sit idle, waiting on the memory bus.

Think of the compute unit as a chef and memory as the walk-in fridge. Low intensity is a chef who walks to the fridge, grabs one egg, cracks it, and walks back, ten thousand times. Most of the shift is spent walking. High intensity is a chef who loads a full cart, then works through it for two hours without moving. Same chef, same fridge. The only thing that changed is how much work happens per trip.

### 0.3 The Roofline model, one equation

Put π, β, and I together and you get the attainable performance of any hardware-workload pair.

```
P = min( π , β × I )
```

Read the min() literally. Performance is capped by whichever ceiling you hit first. If intensity is low, β × I is the smaller term, and the workload is memory-bound. Throughput scales with bandwidth and intensity, and adding compute does nothing. If intensity is high, π is the smaller term, and the workload is compute-bound. The memory pipe is saturated, and the only way to go faster is a chip with more FLOP/s.

Plot P against I on log-log axes and you get the roofline: a diagonal line for the memory ceiling (β × I) that rises until it meets a flat line for the compute ceiling (π). Every workload is a point somewhere under that roof. The chart in each tab is a version of this plot.

### 0.4 The ridge point, the number that decides everything

The two ceilings cross at one intensity value, called the ridge point.

```
I_ridge = π / β    (FLOPs per byte)
```

This is the threshold. A workload below the ridge point is memory-bound. Above it, compute-bound. The ridge point is fixed for a given chip, and it is the single most useful number to remember about a piece of hardware.

| Chip       | Peak compute (BF16) | Bandwidth   | Ridge point   |
|------------|---------------------|-------------|---------------|
| H100 SXM5  | 989 TFLOP/s         | 3,350 GB/s  | ~295 FLOPs/B  |
| H200 SXM5  | 989 TFLOP/s         | 4,800 GB/s  | ~206 FLOPs/B  |
| A100 80GB  | 312 TFLOP/s         | 2,000 GB/s  | ~156 FLOPs/B  |
| TPU v4     | 275 TFLOP/s         | 1,200 GB/s  | ~229 FLOPs/B  |

Look at the H200. It is newer and more expensive than the H100, and its ridge point is lower. That's not a typo. The H200 uses the same GH100 die, so compute is unchanged at 989 TFLOP/s. Only memory improved, with 43% more bandwidth and 76% more capacity. Since the ridge point is π/β, raising β while π stays fixed pulls the ridge point down. For memory-bound work, that's the right trade. But it means "newer and more expensive" does not mean "more FLOP/s." The ridge point tells you what kind of chip you're actually looking at.

### 0.5 Why LLM decode is almost always memory-bound

Here is the part that surprises people. Autoregressive generation, producing one token at a time, is a matrix-vector multiplication. To generate one token, the model loads its full weight matrix from memory and multiplies it against a single activation vector. For a 70B model in BF16, that's about 140 GB of weights fetched to produce one token. The intensity of that operation is roughly batch_size divided by bytes_per_parameter. At batch size 1 in BF16, that's 0.5 FLOPs per byte.

Against an H100 ridge point of 295, an intensity of 0.5 means the workload runs at roughly 0.2% of the chip's compute potential. The rest of the MXU sits idle, waiting for the memory bus. This is why a team can double their compute budget and not generate a single token faster. They were never limited by compute in the first place.

Prefill, processing the prompt, works the other way. It's a matrix-matrix multiplication over all prompt tokens at once, with intensity around (batch × sequence_length) divided by bytes_per_parameter. That lands in the hundreds, often above the ridge point. Same model, same chip, compute-bound during prefill and memory-bound during decode. Keep that in mind. Tab 4 is built on it.

The one idea to carry into every tab below: each technique here moves a workload's point rightward on the roofline, raising its arithmetic intensity, so hardware you already own stops sitting idle.

---

## The tabs

The eight tabs split into two groups. Tabs 1 through 4 cover the physics, where inference efficiency comes from and what changes it. Tabs 5 through 8 cover the economics, translating that physics into dollars and market context. Presented in order, the argument builds from "here is why your chips are idle" to "here is what that costs, and what to do about it."

### Tab 1. Capability-Efficiency Frontier

**The question:** before optimizing how we serve a model, which model should we serve?

**From first principles.** Section 0.5 showed that per-token memory cost scales with the number of parameters loaded. But not every parameter is necessarily loaded for every token. A dense model activates all its weights on every forward pass, so a 70B dense model loads 70B parameters per token. A Mixture-of-Experts (MoE) model routes each token to a small subset of expert sub-networks, so it loads only its active parameter count, even if its total parameter count is much larger. DeepSeek-V3 has 671B total parameters but activates only 37B per token. Its memory bill, and therefore its decode speed, is set by that 37B, not the 671B.

This tab plots open-weight models with active parameters per token on the x-axis and capability (MMLU) on the y-axis. The point is one relationship. MoE models reach frontier capability at a fraction of the active-parameter cost of an equivalent dense model. Capability and serving cost don't have to rise together.

**How to read it.** Points toward the upper left are the prize, high capability, low active-parameter cost, low per-token memory traffic. A dense model sitting to the right of an MoE model with the same MMLU score is, for inference purposes, worse. You're paying more memory bandwidth per token for the same quality.

**The decision it informs.** Model selection is the first and largest efficiency lever, and it happens before any serving optimization. Choosing a well-designed MoE model over a dense model of equal quality can cut per-token memory traffic several times over. That's often a bigger win than any serving trick that follows. The caveat, covered in Tab 2, is that "active parameters" is a best case, and real routing erodes it.

### Tab 2. MoE Roofline Placement

**The question:** how much of the MoE efficiency promise survives a real serving batch?

**From first principles.** The pitch for MoE is "load only the active experts." True for one token. But production inference runs a batch of tokens at once, and different tokens route to different experts. If a 128-expert model activates 1 expert per token, but the 32 tokens in a batch collectively touch 20 different experts, the hardware has to load all 20. That's far more than the "1 active expert" headline suggests. This is expert-loading overhead, the ratio between the experts actually loaded across a batch and the experts a single token would suggest. It runs from 1.0x, meaning every token in the batch hit the same experts, up toward loading the full model in the worst case.

The tab lets you set batch size and an overhead factor, then computes where the workload lands.

```
ideal intensity  = batch / bytes_per_parameter          (the marketing number)
actual intensity = batch / (bytes_per_parameter × overhead)   (what routing gives you)
```

It plots both points on the roofline against the chosen chip's ridge point, and reports the resulting MXU utilization.

**How to read it.** Watch the gap between the ideal and actual points as the overhead slider rises. At 1.0x they sit together, and MoE delivers its full promise. As overhead climbs toward 3x, the actual point moves left and down. Memory traffic you thought was eliminated comes back, and utilization drops. MoE efficiency is not a fixed property of the model. Your batching and routing strategy decide how much of it you keep.

**The decision it informs.** If you run MoE models, expert-routing behavior across your real batch distribution is worth measuring. It sets how much of the advertised efficiency you actually get. Techniques like expert-parallel placement and routing-aware batching exist to keep this overhead close to 1x.

### Tab 3. Speculative Decoding Amortizer

**The question:** can decode's arithmetic intensity go up without changing the model?

**From first principles.** Decode is memory-bound (Section 0.5) because it's sequential and matrix-vector. One weight load buys exactly one token. Speculative decoding attacks that ratio directly. A small, cheap draft model proposes γ candidate tokens through fast autoregression. Then the large verifier model checks all γ+1 positions in one forward pass, which, because it processes multiple positions at once, is matrix-matrix, not matrix-vector. One expensive weight load from the verifier now buys several accepted tokens instead of one. Arithmetic intensity rises, and the workload's point on the roofline moves right.

How many tokens does one verifier pass buy on average? With a per-token acceptance rate α and speculation length γ, the expected number of accepted tokens per round is

```
E = (1 − α^(γ+1)) / (1 − α)
```

Two things matter here. E rises with both α, a better-matched draft model, and γ, a longer speculative run. And the ceiling engineers most often forget is that at perfect acceptance (α = 1), the gain caps at exactly γ+1. You can't amortize a weight load across more tokens than you speculated. Set γ = 8 and your gain caps at 9x, no matter how good the draft model is.

**How to read it.** Set verifier, draft model, chip, γ, and acceptance rate, and the chart shows the baseline decode point and the amortized point on the same roofline. The gap between them is the speedup. If the amortized point is still left of the ridge, decode is still memory-bound, the gap narrowed but didn't close. If it crosses the ridge, decode is now compute-bound.

**The decision it informs.** Speculative decoding is a serving-side change, no model retraining, that can multiply decode throughput several times over. Its payoff depends entirely on acceptance rate. The tab makes the two knobs, draft-model quality and speculation length, and their interaction visible, so you can tell whether a given draft model is worth building before you build it.

### Tab 4. Disaggregated Prefill / Decode

**The question:** if prefill and decode want such different things, why serve them on the same chip?

**From first principles.** Section 0.5 laid out the split. Prefill is matrix-matrix, intensity in the hundreds, compute-bound, and wants a high-π chip whose MXU it can actually saturate. Decode is matrix-vector, intensity near 1, memory-bound, and wants a high-β chip with large HBM. The ratio between the two intensities is roughly the sequence length, which puts it in the 100 to 500x range for typical prompts.

That's a 100 to 500x mismatch in what the two phases want from silicon, and the standard deployment runs both on one generalist chip anyway. The tab lets you set a model, chip, decode batch size, and prefill sequence length, then plots both operating points on one roofline. Prefill sits near the compute ceiling at high utilization. Decode sits near the floor at a fraction of a percent. Same chip, same time.

**How to read it.** The distance between the two points is the argument. One chip can't be optimal for both. Size it for prefill and memory bandwidth goes to waste during decode. Size it for decode and compute goes to waste during prefill. Either way, one phase wastes a large share of the hardware. The three cards below the chart spell out the implication. A decode-optimized chip wants bandwidth over FLOP/s. A prefill-optimized chip wants the opposite. And research (SPAD) finds that serving the two phases on specialized hardware cuts total cost of ownership by 19 to 41% for the same throughput.

**The decision it informs.** This is the case for disaggregated serving and hardware co-design. For a team running inference at scale, splitting prefill and decode into separate pools, even separate chip types, is not an exotic optimization. It recovers a 100 to 500x efficiency gap that a generalist deployment leaves on the table by design.

### Tab 5. Cost & Performance

**The question:** setting workloads aside for a moment, what raw compute does a dollar of hardware buy?

**From first principles.** Tabs 1 through 4 were about intensity, using hardware well. This tab steps back to the hardware itself and asks the simpler question, peak FLOP/s per dollar of list price. It plots BF16 TFLOP/s per $1,000 across chips that have a public list price.

**How to read it.** This is a raw metric on purpose. It says nothing about whether your workload can use those FLOP/s, that's what the roofline tabs are for. It's the baseline before utilization, memory bottlenecks, or cloud markups enter the picture, how much theoretical compute the sticker price buys. Chips sold only through cloud rental, all TPUs and Trainium, have no list price and are left out here. They show up in Tab 7, where rental price is the right lens.

**The decision it informs.** A sanity check on procurement conversations. A chip that looks cheap per FLOP can still be a poor buy if your workload is memory-bound on it, meaning low ridge-point utilization, which is exactly why this tab should not be read alone. It sets up the more honest metric in Tab 7.

### Tab 6. Throughput Head-to-Head

**The question:** in practice, how many tokens per second do these chips deliver on a real model?

**From first principles.** Peak FLOP/s (Tab 5) is a ceiling almost no real workload reaches. Measured throughput on a standard benchmark is ground truth. It already accounts for memory bottlenecks, kernel efficiency, and everything the roofline predicts. This tab shows measured Llama2-70B offline throughput per accelerator.

**A note worth stating plainly when you present this.** These figures are labeled illustrative, not current-gen. They cover the H100, B200, TPU, and MI300X generation to stay consistent with the chips used in tabs 1 through 4. They are not drawn from live benchmark submissions. Checking the current MLPerf Inference v6.0 results from April 2026, none of these chips appear. The field has moved to AMD MI355X, NVIDIA B300-SXM, and NVIDIA RTX PRO 6000 Blackwell running at FP4 precision, and Google did not submit at all. Treat this tab as a teaching device for the shape of the comparison, not as current numbers. That's a useful lesson on its own. Benchmark leadership moves fast, and any hard-coded throughput table ages quickly.

**The decision it informs.** It grounds the abstract roofline reasoning in a number people can hold onto, and it feeds Tab 7, where throughput is the denominator in the cost-per-token math.

### Tab 7. Cost per 1M Tokens

**The question:** the one finance actually asks. What does it cost to serve a million tokens, and what changes it?

**From first principles.** This is where physics turns into money. The cost to produce a fixed number of tokens is

```
cost per 1M tokens = price_per_chip_hour / (throughput_tokens_per_sec × utilization × 3,600) × 1,000,000
```

Three inputs, each traceable to something earlier. Throughput comes from Tab 6, and from the roofline before that. Price is the cloud provider's on-demand list rate, checked against live provider pages. Utilization is the share of peak throughput actually sustained in production, and it's the lever teams underestimate most consistently.

**How to read it.** Move the utilization slider and watch cost shift. Because utilization sits in the denominator, halving it doubles cost per token. A team running at 30% utilization pays more than three times what the same hardware costs a team running at 90%. The comparison chart ranks chips by cost per token on the same cloud at the same utilization, and it often reorders the "cheapest chip" ranking from Tab 5's raw sticker view, because a chip with mediocre FLOP/s per dollar but strong real throughput can win on cost per token.

**A correctness note worth mentioning when presenting.** Checking the cloud prices live caught a real error. An AWS H100 rate had been entered at roughly double the true figure, $6.88 per chip-hour, from a p5.48xlarge at $55.04 per hour across 8 GPUs. Small input errors in this formula flow straight to the bottom line. That's why the tab shows its price assumptions instead of hiding them.

**The decision it informs.** This is the tab for a budget conversation. Every intensity gain from tabs 1 through 4 shows up here as lower cost per token, and it makes the case that raising utilization is often the cheapest saving available, since it requires no new hardware.

### Tab 8. Market Context

**The question:** what's the commercial backdrop this sits inside?

**From first principles.** The first seven tabs are about making your own inference cheaper. This one zooms out to the market that prices the hardware and rents the capacity. It shows quarterly cloud infrastructure spend, year-over-year growth, and the share split across major providers, from Synergy Research.

**How to read it.** Treat it as framing, not as a lever you can pull. The relevant story for an efficiency audience is the growth rate and its direction. Cloud infrastructure spend is large and growing, which is what makes per-token efficiency a real financial concern rather than a rounding error. When the underlying spend grows double digits a year, a 30% efficiency gain is a large absolute number.

**The decision it informs.** It's the "why this matters" context, useful at the start or end of a presentation, connecting the microarchitecture physics of tabs 1 through 4 to the spend that makes the work worth doing.

---

## Putting it together, a decision workflow

Presented in sequence, the tabs form one argument.

The problem, from the roofline in Part 0, is that inference chips sit mostly idle during decode, because decode is memory-bound, and extra compute doesn't fix a memory bottleneck. The first lever, Tab 1, is choosing the right model. A well-designed MoE gives frontier quality at a fraction of the per-token memory cost. The catch, Tab 2, is that real batching erodes that advantage through expert-loading overhead, so routing behavior needs managing. The serving-side multiplier, Tab 3, is speculative decoding, which raises decode intensity without touching the model, bounded by acceptance rate and speculation length. The structural fix, Tab 4, is that prefill and decode want opposite hardware, so serving them separately recovers a 100 to 500x mismatch and 19 to 41% of total cost of ownership. Then the economics. Raw compute per dollar in Tab 5. Real measured throughput in Tab 6. The cost-per-token metric that ties physics to budget and exposes utilization as the cheapest lever, in Tab 7. And the market backdrop that makes all of it worth doing, in Tab 8.

## What this tool does not model

Worth stating up front, since it heads off the sharpest questions.

The roofline gives theoretical ceilings. Real kernels never hit them exactly, so treat every attainable-performance figure as an upper bound. The intensity formulas are first-order. They capture the dominant weight-loading term and leave out KV-cache growth, attention's own memory traffic, and communication overhead across sharded chips, all of which matter at long context and large scale. The throughput numbers in tabs 6 and 7 reflect an older chip generation, not current benchmark data. And the whole framework is about decode-time, memory-bound inference. It says little about training, which is a different, more compute-bound regime. None of this undercuts the core argument. It just marks where the argument's edges are, which is usually where the interesting engineering work is.

---

*Built on the Roofline framework from "Why More Compute Does Not Mean Faster AI" (Part 1) and the six levers in Part 2 of this repository. The analyzer implements the math described here. The tabs are live and interactive at the project's GitHub Pages site.*
