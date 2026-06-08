"""
arithmetic_intensity.py
-----------------------
Estimate the arithmetic intensity of an LLM workload and determine whether
it is Memory Bound or Compute Bound using the Roofline Model.

Usage:
    python arithmetic_intensity.py
    python arithmetic_intensity.py --model 70b --precision bf16 --chip h100 --phase decode
    python arithmetic_intensity.py --model custom --params 13 --chip tpu_v4 --phase prefill --seq_len 512 --batch 32

Reference: "Why More Compute Does Not Mean Faster AI — The Roofline Model Explained"
           https://github.com/yourusername/roofline-model-llm-inference
"""

import argparse
import math

# ---------------------------------------------------------------------------
# Chip hardware specs (TFLOPS BF16, HBM bandwidth GB/s)
# ---------------------------------------------------------------------------
CHIPS = {
    "h100":    {"name": "H100 SXM5",   "tflops": 989,  "bw_gbs": 3350, "hbm_gb": 80},
    "h200":    {"name": "H200 SXM5",   "tflops": 1457, "bw_gbs": 8000, "hbm_gb": 141},
    "a100":    {"name": "A100 80GB",   "tflops": 312,  "bw_gbs": 2000, "hbm_gb": 80},
    "tpu_v4":  {"name": "TPU v4",      "tflops": 275,  "bw_gbs": 1200, "hbm_gb": 32},
    "tpu_v5e": {"name": "TPU v5e",     "tflops": 197,  "bw_gbs": 819,  "hbm_gb": 16},
}

# ---------------------------------------------------------------------------
# Model presets (billion parameters)
# ---------------------------------------------------------------------------
MODEL_PARAMS = {
    "7b":   7,
    "13b":  13,
    "34b":  34,
    "70b":  70,
    "405b": 405,
}

# ---------------------------------------------------------------------------
# Precision: bytes per parameter
# ---------------------------------------------------------------------------
PRECISION_BYTES = {
    "fp32": 4,
    "bf16": 2,
    "fp16": 2,
    "int8": 1,
    "int4": 0.5,
}

# ---------------------------------------------------------------------------
# Core math
# ---------------------------------------------------------------------------

def bytes_per_param(precision: str) -> float:
    return PRECISION_BYTES[precision.lower()]


def model_size_gb(params_b: float, precision: str) -> float:
    """Model weight size in GB."""
    return params_b * 1e9 * bytes_per_param(precision) / 1e9


def chips_needed(params_b: float, precision: str, chip: dict) -> int:
    """Minimum chips required to hold the model."""
    size = model_size_gb(params_b, precision)
    return math.ceil(size / chip["hbm_gb"])


def decode_arithmetic_intensity(params_b: float, precision: str, batch_size: int) -> float:
    """
    Decode phase: matrix-vector multiply.
    Each token generation loads the full weight matrix (~2 * params bytes for BF16 due to read+write)
    and performs ~2 * params FLOPs.

    AI ≈ batch_size  (each additional batch item reuses the weights)
    Exact: (2 * params_b * 1e9 * batch) / (2 * params_b * 1e9 * bytes_per_param)
         = batch / bytes_per_param
    """
    bpp = bytes_per_param(precision)
    return batch_size / bpp


def prefill_arithmetic_intensity(params_b: float, precision: str,
                                  batch_size: int, seq_len: int) -> float:
    """
    Prefill phase: matrix-matrix multiply (parallel over all tokens in the prompt).
    AI ≈ (batch * seq_len) / bytes_per_param
    This simplification holds when seq_len >> 1 and ignores KV cache overhead.
    """
    bpp = bytes_per_param(precision)
    return (batch_size * seq_len) / bpp


def ridge_point(chip: dict) -> float:
    """The arithmetic intensity threshold above which a workload is Compute Bound."""
    return chip["tflops"] * 1e12 / (chip["bw_gbs"] * 1e9)


def attainable_perf(chip: dict, intensity: float) -> float:
    """Attainable performance in TFLOPS, given the Roofline model."""
    return min(chip["tflops"], chip["bw_gbs"] * intensity / 1e3)


# ---------------------------------------------------------------------------
# Report
# ---------------------------------------------------------------------------

def print_separator():
    print("─" * 60)


def run_analysis(params_b: float, precision: str, chip_key: str,
                 phase: str, batch_size: int, seq_len: int):
    chip = CHIPS[chip_key]
    ridgept = ridge_point(chip)

    if phase == "decode":
        intensity = decode_arithmetic_intensity(params_b, precision, batch_size)
        phase_label = f"Decode (batch={batch_size})"
    else:
        intensity = prefill_arithmetic_intensity(params_b, precision, batch_size, seq_len)
        phase_label = f"Prefill (batch={batch_size}, seq_len={seq_len})"

    perf = attainable_perf(chip, intensity)
    util_pct = (perf / chip["tflops"]) * 100
    size_gb = model_size_gb(params_b, precision)
    n_chips = chips_needed(params_b, precision, chip)
    is_memory_bound = intensity < ridgept

    print_separator()
    print(f"  ROOFLINE ANALYSIS")
    print_separator()
    print(f"  Model         : {params_b}B parameters, {precision.upper()}")
    print(f"  Model size    : {size_gb:.1f} GB  →  needs ≥ {n_chips} {chip['name']} chip(s)")
    print(f"  Chip          : {chip['name']}")
    print(f"  Phase         : {phase_label}")
    print_separator()
    print(f"  Ridge Point   : {ridgept:.1f} FLOPs/Byte")
    print(f"  Your Intensity: {intensity:.2f} FLOPs/Byte  ({intensity/ridgept*100:.1f}% of ridge)")
    print_separator()
    print(f"  Peak Compute  : {chip['tflops']} TFLOPS")
    print(f"  Attainable    : {perf:.1f} TFLOPS  ({util_pct:.1f}% MXU utilization)")
    print_separator()

    if is_memory_bound:
        gap = ridgept / intensity
        print(f"  ZONE: ⚠️  MEMORY BOUND")
        print(f"")
        print(f"  The MXU is idle {100 - util_pct:.0f}% of the time, waiting for data.")
        print(f"  To reach the ridge point, you need {gap:.0f}x more arithmetic intensity.")
        print(f"")
        print(f"  Levers to try:")
        print(f"    • Increase batch size (currently {batch_size})")
        print(f"    • Use lower precision (currently {precision.upper()})")
        print(f"    • Enable continuous batching / request batching")
        print(f"    • Apply speculative decoding (increases effective batch size)")
        print(f"    • Use quantization-aware serving (INT8/INT4)")
    else:
        print(f"  ZONE: ✅  COMPUTE BOUND")
        print(f"")
        print(f"  The memory pipe is saturated. The MXU is running near full utilization.")
        print(f"  This is peak efficiency — every dollar of chip value is being captured.")
        print(f"  To go faster, you need a chip with higher peak compute.")

    print_separator()
    print()


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Estimate arithmetic intensity and Roofline zone for an LLM workload.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python arithmetic_intensity.py
  python arithmetic_intensity.py --model 70b --chip h100 --phase decode --batch 1
  python arithmetic_intensity.py --model 70b --chip h100 --phase decode --batch 64
  python arithmetic_intensity.py --model 70b --chip h100 --phase prefill --seq_len 2048 --batch 8
  python arithmetic_intensity.py --model custom --params 13 --chip tpu_v4 --phase decode --batch 16
        """
    )
    parser.add_argument("--model",     choices=list(MODEL_PARAMS.keys()) + ["custom"],
                        default="70b",  help="Model size preset (default: 70b)")
    parser.add_argument("--params",    type=float, default=None,
                        help="Parameter count in billions (required if --model custom)")
    parser.add_argument("--precision", choices=list(PRECISION_BYTES.keys()),
                        default="bf16", help="Numerical precision (default: bf16)")
    parser.add_argument("--chip",      choices=list(CHIPS.keys()),
                        default="h100", help="Target chip (default: h100)")
    parser.add_argument("--phase",     choices=["decode", "prefill"],
                        default="decode", help="Inference phase (default: decode)")
    parser.add_argument("--batch",     type=int, default=1,
                        help="Batch size (default: 1)")
    parser.add_argument("--seq_len",   type=int, default=512,
                        help="Sequence length for prefill (default: 512)")

    args = parser.parse_args()

    if args.model == "custom":
        if args.params is None:
            parser.error("--params is required when --model custom")
        params_b = args.params
    else:
        params_b = MODEL_PARAMS[args.model]

    run_analysis(
        params_b=params_b,
        precision=args.precision,
        chip_key=args.chip,
        phase=args.phase,
        batch_size=args.batch,
        seq_len=args.seq_len,
    )


if __name__ == "__main__":
    main()
