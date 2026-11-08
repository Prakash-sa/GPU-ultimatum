"""Generate the figures for the optimization and profiling cheat sheets.

Run from the repo root:

    .venv/bin/python scripts/make_cheatsheet_figures.py

Writes docs/assets/img/optimization-map.png and docs/assets/img/profiling-funnel.png.
"""
import numpy as np

from make_figures import (AMBER, AMBER_L, BLUE, BLUE_L, GRAY, GRAY_L, INK, MONO, PURPLE, PURPLE_L,
                          RED, RED_L, TEAL, TEAL_L, box, canvas, plt, save)


def optimization_map():
    fig, ax = plt.subplots(figsize=(14, 7.2))
    fig.subplots_adjust(0.07, 0.1, 0.98, 0.9)
    bw, peak = 3.35e3, 67.0            # H100 SXM: GB/s, FP32 TFLOP/s
    x = np.logspace(-2, 3, 400)
    roof = np.minimum(peak, x * bw / 1e3)
    ridge = peak * 1e3 / bw
    ax.loglog(x, roof, color=INK, lw=2.4)
    ax.loglog(x, np.minimum(989.0, x * bw / 1e3), color=PURPLE, lw=1.4, ls="--")
    ax.text(300, 1100, "FP16 tensor-core roof (989)", color=PURPLE, fontsize=9.5, ha="center")
    ax.axvline(ridge, color=GRAY, lw=1, ls=":")
    ax.text(ridge * 1.08, 0.012, f"ridge ≈ {ridge:.0f} FLOP/byte", color=GRAY, fontsize=9.5)
    ax.fill_between(x, 1e-3, roof, where=x < ridge, color=BLUE_L, alpha=0.55)
    ax.fill_between(x, 1e-3, roof, where=x >= ridge, color=AMBER_L, alpha=0.55)
    ax.set_xlim(1e-2, 1e3)
    ax.set_ylim(1e-2, 3e3)
    ax.set_xlabel("arithmetic intensity (FLOP per DRAM byte)")
    ax.set_ylabel("TFLOP/s (H100 SXM, FP32 CUDA cores)")
    ax.set_title("Where is my kernel, and which fixes apply?", fontsize=14, weight="bold", loc="left")

    def note(xy, text, c, ha="left"):
        ax.text(*xy, text, fontsize=9.5, color=c, ha=ha, va="top", family=None,
                bbox=dict(boxstyle="round,pad=0.4", fc="white", ec=c, lw=1.2))

    note((0.013, 1500), "MEMORY BOUND (left of the ridge)\nmove fewer bytes:\n• fuse kernels, keep data on the GPU\n"
         "• coalesce: SoA, threads along rows\n• smaller types (FP32/BF16/FP8)\n• vectorize (float4), reuse via smem/L2\n"
         "• stop at ~80% of DRAM bandwidth", BLUE)
    note((40, 0.6), "COMPUTE BOUND (right of the ridge)\ndo cheaper math:\n• tensor cores (TF32/BF16/FP8)\n"
         "• fewer instructions: FMA, fast math,\n   strength reduction, unroll\n• avoid FP64 on consumer GPUs\n"
         "• stop at ~70-80% of the pipe's peak", AMBER)
    note((0.03, 0.08), "LATENCY BOUND (far below both roofs)\nexpose more parallelism:\n• more blocks/warps, fewer registers\n"
         "• more independent loads per thread (ILP)\n• async copies, pipelining\n• fewer __syncthreads, less divergence", RED)
    ax.scatter([0.083, 0.25, 945], [0.28, 0.6, 40], s=60, color=[BLUE, TEAL, AMBER], zorder=5)
    ax.annotate("vector add", (0.083, 0.28), (0.12, 0.12), fontsize=9, color=BLUE,
                arrowprops=dict(arrowstyle="-", color=BLUE))
    ax.annotate("softmax / norms", (0.25, 0.6), (0.6, 0.25), fontsize=9, color=TEAL,
                arrowprops=dict(arrowstyle="-", color=TEAL))
    ax.annotate("big GEMM", (945, 40), (300, 12), fontsize=9, color=AMBER, arrowprops=dict(arrowstyle="-", color=AMBER))
    note((0.75, 1500), "HOST BOUND (not on this chart: the GPU is idle)\ngaps between kernels on the nsys timeline:\n"
         "• CUDA Graphs, fewer and fused launches\n• no sync or .item() inside the loop\n• pinned memory + async copies on streams", PURPLE)
    save(fig, "optimization-map.png")


def profiling_funnel():
    fig, ax = canvas(15, 8.2, (0, 30), (0, 16.4))
    ax.text(0.3, 15.7, "Profiling: from \"is it busy?\" to \"which line?\"", fontsize=14, weight="bold")
    levels = [
        ("nvidia-smi / DCGM", "Is the GPU busy, hot, throttled, out of memory?", "seconds",
         "nvidia-smi dmon -s pucm", GRAY, GRAY_L, 0.0),
        ("Nsight Systems (nsys)", "Where does wall time go? Gaps? Copies? Syncs? Overlap?", "1 run",
         "nsys profile --stats=true ./app", BLUE, BLUE_L, 1.6),
        ("Nsight Compute (ncu)", "Why is this kernel slow? Memory, compute or latency?", "per kernel",
         "ncu --set full -k regex:name -c 3 ./app", TEAL, TEAL_L, 3.2),
        ("ncu Source view + stalls", "Which line, which stall reason?", "per line",
         "build with -lineinfo, open ncu-ui", AMBER, AMBER_L, 4.8),
        ("compute-sanitizer", "Is it even correct? (OOB, races, uninitialized)", "before timing",
         "compute-sanitizer --tool racecheck ./app", RED, RED_L, 6.4),
    ]
    for k, (tool, q, scope, cmd, c, cl, inset) in enumerate(levels):
        y = 12.6 - k * 2.6
        w = 29.0 - 2 * inset
        box(ax, 0.5 + inset, y, w, 2.1, cl, c, lw=1.6)
        ax.text(0.9 + inset, y + 1.45, tool, fontsize=11.5, weight="bold", color=c)
        ax.text(0.9 + inset, y + 0.55, q, fontsize=10)
        ax.text(0.5 + inset + w - 0.4, y + 1.45, scope, fontsize=9, color=GRAY, ha="right")
        ax.text(0.5 + inset + w - 0.4, y + 0.55, cmd, fontsize=9, family=MONO, ha="right")
    ax.text(0.5, 0.5, "Go down the funnel in order. Most time is lost above the kernel (host gaps, copies), "
            "so ncu comes after nsys; sanitize before you trust any timing.", fontsize=10)
    save(fig, "profiling-funnel.png")


if __name__ == "__main__":
    optimization_map()
    profiling_funnel()
