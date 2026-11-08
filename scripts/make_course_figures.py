"""Generate the figures for the CUDA crash course pages.

Run from the repo root:

    .venv/bin/python scripts/make_course_figures.py

Writes PNG files into docs/assets/img/course/. Reuses the palette and helpers from
make_figures.py so all figures look the same.
"""
from make_figures import (AMBER, AMBER_L, BLUE, BLUE_L, GRAY, GRAY_L, INK, MONO, OUT, PURPLE,
                          PURPLE_L, RED, RED_L, TEAL, TEAL_L, box, canvas, plt)

DIR = OUT / "course"
DIR.mkdir(parents=True, exist_ok=True)


def save(fig, name):
    fig.savefig(DIR / name, dpi=110, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print("wrote course/" + name)


def cell(ax, x, y, text, fc, ec, w=1.0, h=0.8, fs=9, lw=1.1):
    box(ax, x, y, w * 0.94, h, fc, ec, lw=lw)
    if text != "":
        ax.text(x + w * 0.47, y + h / 2, str(text), ha="center", va="center", fontsize=fs, family=MONO)


def arrow(ax, a, b, color=INK, lw=1.5, ls="-", rad=0.0):
    ax.annotate("", xy=b, xytext=a, arrowprops=dict(arrowstyle="-|>", color=color, lw=lw, ls=ls,
                                                    mutation_scale=13, connectionstyle=f"arc3,rad={rad}"))


def label_box(ax, x, y, w, h, text, fc, ec, fs=9.5):
    box(ax, x, y, w, h, fc, ec, lw=1.4)
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=fs)


# ---------------------------------------------------------------- 1. vector add: data paths

def vadd_paths():
    fig, ax = canvas(15, 8.4, (0, 30), (0, 16.8))
    ax.text(0.3, 16.1, "Four ways to get the data to the GPU", fontsize=14, weight="bold")
    rows = [
        ("1. pageable + cudaMemcpy", BLUE, [("your array\n(pageable)", GRAY_L, GRAY), ("hidden pinned\nstaging buffer", AMBER_L, AMBER),
                                            ("PCIe DMA", "white", INK), ("GPU memory", TEAL_L, TEAL)],
         "the driver copies into a pinned buffer first: two copies, lower bandwidth"),
        ("2. pinned (cudaMallocHost)", TEAL, [("your array\n(pinned)", AMBER_L, AMBER), ("PCIe DMA", "white", INK),
                                             ("GPU memory", TEAL_L, TEAL)],
         "the copy engine reads your buffer directly; also required for async copies"),
        ("3. unified (cudaMallocManaged)", PURPLE, [("one pointer\nCPU pages", PURPLE_L, PURPLE), ("page fault\n→ migrate 4-64 KB", RED_L, RED),
                                                   ("page fault\n→ migrate …", RED_L, RED), ("GPU memory", TEAL_L, TEAL)],
         "no explicit copies; pages move on first touch, so the kernel stalls on faults"),
        ("4. unified + prefetch", AMBER, [("one pointer\nCPU pages", PURPLE_L, PURPLE), ("cudaMemPrefetchAsync\nbulk migration", AMBER_L, AMBER),
                                         ("GPU memory", TEAL_L, TEAL)],
         "the pages move in bulk before the kernel: unified convenience, near-pinned speed"),
    ]
    for k, (name, c, stages, note) in enumerate(rows):
        y = 12.6 - k * 3.6
        ax.text(0.5, y + 1.65, name, fontsize=11, weight="bold", color=c)
        x = 0.5
        for j, (t, fc, ec) in enumerate(stages):
            label_box(ax, x, y, 4.6, 1.3, t, fc, ec, fs=9)
            if j < len(stages) - 1:
                arrow(ax, (x + 4.6, y + 0.65), (x + 5.5, y + 0.65), c)
            x += 5.5
        ax.text(23.0, y + 0.65, note, fontsize=9.5, va="center", wrap=True, color=INK)
    save(fig, "vadd-paths.png")


# ---------------------------------------------------------------- 2. matmul: naive vs tiled, overhang

def matmul_steps():
    fig, ax = canvas(16, 6.0, (0, 32), (0, 12.0))
    ax.text(0.3, 11.3, "Matrix multiplication: from one thread per output to shared tiles", fontsize=14, weight="bold")
    # naive
    ax.text(0.5, 10.1, "steps 1-4: each thread walks a row of A and a column of B", fontsize=11, weight="bold", color=BLUE)
    n = 6
    for r in range(n):
        for c in range(n):
            cell(ax, 0.5 + c * 0.7, 4.5 + (n - 1 - r) * 0.7, "", BLUE_L if r == 2 else GRAY_L, BLUE if r == 2 else GRAY, w=0.7, h=0.7)
            cell(ax, 5.5 + c * 0.7, 4.5 + (n - 1 - r) * 0.7, "", TEAL_L if c == 3 else GRAY_L, TEAL if c == 3 else GRAY, w=0.7, h=0.7)
            cell(ax, 10.5 + c * 0.7, 4.5 + (n - 1 - r) * 0.7, "", AMBER if (r == 2 and c == 3) else "white", GRAY, w=0.7, h=0.7)
    for x, t in [(0.5, "A"), (5.5, "B"), (10.5, "C")]:
        ax.text(x + 2.1, 9.0, t, ha="center", fontsize=11, weight="bold")
    ax.text(0.5, 3.4, "2K loads from DRAM/L2 per output, K FMAs: 1 FMA per load", fontsize=9.5)
    ax.text(0.5, 2.6, "neighbouring threads re-read the same row and column", fontsize=9.5, color=RED)
    # tiled
    x0 = 16.5
    ax.text(x0, 10.1, "step 5: a block stages T×T tiles in shared memory", fontsize=11, weight="bold", color=TEAL)
    for r in range(n):
        for c in range(n):
            inA = r in (2, 3) and c in (0, 1)
            inB = r in (0, 1) and c in (2, 3)
            inC = r in (2, 3) and c in (2, 3)
            cell(ax, x0 + c * 0.7, 4.5 + (n - 1 - r) * 0.7, "", BLUE if inA else (BLUE_L if r in (2, 3) else GRAY_L), BLUE if r in (2, 3) else GRAY, w=0.7, h=0.7)
            cell(ax, x0 + 5.0 + c * 0.7, 4.5 + (n - 1 - r) * 0.7, "", TEAL if inB else (TEAL_L if c in (2, 3) else GRAY_L), TEAL if c in (2, 3) else GRAY, w=0.7, h=0.7)
            cell(ax, x0 + 10.0 + c * 0.7, 4.5 + (n - 1 - r) * 0.7, "", AMBER if inC else "white", GRAY, w=0.7, h=0.7)
    for x, t in [(x0, "A"), (x0 + 5.0, "B"), (x0 + 10.0, "C block")]:
        ax.text(x + 2.1, 9.0, t, ha="center", fontsize=11, weight="bold")
    ax.text(x0, 3.4, "each loaded value is used by T threads: T FMAs per load", fontsize=9.5)
    ax.text(x0, 2.6, "tile → __syncthreads → T FMAs → __syncthreads → next tile", fontsize=9.5, color=TEAL)
    ax.text(0.5, 0.8, "Step 2 (no aliasing): __restrict__ + a register accumulator, so C is written once instead of read and written K times. "
            "Steps 3-4: bounds checks for sizes that are not multiples of the block.", fontsize=9.5)
    save(fig, "matmul-steps.png")


def overhang():
    fig, ax = canvas(10, 5.6, (0, 20), (0, 11.2))
    ax.text(0.3, 10.5, "Step 3: N = 50 with 16 × 16 blocks → a 4 × 4 grid covers 64 × 64", fontsize=12.5, weight="bold")
    s = 2.2
    for by in range(4):
        for bx in range(4):
            box(ax, 1.0 + bx * s, 0.8 + (3 - by) * s, s, s, "white", GRAY, lw=1.0)
    m = 50 / 16 * s
    box(ax, 1.0, 0.8 + 4 * s - m, m, m, TEAL_L, TEAL, lw=1.6)
    ax.text(1.0 + m / 2, 0.8 + 4 * s - m / 2, "the 50 × 50 matrix", ha="center", va="center", fontsize=10, color=TEAL)
    box(ax, 1.0 + m, 0.8, 4 * s - m, 4 * s, RED_L, "none", alpha=0.6)
    box(ax, 1.0, 0.8, m, 4 * s - m, RED_L, "none", alpha=0.6)
    ax.text(11.0, 7.0, "threads in the red band\nhave row ≥ N or col ≥ N", fontsize=10, color=RED)
    ax.text(11.0, 5.0, "if (row >= N || col >= N)\n    return;", fontsize=10, family=MONO)
    ax.text(11.0, 3.0, "without it they read and write\npast the end of the arrays", fontsize=10, color=RED)
    save(fig, "matmul-overhang.png")


# ---------------------------------------------------------------- 3. reduction addressing

def reduction():
    fig, ax = canvas(18, 9.6, (0, 36), (0, 19.2))
    ax.text(0.3, 18.5, "Sum reduction in shared memory: who adds what (16 values, 8 threads shown)", fontsize=14, weight="bold")
    vals = list(range(1, 17))

    def panel(x0, title, color, sub, schedule, note):
        ax.text(x0, 16.9, title, fontsize=11.5, weight="bold", color=color)
        ax.text(x0, 16.2, sub, fontsize=9, family=MONO)
        v = list(vals)
        y = 14.6
        for i, val in enumerate(v):
            cell(ax, x0 + i * 0.6, y, val, GRAY_L, GRAY, w=0.6, h=0.65, fs=7)
        for step, pairs in enumerate(schedule):
            y -= 2.4
            nv = list(v)
            for tid, (dst, src) in pairs:
                nv[dst] = v[dst] + v[src]
            for i, val in enumerate(nv):
                active = any(d == i for _, (d, _s) in pairs)
                cell(ax, x0 + i * 0.6, y, val if active else "", color + "44" if active else "white",
                     color if active else GRAY_L, w=0.6, h=0.65, fs=7)
            threads = [t for t, _ in pairs]
            if len(threads) <= 4:
                lab = ", ".join(map(str, threads))
            elif threads == list(range(threads[0], threads[-1] + 1)):
                lab = f"{threads[0]}..{threads[-1]}"
            else:
                lab = f"{threads[0]}, {threads[1]}, .., {threads[-1]}"
            ax.text(x0 + 9.8, y + 0.32, "tid " + lab, fontsize=8, va="center", color=color)
            v = nv
        for k, line in enumerate(note):
            ax.text(x0, 3.6 - k * 0.8, line, fontsize=9, color=RED if line.startswith("✗") else (TEAL if line.startswith("✓") else INK))

    inter = [[(t, (t, t + s)) for t in range(0, 16, 2 * s)] for s in (1, 2, 4, 8)]
    panel(0.5, "1. interleaved (diverged)", RED, "if (tid % (2*s) == 0)", inter,
          ["active threads are 0, 2, 4, … of the block:", "✗ every warp has idle lanes (divergence)", "✗ half the warps do almost nothing"])
    strided = [[(t, (2 * s * t, 2 * s * t + s)) for t in range(16 // (2 * s))] for s in (1, 2, 4, 8)]
    panel(12.5, "2. strided index", AMBER, "idx = 2*s*tid", strided,
          ["active threads are 0..k: no divergence", "✗ idx jumps by 2s: lanes hit the same", "   shared-memory bank (conflicts)"])
    seq = [[(t, (t, t + s)) for t in range(s)] for s in (8, 4, 2, 1)]
    panel(24.5, "3. sequential addressing", TEAL, "if (tid < s) s[tid] += s[tid+s]", seq,
          ["✓ contiguous threads, contiguous addresses", "✓ no divergence until < 32 active", "4-6: add on load, then warp shuffles"])
    save(fig, "reduction-addressing.png")


# ---------------------------------------------------------------- 4. histogram

def histogram():
    fig, ax = canvas(15, 6.0, (0, 30), (0, 12))
    ax.text(0.3, 11.3, "Histogram: where the atomics land", fontsize=14, weight="bold")
    ax.text(0.5, 9.9, "1. global atomics", fontsize=11.5, weight="bold", color=RED)
    for b in range(4):
        cell(ax, 2.0 + b * 2.4, 1.6, f"bin {b}", RED_L, RED, w=2.4, h=1.0)
    for t in range(12):
        x = 0.6 + t * 1.05
        cell(ax, x, 8.2, "", BLUE_L, BLUE, w=0.9, h=0.7)
        arrow(ax, (x + 0.45, 8.2), (2.0 + (t * 7 % 4) * 2.4 + 1.1, 2.6), RED, lw=0.8)
    ax.text(0.5, 0.6, "every element → atomicAdd in global memory: hot bins serialize", fontsize=9.5, color=RED)
    x0 = 15.5
    ax.text(x0, 9.9, "2. per-block shared histograms, then merge", fontsize=11.5, weight="bold", color=TEAL)
    for blk in range(2):
        bx = x0 + blk * 7.0
        box(ax, bx, 4.6, 6.4, 4.4, TEAL_L, TEAL, lw=1.2)
        ax.text(bx + 0.2, 8.5, f"block {blk} (shared memory)", fontsize=9, color=TEAL)
        for t in range(5):
            cell(ax, bx + 0.3 + t * 1.2, 7.3, "", BLUE_L, BLUE, w=1.0, h=0.6)
        for b in range(4):
            cell(ax, bx + 0.3 + b * 1.5, 5.0, "", "white", TEAL, w=1.5, h=0.8)
            arrow(ax, (bx + 1.05 + b * 1.5, 5.0), (x0 + 2.0 + b * 2.4 + 1.1, 2.6), TEAL, lw=0.9)
        for t in range(5):
            arrow(ax, (bx + 0.8 + t * 1.2, 7.3), (bx + 1.05 + (t * 3 % 4) * 1.5, 5.8), BLUE, lw=0.7)
    for b in range(4):
        cell(ax, x0 + 2.0 + b * 2.4, 1.6, f"bin {b}", AMBER_L, AMBER, w=2.4, h=1.0)
    ax.text(x0, 0.6, "fast shared atomics on the SM; global atomics = bins × blocks, not n", fontsize=9.5, color=TEAL)
    save(fig, "histogram.png")


# ---------------------------------------------------------------- 5. convolution

def convolution():
    fig, ax = canvas(16, 8.4, (0, 32), (0, 16.8))
    ax.text(0.3, 16.1, "1-D convolution: where the inputs and the mask come from", fontsize=14, weight="bold")
    n = 16
    # naive
    ax.text(0.5, 14.4, "1-2. naive / constant mask: each output reads m neighbours", fontsize=11, weight="bold", color=BLUE)
    for i in range(n):
        cell(ax, 0.5 + i * 0.9, 12.6, "", BLUE_L if 5 <= i <= 9 else GRAY_L, BLUE if 5 <= i <= 9 else GRAY, w=0.9, h=0.8)
    for j in range(5):
        cell(ax, 0.5 + (5 + j) * 0.9, 11.2, f"m{j}", AMBER_L, AMBER, w=0.9, h=0.7, fs=7)
    cell(ax, 0.5 + 7 * 0.9, 9.8, "y7", TEAL_L, TEAL, w=0.9, h=0.8, fs=8)
    for j in range(5):
        arrow(ax, (0.5 + (5 + j) * 0.9 + 0.42, 11.2), (0.5 + 7 * 0.9 + 0.42, 10.6), BLUE, lw=0.8)
    ax.text(16.0, 13.0, "x (global memory)", fontsize=9.5, color=BLUE)
    ax.text(16.0, 11.5, "mask: global (step 1) or __constant__ (step 2):", fontsize=9.5, color=AMBER)
    ax.text(16.0, 10.9, "all lanes read m[j] together → one broadcast", fontsize=9.5, color=AMBER)
    # tiled
    ax.text(0.5, 8.2, "3. tiled: block inputs + halo staged in shared memory", fontsize=11, weight="bold", color=TEAL)
    for i in range(n + 4):
        halo = i < 2 or i >= n + 2
        cell(ax, 0.5 + i * 0.9, 6.6, "", PURPLE_L if halo else TEAL_L, PURPLE if halo else TEAL, w=0.9, h=0.8)
    ax.text(0.5, 5.8, "halo (r each side)", fontsize=9, color=PURPLE)
    ax.text(0.5 + (n + 4) * 0.9 + 0.5, 7.0, "every read after the load hits shared memory", fontsize=9.5)
    ax.text(0.5, 4.6, "4. cache: only the block's own inputs in shared memory; halo reads go to global memory", fontsize=11,
            weight="bold", color=AMBER)
    for i in range(n):
        cell(ax, 0.5 + 2 * 0.9 + i * 0.9, 3.0, "", TEAL_L, TEAL, w=0.9, h=0.8)
    for i in (0, 1, n + 2, n + 3):
        cell(ax, 0.5 + i * 0.9, 3.0, "L1", "white", AMBER, w=0.9, h=0.8, fs=7)
    ax.text(0.5 + (n + 4) * 0.9 + 0.5, 3.4, "the halo was just loaded by the neighbouring block,", fontsize=9.5)
    ax.text(0.5 + (n + 4) * 0.9 + 0.5, 2.8, "so it is usually an L1/L2 hit: simpler loads", fontsize=9.5)
    ax.text(0.5, 1.2, "5. 2-D: the same with an m × m mask in constant memory over an N × N image.", fontsize=10)
    save(fig, "convolution.png")


if __name__ == "__main__":
    vadd_paths()
    matmul_steps()
    overhang()
    reduction()
    histogram()
    convolution()
