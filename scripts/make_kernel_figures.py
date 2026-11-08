"""Generate the figures for the classic kernels pages.

Run from the repo root:

    .venv/bin/python scripts/make_kernel_figures.py

Writes PNG files into docs/assets/img/kernels/. Reuses the palette and helpers from
make_figures.py so all figures look the same.
"""
import math

from make_figures import (AMBER, AMBER_L, BLUE, BLUE_L, GRAY, GRAY_L, INK, MONO, OUT, PURPLE,
                          PURPLE_L, RED, RED_L, TEAL, TEAL_L, box, canvas, plt)

DIR = OUT / "kernels"
DIR.mkdir(parents=True, exist_ok=True)


def save(fig, name):
    fig.savefig(DIR / name, dpi=110, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print("wrote kernels/" + name)


def cell(ax, x, y, text, fc, ec, w=1.0, h=0.8, fs=9, lw=1.1):
    box(ax, x, y, w * 0.94, h, fc, ec, lw=lw)
    if text != "":
        ax.text(x + w * 0.47, y + h / 2, str(text), ha="center", va="center", fontsize=fs, family=MONO)


def arrow(ax, a, b, color=INK, lw=1.4, rad=0.0):
    ax.annotate("", xy=b, xytext=a, arrowprops=dict(arrowstyle="-|>", color=color, lw=lw, mutation_scale=12,
                                                    connectionstyle=f"arc3,rad={rad}"))


def title(ax, text, y):
    ax.text(0.3, y, text, fontsize=13.5, weight="bold")


# 1. fusion -------------------------------------------------------------------------------

def fusion():
    fig, ax = canvas(14, 5.0, (0, 28), (0, 10))
    title(ax, "Fusion: bias + GELU in one pass", 9.3)
    ax.text(0.5, 7.9, "unfused: 2 kernels, the intermediate goes through DRAM", fontsize=10.5, weight="bold", color=RED)
    steps = [("x", GRAY_L, GRAY), ("+ bias", BLUE_L, BLUE), ("t (DRAM)", RED_L, RED), ("GELU", BLUE_L, BLUE), ("y", TEAL_L, TEAL)]
    for k, (t, fc, ec) in enumerate(steps):
        cell(ax, 0.5 + k * 3.0, 6.2, t, fc, ec, w=2.6, h=1.0, fs=9.5)
        if k:
            arrow(ax, (0.5 + k * 3.0 - 0.5, 6.7), (0.5 + k * 3.0, 6.7), INK)
    ax.text(0.5, 5.3, "bytes: read x, write t, read t, write y = 4 passes over the data", fontsize=9.5, color=RED)
    ax.text(0.5, 3.9, "fused: 1 kernel, the intermediate lives in a register", fontsize=10.5, weight="bold", color=TEAL)
    steps = [("x", GRAY_L, GRAY), ("+ bias → GELU\n(registers)", TEAL_L, TEAL), ("y", TEAL_L, TEAL)]
    for k, (t, fc, ec) in enumerate(steps):
        cell(ax, 0.5 + k * 3.8, 2.0, t, fc, ec, w=3.4, h=1.3, fs=9.5)
        if k:
            arrow(ax, (0.5 + k * 3.8 - 0.5, 2.65), (0.5 + k * 3.8, 2.65), INK)
    ax.text(0.5, 1.0, "bytes: read x, write y = 2 passes → 2× faster for a memory-bound op", fontsize=9.5, color=TEAL)
    ax.text(17.0, 7.0, "SwiGLU, same idea:", fontsize=10.5, weight="bold", color=PURPLE)
    for k, t in enumerate(["gate", "up"]):
        cell(ax, 17.0 + k * 4.4, 5.6, t, PURPLE_L, PURPLE, w=4.2, h=0.9)
    ax.text(17.0, 4.6, "y = SiLU(gate) · up", fontsize=11, family=MONO)
    ax.text(17.0, 3.8, "one read of each half, one write of y", fontsize=9.5, color=GRAY)
    save(fig, "fusion.png")


# 2. RoPE -----------------------------------------------------------------------------------

def rope():
    fig, ax = canvas(14, 5.6, (0, 28), (0, 11.2))
    title(ax, "RoPE: rotate pairs (i, i + D/2) by angle pos · θᵢ", 10.5)
    D = 8
    for i in range(D):
        c = [BLUE_L, TEAL_L, AMBER_L, PURPLE_L][i % 4]
        e = [BLUE, TEAL, AMBER, PURPLE][i % 4]
        cell(ax, 0.5 + i * 1.3, 7.6, f"x{i}", c, e, w=1.3)
    for i in range(D // 2):
        ax.annotate("", xy=(0.5 + (i + D // 2) * 1.3 + 0.6, 8.4), xytext=(0.5 + i * 1.3 + 0.6, 8.4),
                    arrowprops=dict(arrowstyle="<->", color=[BLUE, TEAL, AMBER, PURPLE][i], lw=1.2,
                                    connectionstyle="arc3,rad=-0.35"))
    ax.text(0.5, 6.6, "one head's vector, D = 8: four pairs, four frequencies θᵢ = base^(−2i/D)", fontsize=9.5)
    ax.text(0.5, 5.4, "x1' = x1·cos − x2·sin", fontsize=11, family=MONO)
    ax.text(0.5, 4.6, "x2' = x2·cos + x1·sin", fontsize=11, family=MONO)
    ax.text(0.5, 3.4, "θ₀ is fast (spins every token), θ₃ is slow: different", fontsize=9.5, color=GRAY)
    ax.text(0.5, 2.8, "pairs encode position at different scales", fontsize=9.5, color=GRAY)
    cx, cy, r = 20.0, 5.2, 3.2
    ax.add_patch(plt.Circle((cx, cy), r, fill=False, color=GRAY, lw=1))
    for pos, c in [(0, GRAY), (1, BLUE), (2, TEAL), (3, AMBER)]:
        a = 0.35 + pos * 0.55
        ax.annotate("", xy=(cx + r * math.cos(a), cy + r * math.sin(a)), xytext=(cx, cy),
                    arrowprops=dict(arrowstyle="-|>", color=c, lw=1.8))
        ax.text(cx + (r + 0.6) * math.cos(a), cy + (r + 0.6) * math.sin(a), f"pos {pos}", fontsize=9, color=c,
                ha="center", va="center")
    ax.text(cx, cy - r - 1.0, "the pair (x1, x2) as a 2-D vector, rotated by pos · θ", fontsize=9.5, ha="center")
    save(fig, "rope.png")


# 3. online logsumexp -------------------------------------------------------------------------

def logsumexp():
    fig, ax = canvas(14, 4.8, (0, 28), (0, 9.6))
    title(ax, "Fused cross-entropy: one pass with a running (max, sum)", 8.9)
    vals = [1.0, 3.0, 0.5, 4.0, 2.0]
    m, s = -math.inf, 0.0
    for k, v in enumerate(vals):
        x = 0.5 + k * 5.3
        cell(ax, x, 6.2, f"logit {v}", GRAY_L, GRAY, w=3.0, h=0.9)
        if v > m:
            s = (s * math.exp(m - v) if m != -math.inf else 0.0) + 1.0
            m = v
            c, note = RED, "new max: rescale s"
        else:
            s += math.exp(v - m)
            c, note = TEAL, "add exp(v − m)"
        cell(ax, x, 4.2, f"m={m:g}", c + "33", c, w=3.0, h=0.8)
        cell(ax, x, 3.2, f"s={s:.3f}", c + "33", c, w=3.0, h=0.8)
        ax.text(x, 2.4, note, fontsize=8.5, color=c)
        arrow(ax, (x + 1.4, 6.2), (x + 1.4, 5.0), GRAY)
    ax.text(0.5, 1.0, f"loss = m + log(s) − logit[target] = {m + math.log(s):.3f} − logit[target]   "
            "(no N × V softmax ever written; warps merge (m, s) pairs at the end)", fontsize=9.5)
    save(fig, "logsumexp.png")


# 4. top-k ------------------------------------------------------------------------------------

def topk():
    fig, ax = canvas(14, 4.8, (0, 28), (0, 9.6))
    title(ax, "Top-k: k rounds of block argmax over a row in shared memory (k = 3)", 8.9)
    row = [0.2, 0.9, 0.4, 0.7, 0.1, 0.95, 0.3, 0.8]
    alive = list(row)
    for r in range(3):
        y = 6.3 - r * 2.0
        best = max(range(8), key=lambda j: alive[j])
        for j, v in enumerate(alive):
            if v == -1:
                cell(ax, 0.5 + j * 1.5, y, "−∞", GRAY_L, GRAY, w=1.5)
            else:
                cell(ax, 0.5 + j * 1.5, y, v, TEAL if j == best else "white", TEAL if j == best else GRAY, w=1.5)
        ax.text(13.0, y + 0.4, f"round {r + 1}: max = {alive[best]} at index {best}  → record, set to −∞", fontsize=9.5)
        alive[best] = -1
    ax.text(0.5, 0.5, "one DRAM read of the row, then k cheap passes over shared memory; each argmax is "
            "warp shuffles + one cross-warp step", fontsize=9.5)
    save(fig, "topk.png")


# 5. bitonic ----------------------------------------------------------------------------------

def bitonic():
    fig, ax = canvas(14, 6.4, (0, 28), (0, 12.8))
    title(ax, "Bitonic sort network, n = 8: 6 stages, each one fully parallel", 12.1)
    n = 8
    stages = [(k, j) for k in (2, 4, 8) for j in [kk for kk in (4, 2, 1) if kk < k]]
    ys = {i: 10.4 - i * 1.2 for i in range(n)}
    for i in range(n):
        ax.plot([1.0, 26.5], [ys[i], ys[i]], color=GRAY_L, lw=2, zorder=0)
        ax.text(0.6, ys[i], str(i), va="center", fontsize=9, family=MONO)
    for s_, (k, j) in enumerate(stages):
        x = 2.5 + s_ * 4.0
        ax.text(x, 11.3, f"k={k} j={j}", fontsize=9, ha="center", family=MONO)
        for i in range(n):
            l = i ^ j
            if l > i:
                asc = (i & k) == 0
                c = BLUE if asc else RED
                xx = x + (i % j) * 0.55 - (j - 1) * 0.275      # spread overlapping pairs apart
                ax.plot([xx, xx], [ys[i], ys[l]], color=c, lw=2)
                ax.plot([xx, xx], [ys[i], ys[l]], "o", color=c, ms=5)
                ax.annotate("", xy=(xx, ys[l] if asc else ys[i]), xytext=(xx, (ys[i] + ys[l]) / 2),
                            arrowprops=dict(arrowstyle="-|>", color=c, lw=1.4))
    ax.text(0.6, 0.5, "each vertical line is one compare-exchange (arrow: where the larger value goes; blue = ascending, "
            "red = descending). Partner = i XOR j.", fontsize=9.5)
    save(fig, "bitonic.png")


# 6. radix ------------------------------------------------------------------------------------

def radix():
    fig, ax = canvas(15, 6.4, (0, 30), (0, 12.8))
    title(ax, "Radix sort pass: count → scan → scatter (2 blocks, 2-bit digits shown)", 12.1)
    keys = [[2, 0, 3, 2], [1, 2, 0, 2]]
    cols = [GRAY_L, BLUE_L, TEAL_L, AMBER_L]
    edge = [GRAY, BLUE, TEAL, AMBER]
    for b, ks in enumerate(keys):
        for j, d in enumerate(ks):
            cell(ax, 0.5 + b * 5.0 + j * 1.2, 9.6, d, cols[d], edge[d], w=1.2)
        ax.text(0.5 + b * 5.0, 10.7, f"block {b}", fontsize=9.5)
    ax.text(10.8, 10.0, "keys (their current digit shown)", fontsize=10, color=GRAY)
    counts = [[sum(1 for d in ks if d == dd) for ks in keys] for dd in range(4)]
    flat = [counts[d][b] for d in range(4) for b in range(2)]
    for k, v in enumerate(flat):
        d, b = divmod(k, 2)
        cell(ax, 0.5 + k * 1.3, 7.0, v, cols[d], edge[d], w=1.3)
        ax.text(0.5 + k * 1.3 + 0.6, 6.5, f"d{d}b{b}", fontsize=7, ha="center", color=GRAY)
    ax.text(11.5, 7.3, "1 · count each digit per block, laid out [digit][block]", fontsize=10, weight="bold")
    scan, acc = [], 0
    for v in flat:
        scan.append(acc)
        acc += v
    for k, v in enumerate(scan):
        d, b = divmod(k, 2)
        cell(ax, 0.5 + k * 1.3, 4.6, v, cols[d], edge[d], w=1.3)
    out = sorted([(d, b, j) for b, ks in enumerate(keys) for j, d in enumerate(ks)])
    ax.text(11.5, 4.9, "2 · exclusive scan → the first output slot of each (digit, block)", fontsize=10, weight="bold")
    for k, (d, b, j) in enumerate(out):
        cell(ax, 0.5 + k * 1.3, 1.8, d, cols[d], edge[d], w=1.3)
        ax.text(0.5 + k * 1.3 + 0.6, 1.3, f"b{b}", fontsize=7, ha="center", color=GRAY)
    ax.text(11.5, 2.4, "3 · scatter: slot + rank among equal digits earlier in the block", fontsize=10, weight="bold")
    ax.text(11.5, 1.7, "stable: equal digits keep block order, then within-block order", fontsize=9.5, color=TEAL)
    ax.text(0.5, 0.3, "Repeat for every digit, lowest first (8 passes of 4 bits for 32-bit keys).", fontsize=9.5)
    save(fig, "radix.png")


# 7. Philox -----------------------------------------------------------------------------------

def philox():
    fig, ax = canvas(14, 4.6, (0, 28), (0, 9.2))
    title(ax, "Counter-based RNG: random number = f(seed, element index)", 8.5)
    cell(ax, 0.5, 5.6, "seed", AMBER_L, AMBER, w=3.0, h=1.0)
    cell(ax, 0.5, 4.0, "counter = i / 4", BLUE_L, BLUE, w=3.0, h=1.0)
    box(ax, 5.0, 3.6, 6.0, 3.4, PURPLE_L, PURPLE, lw=1.6)
    ax.text(8.0, 5.6, "Philox4x32-10", ha="center", fontsize=11, weight="bold", color=PURPLE)
    ax.text(8.0, 4.6, "10 rounds of\nmultiply-hi/lo + xor", ha="center", fontsize=9)
    arrow(ax, (3.3, 6.1), (5.0, 5.6))
    arrow(ax, (3.3, 4.5), (5.0, 4.8))
    for q in range(4):
        cell(ax, 12.8 + q * 1.8, 4.9, f"r{q}", TEAL_L, TEAL, w=1.8)
    arrow(ax, (11.0, 5.3), (12.8, 5.3))
    ax.text(12.8, 4.0, "4 uniforms → 4 elements", fontsize=9.5)
    ax.text(20.5, 6.0, "• no state: any thread, any order", fontsize=9.5)
    ax.text(20.5, 5.2, "• same (seed, i) → same mask:", fontsize=9.5)
    ax.text(20.5, 4.6, "   backward regenerates it", fontsize=9.5)
    ax.text(20.5, 3.8, "• launch shape doesn't matter", fontsize=9.5)
    ax.text(0.5, 1.5, "keep = (r >> 8) · 2⁻²⁴ ≥ p,   y = keep ? x / (1 − p) : 0", fontsize=11, family=MONO)
    save(fig, "philox.png")


# 8. int8 -------------------------------------------------------------------------------------

def int8():
    fig, ax = canvas(14, 4.8, (0, 28), (0, 9.6))
    title(ax, "Per-row int8 quantization: absmax → scale → round", 8.9)
    x = [0.31, -1.27, 0.05, 0.88, -0.44]
    amax = max(abs(v) for v in x)
    for j, v in enumerate(x):
        cell(ax, 0.5 + j * 2.2, 6.4, v, BLUE_L if abs(v) != amax else BLUE, BLUE, w=2.2)
    ax.text(12.0, 6.7, f"max |x| = {amax}  →  scale = {amax}/127 = {amax / 127:.4f}", fontsize=10)
    for j, v in enumerate(x):
        q = max(-127, min(127, round(v * 127 / amax)))
        cell(ax, 0.5 + j * 2.2, 4.2, q, TEAL_L, TEAL, w=2.2)
        arrow(ax, (0.5 + j * 2.2 + 1.0, 6.4), (0.5 + j * 2.2 + 1.0, 5.0), GRAY)
    ax.text(12.0, 4.5, "q = round(x · 127 / max)   int8, 4× smaller than FP32", fontsize=10)
    for j, v in enumerate(x):
        q = max(-127, min(127, round(v * 127 / amax)))
        cell(ax, 0.5 + j * 2.2, 2.0, f"{q * amax / 127:.3f}", AMBER_L, AMBER, w=2.2, fs=8)
        arrow(ax, (0.5 + j * 2.2 + 1.0, 4.2), (0.5 + j * 2.2 + 1.0, 2.8), GRAY)
    ax.text(12.0, 2.3, "x' = q · scale   error ≤ scale / 2", fontsize=10)
    ax.text(0.5, 0.6, "One outlier sets the scale for the whole row: that is why LLM quantizers use small groups "
            "(e.g. 128 values) or keep outliers in FP16.", fontsize=9.5, color=GRAY)
    save(fig, "int8.png")


# 9. embedding bag ----------------------------------------------------------------------------

def embedding_bag():
    fig, ax = canvas(14, 5.6, (0, 28), (0, 11.2))
    title(ax, "Embedding bag: gather rows, reduce per bag (offsets = CSR row pointers)", 10.5)
    for r in range(6):
        for d in range(4):
            cell(ax, 0.5 + d * 0.9, 8.4 - r * 0.9, "", BLUE_L if r in (1, 4) else (TEAL_L if r == 2 else GRAY_L),
                 BLUE if r in (1, 4) else (TEAL if r == 2 else GRAY), w=0.9, h=0.8)
        ax.text(4.4, 8.8 - r * 0.9, f"row {r}", fontsize=8.5, va="center")
    ax.text(0.5, 9.6, "table V × D", fontsize=10, weight="bold")
    ax.text(8.0, 9.6, "indices", fontsize=10, weight="bold")
    for j, v in enumerate([1, 4, 2]):
        cell(ax, 8.0 + j * 1.2, 8.4, v, BLUE_L if v != 2 else TEAL_L, BLUE if v != 2 else TEAL, w=1.2)
    ax.text(8.0, 3.6, "offsets [0, 2, 3]: bag 0 = indices[0:2] = {1, 4},  bag 1 = indices[2:3] = {2}", fontsize=9.5)
    ax.text(16.0, 9.6, "out B × D", fontsize=10, weight="bold")
    for b, (c, e) in enumerate([(BLUE, BLUE_L), (TEAL, TEAL_L)]):
        for d in range(4):
            cell(ax, 16.0 + d * 0.9, 8.4 - b * 1.2, "", e, c, w=0.9, h=0.8)
        ax.text(19.9, 8.8 - b * 1.2, f"bag {b} = Σ rows", fontsize=9, va="center", color=c)
    arrow(ax, (4.0, 7.5), (16.0, 8.7), BLUE, rad=-0.1)
    arrow(ax, (4.0, 4.8), (16.0, 8.5), BLUE, rad=0.1)
    arrow(ax, (4.0, 6.6), (16.0, 7.3), TEAL)
    ax.text(0.5, 2.0, "one block per bag; threads across D, so each gathered row is one coalesced read.", fontsize=9.5)
    ax.text(0.5, 1.2, "Bound by the random row reads (table ≫ L2). The backward pass is a scatter-add into the "
            "table gradient.", fontsize=9.5, color=GRAY)
    save(fig, "embedding-bag.png")


# 10. max pooling -----------------------------------------------------------------------------

def maxpool():
    fig, ax = canvas(14, 5.6, (0, 28), (0, 11.2))
    title(ax, "2-D max pooling, k = 2, s = 2: one thread per output", 10.5)
    vals = [[1, 3, 2, 0], [5, 2, 1, 4], [0, 6, 3, 2], [2, 1, 7, 1]]
    cols = [[BLUE_L, BLUE_L, TEAL_L, TEAL_L], [BLUE_L, BLUE_L, TEAL_L, TEAL_L],
            [AMBER_L, AMBER_L, PURPLE_L, PURPLE_L], [AMBER_L, AMBER_L, PURPLE_L, PURPLE_L]]
    for r in range(4):
        for c in range(4):
            win = (r // 2, c // 2)
            block = [vals[rr][cc] for rr in range(win[0] * 2, win[0] * 2 + 2) for cc in range(win[1] * 2, win[1] * 2 + 2)]
            is_max = vals[r][c] == max(block)
            cell(ax, 0.5 + c * 1.6, 7.6 - r * 1.6, vals[r][c], cols[r][c], INK if is_max else GRAY, w=1.6, h=1.4,
                 fs=11, lw=2.4 if is_max else 1.0)
    ax.text(0.5, 9.4, "input 4 × 4", fontsize=10, weight="bold")
    out = [[5, 4], [6, 7]]
    idx = [[4, 7], [9, 14]]
    for r in range(2):
        for c in range(2):
            cell(ax, 10.5 + c * 1.8, 6.0 - r * 1.8, out[r][c], [[BLUE_L, TEAL_L], [AMBER_L, PURPLE_L]][r][c], INK,
                 w=1.8, h=1.5, fs=11)
            cell(ax, 16.5 + c * 1.8, 6.0 - r * 1.8, idx[r][c], "white", GRAY, w=1.8, h=1.5, fs=10)
    ax.text(10.5, 8.0, "output", fontsize=10, weight="bold")
    ax.text(16.5, 8.0, "argmax (h·W + w)", fontsize=10, weight="bold")
    arrow(ax, (7.2, 5.2), (10.4, 5.2), GRAY)
    ax.text(0.5, 1.0, "The argmax indices let the backward pass route each gradient to the one input that won. "
            "With overlapping windows (s < k) several outputs can pick the same input → atomicAdd.", fontsize=9.5)
    save(fig, "maxpool.png")


if __name__ == "__main__":
    fusion()
    rope()
    logsumexp()
    topk()
    bitonic()
    radix()
    philox()
    int8()
    embedding_bag()
    maxpool()
