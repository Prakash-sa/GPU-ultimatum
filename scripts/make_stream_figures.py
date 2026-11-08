"""Generate the figures for the streams practice problems (S1-S4) in the LeetGPU section.

Run from the repo root:

    .venv/bin/python scripts/make_stream_figures.py

Writes PNG files into docs/assets/img/streams/. Reuses the palette and helpers from
make_figures.py so all figures look the same.
"""
from make_figures import (AMBER, AMBER_L, BLUE, BLUE_L, GRAY, GRAY_L, INK, MONO, OUT, PURPLE,
                          PURPLE_L, RED, RED_L, TEAL, TEAL_L, box, canvas, plt)

DIR = OUT / "streams"
DIR.mkdir(parents=True, exist_ok=True)


def save(fig, name):
    fig.savefig(DIR / name, dpi=110, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print("wrote streams/" + name)


def cell(ax, x, y, w, h, text, fc, ec, fs=9):
    box(ax, x, y, w, h, fc, ec, lw=1.1)
    if text:
        ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=fs, family=MONO)


def arrow(ax, a, b, color=INK, lw=1.4, rad=0.0, ls="-"):
    ax.annotate("", xy=b, xytext=a, arrowprops=dict(arrowstyle="-|>", color=color, lw=lw, ls=ls,
                                                    mutation_scale=12, connectionstyle=f"arc3,rad={rad}"))


def lane_labels(ax, ys, names, x=0.3):
    for y, n in zip(ys, names):
        ax.text(x, y + 0.4, n, fontsize=9.5, va="center", weight="bold", color=INK)


H2D, KER, D2H = (BLUE_L, BLUE), (TEAL_L, TEAL), (AMBER_L, AMBER)


# S1: copy / compute pipeline ------------------------------------------------------------

def pipeline():
    fig, ax = canvas(14, 6.6, (0, 30), (0, 14))
    ax.text(0.3, 13.2, "S1. One stream vs chunks on two streams", fontsize=13.5, weight="bold")
    x0, u = 5.0, 1.6                        # time origin, width of one chunk-stage

    ax.text(0.3, 11.7, "1 stream, 1 chunk: each engine waits for the previous one", fontsize=10, color=RED)
    ys = [10.3, 9.2, 8.1]
    lane_labels(ax, ys, ["H2D copy", "SMs", "D2H copy"])
    for k, (fc, ec), lab in [(0, H2D, "H2D all"), (1, KER, "kernel all"), (2, D2H, "D2H all")]:
        cell(ax, x0 + k * 4 * u, ys[k], 4 * u, 0.8, lab, fc, ec)
    ax.annotate("", xy=(x0 + 12 * u, 7.6), xytext=(x0, 7.6), arrowprops=dict(arrowstyle="<->", color=RED))
    ax.text(x0 + 6 * u, 7.1, "total = 12 units", ha="center", fontsize=9.5, color=RED)

    ax.text(0.3, 6.0, "4 chunks, round-robin over streams A and B: the three engines overlap", fontsize=10, color=TEAL)
    ys = [4.6, 3.5, 2.4]
    lane_labels(ax, ys, ["H2D copy", "SMs", "D2H copy"])
    for c in range(4):
        st = "AB"[c % 2]
        for k, (fc, ec) in enumerate([H2D, KER, D2H]):
            cell(ax, x0 + (c + k) * u, ys[k], u, 0.8, f"{st}{c}", fc, ec, fs=8.5)
    ax.annotate("", xy=(x0 + 6 * u, 1.9), xytext=(x0, 1.9), arrowprops=dict(arrowstyle="<->", color=TEAL))
    ax.text(x0 + 3 * u, 1.4, "total = 6 units: 2× faster", ha="center", fontsize=9.5, color=TEAL)

    tx = x0 + 13 * u
    ax.text(tx, 5.6, "Needed for the overlap:", fontsize=10, weight="bold")
    for i, t in enumerate(["• pinned host memory (cudaMallocHost)",
                           "• cudaMemcpyAsync, not cudaMemcpy",
                           "• a stream per in-flight chunk",
                           "• 2 copy engines: H2D + D2H",
                           "best case: max(copy, compute),",
                           "not copy + compute"]):
        ax.text(tx, 4.8 - 0.75 * i, t, fontsize=9.2, family=MONO if i < 4 else None,
                color=INK if i < 4 else PURPLE)
    save(fig, "pipeline.png")


# S2: fork-join DAG -----------------------------------------------------------------------

def fork_join():
    fig, ax = canvas(14, 5.6, (0, 30), (0, 12))
    ax.text(0.3, 11.2, "S2. Fork-join: two independent reductions, then one kernel that needs both",
            fontsize=13.5, weight="bold")
    ys = {"0": 5.2, "a": 8.3, "b": 2.1}
    for k, n in [("a", "stream a"), ("0", "stream 0"), ("b", "stream b")]:
        ax.text(0.3, ys[k] + 0.45, n, fontsize=10, weight="bold", va="center")
        ax.plot([3.0, 29.5], [ys[k] + 0.45] * 2, color=GRAY_L, lw=1, zorder=0)

    cell(ax, 3.4, ys["0"], 3.4, 0.9, "x ready", GRAY_L, GRAY)
    ax.plot(7.6, ys["0"] + 0.45, "D", color=PURPLE, ms=9)
    ax.text(7.6, ys["0"] - 0.4, "record\nready", ha="center", va="top", fontsize=8.5, color=PURPLE)
    for k in "ab":
        arrow(ax, (7.7, ys["0"] + 0.45), (9.4, ys[k] + 0.45), PURPLE)
    ax.text(8.0, 7.3, "wait", fontsize=8.5, color=PURPLE)

    cell(ax, 9.6, ys["a"], 2.6, 0.9, "memset", GRAY_L, GRAY, fs=8.5)
    cell(ax, 12.4, ys["a"], 5.0, 0.9, "sum_kernel", TEAL_L, TEAL)
    cell(ax, 9.6, ys["b"], 2.6, 0.9, "memset", GRAY_L, GRAY, fs=8.5)
    cell(ax, 12.4, ys["b"], 6.6, 0.9, "absmax_kernel", BLUE_L, BLUE)
    for k, x, lab in [("a", 17.9, "sum_done"), ("b", 19.5, "max_done")]:
        ax.plot(x, ys[k] + 0.45, "D", color=PURPLE, ms=9)
        ax.text(x, ys[k] + 1.15 if k == "a" else ys[k] - 0.55, lab, ha="center", fontsize=8.5, color=PURPLE)
        arrow(ax, (x + 0.1, ys[k] + 0.45), (21.3, ys["0"] + 0.45), PURPLE)
    ax.text(20.3, 6.6, "join: cudaStreamWaitEvent(0, …)", fontsize=8.5, color=PURPLE)
    cell(ax, 21.5, ys["0"], 5.2, 0.9, "normalize", AMBER_L, AMBER)

    ax.text(3.4, 0.6, "a and b run side by side; normalize starts only after BOTH events. "
            "Drop one wait and it can read a half-finished sum.", fontsize=9.5, color=RED)
    save(fig, "fork-join.png")


# S3: halo overlap ------------------------------------------------------------------------

def halo():
    fig, ax = canvas(14, 7.6, (0, 30), (0, 16.2))
    ax.text(0.3, 15.5, "S3. Two halves with ghost cells; the halo exchange hides behind the interior",
            fontsize=13.5, weight="bold")
    # the two halves
    w = 0.9
    for k, (x0, fc, ec, name) in enumerate([(1.0, TEAL_L, TEAL, "half 0 (GPU 0)"), (15.5, BLUE_L, BLUE, "half 1 (GPU 1)")]):
        ax.text(x0, 14.0, name, fontsize=10, weight="bold", color=ec)
        for j in range(12):
            ghost = j in (0, 11)
            edge = j in (1, 10)
            f, e = (GRAY_L, GRAY) if ghost else ((AMBER_L, AMBER) if edge else (fc, ec))
            box(ax, x0 + j * w, 12.4, w * 0.92, 0.9, f, e, lw=1.0, ls="--" if ghost else "-")
        ax.text(x0 + 0.45, 12.1, "ghost", ha="center", va="top", fontsize=7.5, color=GRAY)
        ax.text(x0 + 11 * w + 0.45, 12.1, "ghost", ha="center", va="top", fontsize=7.5, color=GRAY)
    arrow(ax, (1.0 + 10 * w + 0.45, 13.4), (15.5 + 0.45, 13.4), AMBER, rad=-0.25)
    arrow(ax, (15.5 + 1 * w + 0.45, 12.35), (1.0 + 11 * w + 0.45, 12.35), AMBER, rad=-0.25)
    ax.text(12.6, 14.4, "halo exchange", fontsize=9, color=AMBER, ha="center")
    ax.text(1.0, 11.2, "amber = edge cells (need the neighbour's ghost)   colour = interior (only local data)",
            fontsize=9, color=INK)

    # timeline over two steps
    yc, yi = 8.0, 5.6
    ax.text(0.3, yc + 0.45, "comm", fontsize=10, weight="bold", va="center")
    ax.text(0.3, yi + 0.45, "compute", fontsize=10, weight="bold", va="center")
    for t, x0 in enumerate([4.0, 16.5]):
        ax.text(x0, 9.6, f"step {t}", fontsize=10, weight="bold")
        cell(ax, x0, yc, 2.6, 0.9, "edges", AMBER_L, AMBER, fs=8.5)
        cell(ax, x0 + 2.7, yc, 3.2, 0.9, "halo copy", AMBER_L, AMBER, fs=8.5)
        cell(ax, x0, yi, 9.0, 0.9, "interior cells (most of the work)", TEAL_L, TEAL, fs=8.5)
        ax.plot(x0 + 6.1, yc + 0.45, "D", color=PURPLE, ms=8)
        ax.plot(x0 + 9.2, yi + 0.45, "D", color=PURPLE, ms=8)
    arrow(ax, (10.2, yc + 0.45), (16.4, yi + 0.6), PURPLE, rad=0.0)
    arrow(ax, (13.3, yi + 0.45), (16.4, yc + 0.3), PURPLE, rad=0.0)
    ax.text(14.6, 7.35, "each stream waits for\nthe other's last step", fontsize=8.5, color=PURPLE, ha="center")

    for i, t in enumerate(["comm waits compute_done: edges read interior cells of the last step",
                           "compute waits comm_done: interior reads the edge cells of the last step",
                           "the exchange costs nothing extra when interior time > edges + copy"]):
        ax.text(1.0, 3.6 - 0.9 * i, t, fontsize=9.5, color=PURPLE if i < 2 else TEAL)
    save(fig, "halo-overlap.png")


# S4: CUDA graph ---------------------------------------------------------------------------

def graph():
    fig, ax = canvas(14, 6.6, (0, 30), (0, 14))
    ax.text(0.3, 13.2, "S4. Capture one step, replay it: one launch per step instead of one per kernel",
            fontsize=13.5, weight="bold")
    ax.text(0.3, 11.6, "plain launches", fontsize=10, weight="bold", color=RED)
    ax.text(0.3, 10.9, "CPU", fontsize=9)
    ax.text(0.3, 9.8, "GPU", fontsize=9)
    x = 3.0
    for t in range(3):
        for k, (lab, fc, ec, gw) in enumerate([("heat", TEAL_L, TEAL, 0.9), ("flip", PURPLE_L, PURPLE, 0.4)]):
            cell(ax, x, 10.7, 1.5, 0.6, "launch", RED_L, RED, fs=7.5)
            cell(ax, x + 1.5, 9.6, gw, 0.6, "", fc, ec)
            x += 1.5 + gw + 0.1
    ax.text(x + 0.3, 9.9, "GPU idles while the CPU\nlaunches each tiny kernel", fontsize=9, color=RED, va="center")

    ax.text(0.3, 8.1, "graph replay", fontsize=10, weight="bold", color=TEAL)
    ax.text(0.3, 7.4, "CPU", fontsize=9)
    ax.text(0.3, 6.3, "GPU", fontsize=9)
    x = 3.0
    for t in range(3):
        cell(ax, x, 7.2, 1.3, 0.6, "graph", TEAL_L, TEAL, fs=7.5)
        x += 1.4
    gx = 4.3
    for t in range(3):
        cell(ax, gx, 6.1, 0.9, 0.6, "", TEAL_L, TEAL)
        cell(ax, gx + 0.95, 6.1, 0.4, 0.6, "", PURPLE_L, PURPLE)
        gx += 1.45
    ax.text(gx + 0.6, 6.4, "kernels back to back", fontsize=9, color=TEAL, va="center")

    # parity flag
    ax.text(0.3, 4.4, "Arguments are frozen at capture, so ping-pong uses a flag in device memory:",
            fontsize=10, weight="bold")
    for t in range(4):
        x0 = 1.0 + t * 7.0
        p = t % 2
        ax.text(x0, 3.4, f"replay {t}: parity = {p}", fontsize=9, family=MONO)
        cell(ax, x0, 1.9, 2.4, 0.8, "buf0", TEAL_L if p == 0 else AMBER_L, TEAL if p == 0 else AMBER, fs=8.5)
        cell(ax, x0 + 3.2, 1.9, 2.4, 0.8, "buf1", AMBER_L if p == 0 else TEAL_L, AMBER if p == 0 else TEAL, fs=8.5)
        a, b = (x0 + 2.4, x0 + 3.2) if p == 0 else (x0 + 3.2, x0 + 2.4)
        arrow(ax, (a, 2.3), (b, 2.3), INK)
    ax.text(1.0, 0.9, "teal = read (src), amber = written (dst); the flip kernel toggles parity at the end of each replay",
            fontsize=9, color=INK)
    save(fig, "graph.png")


if __name__ == "__main__":
    pipeline()
    fork_join()
    halo()
    graph()
