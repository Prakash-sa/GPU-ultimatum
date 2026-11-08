"""Generate the figures for the "C++ for GPU programmers" course.

Run from the repo root:

    .venv/bin/python scripts/make_cpp_figures.py

Writes PNG files into docs/assets/img/cpp/. Reuses the palette and helpers from
make_figures.py so all figures look the same.
"""
from make_figures import (AMBER, AMBER_L, BLUE, BLUE_L, GRAY, GRAY_L, INK, MONO, OUT, PURPLE,
                          PURPLE_L, RED, RED_L, TEAL, TEAL_L, box, canvas, plt)

DIR = OUT / "cpp"
DIR.mkdir(parents=True, exist_ok=True)


def save(fig, name):
    fig.savefig(DIR / name, dpi=110, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print("wrote cpp/" + name)


def cell(ax, x, y, text, fc, ec, w=1.0, h=0.8, fs=10, mono=True, lw=1.2):
    box(ax, x, y, w * 0.94, h, fc, ec, lw=lw)
    ax.text(x + w * 0.47, y + h / 2, text, ha="center", va="center", fontsize=fs,
            family=MONO if mono else None)


def arrow(ax, a, b, color=INK, lw=1.6, style="-|>", rad=0.0, ls="-"):
    ax.annotate("", xy=b, xytext=a, arrowprops=dict(arrowstyle=style, color=color, lw=lw, ls=ls,
                                                    mutation_scale=14, connectionstyle=f"arc3,rad={rad}"))


def title(ax, x, y, text, fs=14):
    ax.text(x, y, text, fontsize=fs, weight="bold")


# ---------------------------------------------------------------- 1. stack, heap, pointer, reference

def memory_model():
    fig, ax = canvas(14, 6.4, (0, 28), (0, 12.8))
    title(ax, 0.3, 12.1, "Stack, heap, pointers and references")
    # stack
    box(ax, 0.5, 1.0, 9.5, 9.8, BLUE_L, BLUE, lw=1.6)
    ax.text(5.25, 10.2, "stack (this function's variables)", ha="center", fontsize=11, weight="bold", color=BLUE)
    rows = [("int x", "5", 8.4), ("int& r", "→ x", 7.0), ("float* p", "0x7f..a0", 5.6), ("float* q", "nullptr", 4.2),
            ("std::vector v", "ptr|size|cap", 2.8)]
    for name, val, y in rows:
        ax.text(1.0, y + 0.4, name, va="center", fontsize=10.5, family=MONO)
        cell(ax, 5.0, y, val, "white", BLUE, w=4.4)
    arrow(ax, (5.0, 7.4), (5.0, 8.4), BLUE, rad=0.6)
    ax.text(1.0, 1.5, "freed automatically when the function returns", fontsize=9.5, color=GRAY)
    # heap
    box(ax, 13.5, 1.0, 14.0, 9.8, AMBER_L, AMBER, lw=1.6)
    ax.text(20.5, 10.2, "heap (new / malloc / vector storage)", ha="center", fontsize=11, weight="bold", color=AMBER)
    ax.text(14.0, 8.9, "new float[4]", fontsize=10, family=MONO)
    for k, v in enumerate(["1.0", "2.0", "3.0", "4.0"]):
        cell(ax, 14.0 + k * 1.6, 7.6, v, "white", AMBER, w=1.6)
    ax.text(14.0, 6.0, "vector's buffer (size 3, capacity 4)", fontsize=10, family=MONO)
    for k, v in enumerate(["a", "b", "c", ""]):
        cell(ax, 14.0 + k * 1.6, 4.7, v, "white" if v else GRAY_L, AMBER, w=1.6)
    arrow(ax, (9.4, 6.0), (14.0, 8.0), AMBER)
    arrow(ax, (9.4, 3.2), (14.0, 5.1), AMBER)
    ax.text(14.0, 2.6, "lives until delete / free / the owner's destructor:", fontsize=9.5, color=GRAY)
    ax.text(14.0, 1.8, "forget it → leak; free twice → crash", fontsize=9.5, color=RED)
    ax.text(10.3, 4.6, "*p reads\nthe heap", fontsize=9, color=AMBER, ha="left")
    save(fig, "memory-model.png")


# ---------------------------------------------------------------- 2. vector growth

def vector_growth():
    fig, ax = canvas(14, 5.2, (0, 28), (0, 10.4))
    title(ax, 0.3, 9.7, "std::vector: size, capacity and reallocation")
    y = 7.6
    steps = [(1, 1), (2, 2), (3, 4), (5, 8)]
    for k, (size, cap) in enumerate(steps):
        yy = y - k * 1.8
        ax.text(0.3, yy + 0.4, f"push #{size}", va="center", fontsize=10, family=MONO)
        for i in range(cap):
            used = i < size
            cell(ax, 4.0 + i * 1.2, yy, chr(97 + i) if used else "", TEAL_L if used else GRAY_L, TEAL if used else GRAY, w=1.2)
        grew = k > 0 and steps[k - 1][1] < cap
        ax.text(4.0 + 8 * 1.2 + 0.6, yy + 0.4,
                f"size {size}, capacity {cap}" + ("   ← new buffer, old elements moved" if grew else ""),
                va="center", fontsize=10, color=RED if grew else INK)
    ax.text(0.3, 0.4, "Growth doubles the capacity, so push_back is O(1) on average, but every growth moves all "
            "elements and invalidates pointers into the old buffer.  reserve(n) allocates once.",
            fontsize=10)
    save(fig, "vector-growth.png")


# ---------------------------------------------------------------- 3. ownership

def ownership():
    fig, ax = canvas(15, 6.6, (0, 30), (0, 13.2))
    title(ax, 0.3, 12.5, "Who frees the memory? unique_ptr, shared_ptr, span")
    # unique
    ax.text(0.5, 11.0, "unique_ptr: one owner, moves", fontsize=11.5, weight="bold", color=BLUE)
    cell(ax, 0.5, 9.0, "a", BLUE_L, BLUE, w=2.0)
    cell(ax, 0.5, 6.6, "b", BLUE_L, BLUE, w=2.0)
    cell(ax, 5.0, 7.8, "float[1000]", AMBER_L, AMBER, w=3.6)
    arrow(ax, (2.4, 9.4), (5.0, 8.4), BLUE, ls="--")
    arrow(ax, (2.4, 7.0), (5.0, 8.0), BLUE)
    ax.text(0.5, 5.6, "auto b = std::move(a);", fontsize=10, family=MONO)
    ax.text(0.5, 4.9, "a is now empty (dashed), b owns", fontsize=9.5, color=GRAY)
    ax.text(0.5, 4.2, "freed once, when b dies", fontsize=9.5, color=GRAY)
    # shared
    ax.text(10.5, 11.0, "shared_ptr: shared, counted", fontsize=11.5, weight="bold", color=PURPLE)
    for k, n in enumerate(["mesh", "solver", "output"]):
        cell(ax, 10.5, 9.4 - k * 1.5, n, PURPLE_L, PURPLE, w=2.6)
        arrow(ax, (13.0, 9.8 - k * 1.5), (15.3, 8.3), PURPLE)
    box(ax, 15.3, 7.1, 4.0, 2.2, "white", PURPLE, lw=1.4)
    ax.text(17.3, 8.6, "count = 3", ha="center", fontsize=10, family=MONO, color=PURPLE)
    ax.text(17.3, 7.7, "Mesh", ha="center", fontsize=10, family=MONO)
    ax.text(10.5, 5.6, "each copy: atomic ++count", fontsize=9.5, color=GRAY)
    ax.text(10.5, 4.9, "freed when the count hits 0", fontsize=9.5, color=GRAY)
    # span
    ax.text(21.0, 11.0, "span: a view, owns nothing", fontsize=11.5, weight="bold", color=TEAL)
    for i in range(8):
        cell(ax, 21.0 + i * 1.0, 8.6, str(i), "white", AMBER, w=1.0)
    box(ax, 25.0, 8.45, 3.95, 1.1, "none", TEAL, lw=2.4)
    ax.text(21.0, 7.6, "vector<float> v (owns)", fontsize=9.5, color=AMBER)
    ax.text(25.0, 10.0, "span(v).subspan(4)", fontsize=9.5, color=TEAL, family=MONO)
    ax.text(21.0, 5.6, "= {pointer, size}: 16 bytes,", fontsize=9.5, color=GRAY)
    ax.text(21.0, 4.9, "cheap to pass by value,", fontsize=9.5, color=GRAY)
    ax.text(21.0, 4.2, "like a kernel's (ptr, n) arguments", fontsize=9.5, color=GRAY)
    ax.text(0.5, 1.6, "Rule of thumb: own with unique_ptr (or a container), look with a span or a const&, "
            "share with shared_ptr only when ownership really is shared.", fontsize=10.5)
    save(fig, "ownership.png")


# ---------------------------------------------------------------- 4. copy vs move

def copy_move():
    fig, ax = canvas(14, 5.4, (0, 28), (0, 10.8))
    title(ax, 0.3, 10.1, "Copy vs move")
    for x0, label, moved in [(0.5, "Buffer b = a;          // copy", False), (14.5, "Buffer c = std::move(a);  // move", True)]:
        ax.text(x0, 8.6, label, fontsize=10.5, family=MONO, weight="bold", color=BLUE if not moved else TEAL)
        cell(ax, x0, 6.6, "a: ptr, n", BLUE_L if not moved else GRAY_L, BLUE if not moved else GRAY, w=3.6)
        cell(ax, x0, 3.6, ("b" if not moved else "c") + ": ptr, n", TEAL_L, TEAL, w=3.6)
        for i in range(4):
            cell(ax, x0 + 6.0 + i * 1.0, 6.6, "x", AMBER_L, AMBER, w=1.0)
        if moved:
            arrow(ax, (x0 + 3.4, 4.0), (x0 + 6.0, 6.8), TEAL)
            ax.text(x0, 2.4, "steal the pointer, empty the source:", fontsize=9.5)
            ax.text(x0, 1.7, "O(1), no allocation, `a` left empty", fontsize=9.5, color=TEAL)
            ax.text(x0 + 0.2, 6.0, "nullptr, 0", fontsize=9, color=GRAY, family=MONO)
        else:
            arrow(ax, (x0 + 3.4, 7.0), (x0 + 6.0, 7.0), BLUE)
            for i in range(4):
                cell(ax, x0 + 6.0 + i * 1.0, 3.6, "x", AMBER_L, AMBER, w=1.0)
            arrow(ax, (x0 + 3.4, 4.0), (x0 + 6.0, 4.0), TEAL)
            ax.text(x0, 2.4, "new allocation + copy every element:", fontsize=9.5)
            ax.text(x0, 1.7, "O(n); both objects stay usable", fontsize=9.5, color=BLUE)
    save(fig, "copy-move.png")


# ---------------------------------------------------------------- 5. AoS vs SoA

def aos_soa():
    fig, ax = canvas(15, 5.4, (0, 30), (0, 10.8))
    title(ax, 0.3, 10.1, "Array of structs vs struct of arrays: what one warp reads")
    ax.text(0.5, 8.4, "AoS: vector<Vec3>", fontsize=11, weight="bold", color=RED)
    cols = [(BLUE_L, BLUE), (TEAL_L, TEAL), (PURPLE_L, PURPLE)]
    for i in range(24):
        fc, ec = cols[i % 3]
        cell(ax, 0.5 + i * 1.15, 6.8, "xyz"[i % 3] + str(i // 3), fc, ec, w=1.15, fs=8)
    for i in range(8):
        ax.annotate("", xy=(0.5 + 3 * i * 1.15 + 0.55, 7.65), xytext=(0.5 + 3 * i * 1.15 + 0.55, 8.2),
                    arrowprops=dict(arrowstyle="-|>", color=RED, lw=1.2))
    ax.text(0.5, 6.0, "threads read x0, x1, x2, ...: every 3rd float → 3× the memory transactions", fontsize=10, color=RED)
    ax.text(0.5, 4.4, "SoA: struct { vector<float> x, y, z; }", fontsize=11, weight="bold", color=TEAL)
    for j, (name, (fc, ec)) in enumerate(zip("xyz", cols)):
        for i in range(8):
            cell(ax, 0.5 + j * 9.6 + i * 1.15, 2.6, f"{name}{i}", fc, ec, w=1.15, fs=8)
    for i in range(8):
        ax.annotate("", xy=(0.5 + i * 1.15 + 0.55, 3.45), xytext=(0.5 + i * 1.15 + 0.55, 4.0),
                    arrowprops=dict(arrowstyle="-|>", color=TEAL, lw=1.2))
    ax.text(0.5, 1.6, "threads read x0..x7: contiguous → fully coalesced", fontsize=10, color=TEAL)
    save(fig, "aos-soa.png")


# ---------------------------------------------------------------- 6. vtable

def vtable():
    fig, ax = canvas(15, 6.0, (0, 30), (0, 12))
    title(ax, 0.3, 11.3, "Virtual functions: one indirect jump per call")
    ax.text(0.5, 10.1, "vector<unique_ptr<BoundaryCondition>>", fontsize=10, family=MONO)
    objs = [("Dirichlet", 7.4, BLUE, BLUE_L), ("Neumann", 4.0, TEAL, TEAL_L)]
    for name, y, c, cl in objs:
        box(ax, 0.5, y, 5.0, 2.2, cl, c, lw=1.4)
        ax.text(3.0, y + 1.7, name + " object", ha="center", fontsize=10, weight="bold", color=c)
        cell(ax, 0.8, y + 0.3, "data", "white", c, w=2.5, fs=9)
        cell(ax, 3.4, y + 0.3, "vptr", "white", c, w=1.8, fs=9)
        box(ax, 9.5, y, 6.0, 2.2, "white", c, lw=1.4)
        ax.text(12.5, y + 1.7, name + " vtable", ha="center", fontsize=10, weight="bold", color=c)
        ax.text(9.8, y + 1.0, "ghost  →", fontsize=9, family=MONO)
        ax.text(9.8, y + 0.35, "name   →", fontsize=9, family=MONO)
        arrow(ax, (5.1, y + 0.7), (9.5, y + 1.1), c)
        box(ax, 19.5, y + 0.6, 6.5, 1.0, cl, c)
        ax.text(22.75, y + 1.1, f"{name}::ghost()", ha="center", fontsize=9, family=MONO)
        arrow(ax, (12.4, y + 1.15), (19.5, y + 1.1), c)
    ax.text(0.5, 2.6, "walls[i]->ghost(x): load vptr → load the slot → jump.  Flexible, but the call can't be inlined.",
            fontsize=10)
    ax.text(0.5, 1.7, "In a GPU kernel, prefer templates (resolved at compile time) or a switch on an enum:",
            fontsize=10, color=RED)
    ax.text(0.5, 1.0, "different lanes taking different targets serialize the warp.", fontsize=10, color=RED)
    save(fig, "vtable.png")


# ---------------------------------------------------------------- 7. diamond

def diamond():
    fig, ax = canvas(13, 5.0, (0, 26), (0, 10))
    title(ax, 0.3, 9.3, "The diamond: plain vs virtual inheritance")

    def node(x, y, t, c, cl):
        box(ax, x, y, 3.6, 1.0, cl, c, lw=1.4)
        ax.text(x + 1.8, y + 0.5, t, ha="center", va="center", fontsize=9.5, family=MONO)

    for x0, virt in [(0.5, False), (13.5, True)]:
        ax.text(x0, 8.0, "virtual (one shared base)" if virt else "plain (two copies of the base)",
                fontsize=11, weight="bold", color=TEAL if virt else RED)
        node(x0 + 4.2, 1.2, "File", BLUE, BLUE_L)
        node(x0 + 1.0, 3.7, "Readable", BLUE, BLUE_L)
        node(x0 + 7.4, 3.7, "Writable", BLUE, BLUE_L)
        arrow(ax, (x0 + 5.6, 2.2), (x0 + 2.8, 3.7))
        arrow(ax, (x0 + 6.4, 2.2), (x0 + 9.2, 3.7))
        if virt:
            node(x0 + 4.2, 6.2, "Resource id", TEAL, TEAL_L)
            arrow(ax, (x0 + 2.8, 4.7), (x0 + 5.4, 6.2))
            arrow(ax, (x0 + 9.2, 4.7), (x0 + 6.6, 6.2))
        else:
            node(x0 + 1.0, 6.2, "Resource id", RED, RED_L)
            node(x0 + 7.4, 6.2, "Resource id", RED, RED_L)
            arrow(ax, (x0 + 2.8, 4.7), (x0 + 2.8, 6.2))
            arrow(ax, (x0 + 9.2, 4.7), (x0 + 9.2, 6.2))
    ax.text(0.5, 0.3, "f.id is ambiguous with two copies; with virtual inheritance there is exactly one.", fontsize=10)
    save(fig, "diamond.png")


# ---------------------------------------------------------------- 8. template instantiation

def templates():
    fig, ax = canvas(14, 5.0, (0, 28), (0, 10))
    title(ax, 0.3, 9.3, "Templates: one definition, one compiled copy per type, all at compile time")
    box(ax, 0.5, 3.6, 8.5, 3.6, PURPLE_L, PURPLE, lw=1.6)
    ax.text(0.9, 6.5, "template <typename T>", fontsize=10.5, family=MONO)
    ax.text(0.9, 5.7, "T axpy(T a, T x, T y) {", fontsize=10.5, family=MONO)
    ax.text(1.5, 4.9, "return a * x + y;", fontsize=10.5, family=MONO)
    ax.text(0.9, 4.1, "}", fontsize=10.5, family=MONO)
    outs = [("axpy<float>   →  FFMA (fp32 FMA)", BLUE, BLUE_L, 7.2), ("axpy<double>  →  DFMA (fp64 FMA)", TEAL, TEAL_L, 5.0),
            ("axpy<int>     →  IMAD", AMBER, AMBER_L, 2.8)]
    for t, c, cl, y in outs:
        box(ax, 14.0, y, 12.5, 1.5, cl, c, lw=1.4)
        ax.text(14.4, y + 0.75, t, va="center", fontsize=10.5, family=MONO)
        arrow(ax, (9.0, 5.4), (14.0, y + 0.75), c)
    ax.text(0.5, 1.4, "Each use with a new type stamps out new code: no runtime dispatch, full inlining. "
            "CUB, Thrust and CUTLASS are built this way.", fontsize=10)
    save(fig, "templates.png")


# ---------------------------------------------------------------- 9. iterators and algorithms

def algorithms():
    fig, ax = canvas(15, 7.0, (0, 30), (0, 14))
    title(ax, 0.3, 13.3, "Ranges and the algorithms GPU libraries parallelize")
    x0 = 0.5
    vals = [3, 1, 4, 1, 5, 9, 2, 6]
    for i, v in enumerate(vals):
        cell(ax, x0 + i * 1.3, 10.6, str(v), GRAY_L, INK, w=1.3)
    cell(ax, x0 + 8 * 1.3, 10.6, "", "white", GRAY, w=1.3)
    ax.text(x0 + 0.65, 12.0, "begin()", ha="center", fontsize=9.5, family=MONO, color=BLUE)
    ax.text(x0 + 8 * 1.3 + 0.65, 12.0, "end()", ha="center", fontsize=9.5, family=MONO, color=RED)
    arrow(ax, (x0 + 0.65, 11.8), (x0 + 0.65, 11.4), BLUE)
    arrow(ax, (x0 + 8 * 1.3 + 0.65, 11.8), (x0 + 8 * 1.3 + 0.65, 11.4), RED)
    ax.text(x0 + 12.2, 11.0, "half-open [begin, end): end is one past the last element", fontsize=10)
    rows = [
        ("transform (map)", [v * v for v in vals], BLUE, BLUE_L, "x → x²: one thread per element"),
        ("inclusive_scan", [sum(vals[:i + 1]) for i in range(8)], TEAL, TEAL_L, "running totals: CUB DeviceScan"),
        ("copy_if (compact)", [v for v in vals if v > 2] + [""] * 3, PURPLE, PURPLE_L, "keep x > 2: ballot + popc on a GPU"),
        ("sort", sorted(vals), AMBER, AMBER_L, "radix sort on a GPU (CUB DeviceRadixSort)"),
        ("reduce", [sum(vals)], RED, RED_L, "sum: tree of warp shuffles"),
    ]
    for k, (name, out, c, cl, note) in enumerate(rows):
        y = 8.6 - k * 1.7
        ax.text(x0, y + 0.4, name, va="center", fontsize=10, family=MONO, color=c)
        for i, v in enumerate(out):
            cell(ax, x0 + 6.5 + i * 1.3, y, str(v), cl if v != "" else "white", c if v != "" else GRAY, w=1.3, fs=9)
        ax.text(x0 + 6.5 + 8 * 1.3 + 0.5, y + 0.4, note, va="center", fontsize=9.5, color=GRAY)
    save(fig, "algorithms.png")


# ---------------------------------------------------------------- 10. threads: race, mutex, atomic

def threads():
    fig, ax = canvas(15, 6.6, (0, 30), (0, 13.2))
    title(ax, 0.3, 12.5, "Two threads incrementing one counter")
    panels = [
        (0.5, "data race (no sync)", RED, [("T0", "read 5", 0), ("T1", "read 5", 1), ("T0", "write 6", 2), ("T1", "write 6", 3)],
         "result 6: one increment lost (and undefined behaviour)"),
        (10.5, "std::mutex", BLUE, [("T0", "lock · 5→6 · unlock", 0), ("T1", "wait …", 0),
                                    ("T1", "lock · 6→7 · unlock", 2)], "result 7, but T1 waited: serial"),
        (20.5, "std::atomic fetch_add", TEAL, [("T0", "fetch_add → 6", 0), ("T1", "fetch_add → 7", 1)],
         "result 7, one hardware instruction each"),
    ]
    for x0, name, c, events, result in panels:
        ax.text(x0, 10.8, name, fontsize=11.5, weight="bold", color=c)
        for lane, y in (("T0", 8.6), ("T1", 6.4)):
            ax.text(x0, y + 0.45, lane, fontsize=10, family=MONO, va="center")
            ax.plot([x0 + 1.2, x0 + 9.2], [y + 0.45, y + 0.45], color=GRAY_L, lw=6, solid_capstyle="round", zorder=0)
        for lane, text, t in events:
            y = 8.6 if lane == "T0" else 6.4
            w = 4.0 if "·" in text else (3.4 if "wait" in text else 3.6)
            xx = x0 + 1.4 + t * 1.8
            box(ax, xx, y, w * 0.95 if xx + w < x0 + 9.4 else x0 + 9.2 - xx, 0.9,
                "white" if "wait" in text else c + "33", c, lw=1.2)
            ax.text(xx + 0.15, y + 0.45, text, va="center", fontsize=8.5, family=MONO)
        ax.text(x0, 4.6, "time →", fontsize=9, color=GRAY)
        ax.text(x0, 3.4, result, fontsize=9.5, color=c, wrap=True)
    ax.text(0.5, 1.2, "GPU equivalents: the race is the same; atomicAdd is the atomic; a lock built from atomicCAS "
            "is the mutex (rarely a good idea on 1,000s of threads).", fontsize=10)
    save(fig, "threads.png")


# ---------------------------------------------------------------- 11. RAII lifetime

def raii():
    fig, ax = canvas(14, 4.6, (0, 28), (0, 9.2))
    title(ax, 0.3, 8.5, "RAII: the scope is the lifetime")
    ax.plot([1.0, 26.0], [4.0, 4.0], color=INK, lw=1.4)
    for x, t in [(1.0, "{"), (26.0, "}")]:
        ax.text(x, 4.5, t, fontsize=18, family=MONO, ha="center")
    spans = [(2.5, 24.5, 5.4, "Buffer a(1000)", BLUE, "constructor: cudaMalloc / new"),
             (7.0, 22.0, 2.0, "Buffer b = a", TEAL, "copy constructor: second allocation")]
    for x0, x1, y, t, c, note in spans:
        box(ax, x0, y, x1 - x0, 1.0, c + "33", c, lw=1.4)
        ax.text(x0 + 0.3, y + 0.5, t, va="center", fontsize=10, family=MONO)
        ax.text(x0 + 6.5, y + 0.5, note, va="center", fontsize=9.5, color=GRAY)
    ax.text(22.1, 1.2, "~b()", fontsize=10, family=MONO, color=TEAL)
    ax.text(24.6, 6.6, "~a()", fontsize=10, family=MONO, color=BLUE)
    ax.text(0.5, 0.2, "Destructors run in reverse order at the closing brace, on a normal exit, an early return "
            "or an exception: nothing leaks.", fontsize=10)
    save(fig, "raii.png")


# ---------------------------------------------------------------- 12. host vs device C++

def host_device():
    fig, ax = canvas(15, 6.8, (0, 30), (0, 13.6))
    title(ax, 0.3, 12.9, "Which C++ works where in a .cu file")
    cols = [("host code only", RED, RED_L, ["std::vector, std::string, iostream", "new/delete of big buffers*",
                                             "exceptions, RTTI (dynamic_cast)", "std::thread, std::mutex", "file I/O, <random>"]),
            ("both host and device", TEAL, TEAL_L, ["templates, if constexpr, concepts", "structs, classes, operators",
                                                   "constexpr functions", "lambdas (__device__, --extended-lambda)",
                                                   "functors with __host__ __device__"]),
            ("device, via CUDA's libraries", BLUE, BLUE_L, ["cuda::std::span, array, atomic", "cuda::std::complex",
                                                           "math: sinf, fmaf, __expf …", "cooperative_groups",
                                                           "CUB, Thrust (host-callable)"])]
    for k, (name, c, cl, items) in enumerate(cols):
        x0 = 0.5 + k * 10.0
        box(ax, x0, 2.4, 9.4, 9.2, cl, c, lw=1.6)
        ax.text(x0 + 4.7, 10.9, name, ha="center", fontsize=11.5, weight="bold", color=c)
        for i, it in enumerate(items):
            ax.text(x0 + 0.4, 9.6 - i * 1.4, "• " + it, fontsize=10)
    ax.text(0.5, 1.2, "* device-side malloc/new exist but are slow and use a small heap: allocate on the host. "
            "Virtual functions work only for objects created on the device.", fontsize=9.5, color=GRAY)
    save(fig, "host-device.png")


if __name__ == "__main__":
    memory_model()
    vector_growth()
    ownership()
    copy_move()
    aos_soa()
    vtable()
    diamond()
    templates()
    algorithms()
    threads()
    raii()
    host_device()
