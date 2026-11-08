"""Test the CUDA C++ solutions, with or without a GPU.

Without a GPU (default): each .cu file is rewritten (kernel<<<grid, block>>>(args) ->
cuda_cpu::launch(...)), compiled as C++20 against cuda_cpu.h, loaded with ctypes, and its
solve() is compared with a PyTorch reference on the same cases as the Triton tests.
With --gpu: the same files are built with nvcc and run on CUDA tensors.

    docker run --rm -v "$PWD":/w -w /w cuda-cpu-test python tests/cuda/run_tests.py [name-filter]
    python tests/cuda/run_tests.py --gpu [name-filter]          # on a machine with nvcc + a GPU

The image is python:3.11-slim plus g++ and CPU PyTorch (see the page "Running the CUDA
versions"). tests/cuda/nvcc_check.sh compiles the same files with the real nvcc.
"""
import ctypes
import math
import re
import subprocess
import sys
import tempfile
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
BUILD = Path(tempfile.gettempdir()) / "cuda_cpu_build"
FAILED = []
GPU = "--gpu" in sys.argv


# ---------------------------------------------------------------- build

def _match(src, i, open_ch, close_ch):
    depth = 0
    for k in range(i, len(src)):
        depth += src[k] == open_ch
        depth -= src[k] == close_ch
        if depth == 0:
            return k
    raise ValueError("unbalanced")


def _split_top(s):
    parts, depth, cur = [], 0, ""
    for ch in s:
        depth += ch in "([{"
        depth -= ch in ")]}"
        if ch == "," and depth == 0:
            parts.append(cur)
            cur = ""
        else:
            cur += ch
    return parts + [cur]


def to_cpp(src):
    """CUDA source -> C++ for the CPU emulator."""
    src = re.sub(r"#include\s*<(cuda(_runtime|_fp16)?\.h|cooperative_groups(/\w+)?\.h)>\s*\n", "", src)
    out, i = [], 0
    while (k := src.find("<<<", i)) >= 0:
        j, depth = k, 0                                   # walk back over name<template args>
        while j > 0:
            ch = src[j - 1]
            if ch == ">":
                depth += 1
            elif ch == "<":
                depth -= 1
            elif depth == 0 and not (ch.isalnum() or ch in "_:"):
                break
            j -= 1
        e = src.index(">>>", k)
        grid, block = (p.strip() for p in _split_top(src[k + 3:e])[:2])
        p = src.index("(", e)
        q = _match(src, p, "(", ")")
        out.append(src[i:j])
        out.append(f"cuda_cpu::launch(dim3({grid}), dim3({block}), {src[j:k].strip()}, {src[p + 1:q]})")
        i = q + 1
    out.append(src[i:])
    return '#include "cuda_cpu.h"\n' + "".join(out)


def build(cu):
    BUILD.mkdir(exist_ok=True)
    key = str(cu.relative_to(ROOT).with_suffix("")).replace("/", "__")   # unique per file
    lib = BUILD / (key + ".so")
    if GPU:
        subprocess.run(["nvcc", "-O3", "-arch=native", "-shared", "-Xcompiler", "-fPIC",
                        "-o", str(lib), str(cu)], check=True)
        return ctypes.CDLL(str(lib))
    cpp = BUILD / (key + ".cpp")
    cpp.write_text(to_cpp(cu.read_text()))
    subprocess.run(["g++", "-std=c++20", "-O2", "-ffp-contract=off", "-fPIC", "-shared", "-pthread",
                    "-Wno-unknown-pragmas", f"-I{HERE}", "-o", str(lib), str(cpp)], check=True)
    return ctypes.CDLL(str(lib))


def fn(lib, name, *types):
    f = getattr(lib, name)
    f.argtypes = list(types)
    f.restype = None
    return f


P, I, F = ctypes.c_void_p, ctypes.c_int, ctypes.c_float


def ptr(t):
    return ctypes.c_void_p(t.data_ptr())


def check(name, got, want, atol, rtol):
    got, want = got.double(), want.double()
    err = (got - want).abs()
    ok = bool((err <= atol + rtol * want.abs()).all()) and got.shape == want.shape
    worst = err.max().item() if err.numel() else 0.0
    print(f"  {'ok  ' if ok else 'FAIL'} {name:40s} max abs err {worst:.2e}")
    if not ok:
        FAILED.append(name)


# ---------------------------------------------------------------- problems

def t01(lib):
    solve = fn(lib, "solve", P, P, P, I)
    for n in [1, 31, 33, 1023, 1025, 100_003]:
        a, b = torch.empty(n).uniform_(-1000, 1000), torch.empty(n).uniform_(-1000, 1000)
        c = torch.empty(n)
        solve(ptr(a), ptr(b), ptr(c), n)
        check(f"N={n}", c, a + b, 1e-5, 1e-5)


def t02(lib):
    solve = fn(lib, "solve", P, P, I, I)
    for r, c in [(1, 1), (1, 70), (70, 1), (63, 65), (129, 31), (200, 333)]:
        x = torch.empty(r, c).uniform_(-10, 10)
        y = torch.empty(c, r)
        solve(ptr(x), ptr(y), r, c)
        check(f"{r}x{c}", y, x.t(), 0, 0)


def t03(lib):
    solve = fn(lib, "solve", P, P, I)
    for n in [1, 7, 4095, 4097, 300_001]:
        x = torch.empty(n).uniform_(-1000, 1000)
        out = torch.empty(1)
        solve(ptr(x), ptr(out), n)
        check(f"N={n}", out, x.double().sum().reshape(1), 1e-5, 1e-5)


def t04(lib):
    solve = fn(lib, "solve", P, P, P, I)
    for n in [1, 31, 4097, 250_007]:
        a, b = torch.empty(n).uniform_(-1, 1), torch.empty(n).uniform_(-1, 1)
        out = torch.zeros(1)
        solve(ptr(a), ptr(b), ptr(out), n)
        check(f"N={n}", out, (a.double() @ b.double()).reshape(1), 1e-5, 1e-5)


def t05(lib):
    solve = fn(lib, "solve", P, P, I)
    ref = lambda x: torch.softmax(x.double(), 0)
    for n in [1, 2, 4095, 4097, 40_001]:
        x = torch.empty(n).uniform_(-10, 10)
        y = torch.empty(n)
        solve(ptr(x), ptr(y), n)
        check(f"N={n}", y, ref(x), 1e-5, 1e-5)
    x = torch.full((5000,), 80.0)
    y = torch.empty_like(x)
    solve(ptr(x), ptr(y), 5000)
    check("large equal values (overflow)", y, ref(x), 1e-5, 1e-5)


def t06(lib):
    solve = fn(lib, "solve", P, F, F, P, I, F)
    for n in [1, 3, 4095, 4097, 33_333]:
        x = torch.empty(n).uniform_(-100, 100)
        y = torch.empty(n)
        solve(ptr(x), 1.5, -2.0, ptr(y), n, 1e-5)
        xd = x.double()
        check(f"N={n}", y, 1.5 * xd / torch.sqrt((xd ** 2).mean() + 1e-5) - 2.0, 1e-5, 1e-5)


def t07(lib):
    solve = fn(lib, "solve", P, P, P, P, I, I, F)
    for n, c in [(1, 1), (3, 7), (17, 100), (33, 512), (9, 1024), (5, 4096)]:
        x = torch.empty(n, c).uniform_(-100, 100) + 1000.0
        w, b = torch.empty(c).uniform_(0.1, 10), torch.empty(c).uniform_(-10, 10)
        y = torch.empty_like(x)
        solve(ptr(x), ptr(w), ptr(b), ptr(y), n, c, 1e-5)
        want = torch.nn.functional.layer_norm(x.double(), (c,), w.double(), b.double(), 1e-5)
        check(f"{n}x{c}", y, want, 1e-4, 1e-4)


def t08(lib):
    solve = fn(lib, "solve", P, P, P, P, I, I, F)
    for n, c in [(1, 1), (3, 5), (7, 1000), (2, 4096), (2, 4097), (3, 20_000)]:
        x, r = torch.empty(n, c).uniform_(-10, 10), torch.empty(n, c).uniform_(-10, 10)
        w = torch.empty(c).uniform_(0.5, 1.5)
        out = torch.empty_like(x)
        solve(ptr(x), ptr(r), ptr(w), ptr(out), n, c, 1e-5)
        z = (x + r).double()
        check(f"{n}x{c}", out, z / torch.sqrt((z ** 2).mean(-1, keepdim=True) + 1e-5) * w, 1e-5, 1e-5)


def t09(lib):
    solve = fn(lib, "solve", P, P, P, I, I, I, F, F)
    for M, N, K in [(16, 16, 16), (17, 33, 50), (64, 48, 100), (100, 70, 129)]:
        a = torch.empty(M, K).uniform_(-1, 1).half()
        b = torch.empty(K, N).uniform_(-1, 1).half()
        c = torch.empty(M, N).uniform_(-1, 1).half()
        want = (0.5 * (a.float() @ b.float()) + 1.5 * c.float()).half()
        solve(ptr(a), ptr(b), ptr(c), M, N, K, 0.5, 1.5)
        check(f"{M}x{N}x{K}", c, want, 0.05, 0.05)


def t10(lib):
    solve = fn(lib, "solve", P, P, P, I, I, I, I)
    for B, M, N, K in [(1, 1, 1, 1), (3, 17, 33, 9), (2, 64, 40, 100), (2, 70, 65, 33)]:
        a = torch.empty(B, M, K).uniform_(-5, 5).half()
        b = torch.empty(B, K, N).uniform_(-5, 5).half()
        c = torch.empty(B, M, N).half()
        solve(ptr(a), ptr(b), ptr(c), B, M, N, K)
        check(f"{B}x{M}x{N}x{K}", c, torch.bmm(a.float(), b.float()).half(), 0.05, 0.05)


def t11(lib):
    solve = fn(lib, "solve", P, P, P, I, I, I)
    for N, D, M in [(1, 1, 1), (4, 3, 3), (2, 5, 1), (1000, 37, 50), (3000, 64, 7)]:
        src = torch.empty(N, D).uniform_(-10, 10)
        index = torch.randint(0, M, (N,), dtype=torch.int32)
        out = torch.full((M, D), 123.0)                   # stale data must be cleared
        solve(ptr(src), ptr(index), ptr(out), N, D, M)
        want = torch.zeros(M, D, dtype=torch.float64).index_add_(0, index.long(), src.double())
        check(f"N={N} D={D} M={M}", out, want, 1e-4, 1e-4)


def t12(lib):
    solve = fn(lib, "solve", P, P, I)

    def ref(p, n):
        q = p.view(n, 3)
        d = ((q[:, None, :] - q[None, :, :]) ** 2).sum(2)
        d.fill_diagonal_(float("inf"))
        return d.argmin(1).int()

    for n in [1, 2, 3, 65, 129, 1000]:
        pts = torch.empty(n * 3).uniform_(-1000, 1000)
        idx = torch.full((n,), -1, dtype=torch.int32)
        solve(ptr(pts), ptr(idx), n)
        check(f"N={n}", idx, ref(pts, n), 0, 0)
    pts = torch.tensor([0, 0, 0, 1, 0, 0, 5, 5, 5], dtype=torch.float32)
    idx = torch.empty(3, dtype=torch.int32)
    solve(ptr(pts), ptr(idx), 3)
    check("LeetGPU example", idx, torch.tensor([1, 0, 1]), 0, 0)
    pts = torch.zeros(300 * 3)
    idx = torch.empty(300, dtype=torch.int32)
    solve(ptr(pts), ptr(idx), 300)
    check("all ties", idx, ref(pts, 300), 0, 0)


def t13(lib):
    solve = fn(lib, "solve", P, P, P, I)
    for n in [1, 2, 63, 65, 500]:
        p = torch.empty(n, 3).uniform_(-10, 10)
        m = torch.empty(n).uniform_(0.1, 10)
        f = torch.zeros(n, 3)
        solve(ptr(p), ptr(m), ptr(f), n)
        pd, md = p.double(), m.double()
        r = pd[None, :, :] - pd[:, None, :]
        want = (md[None, :, None] * r / (((r * r).sum(2) + 1e-6) ** 1.5)[:, :, None]).sum(1)
        check(f"N={n}", f, want, 1e-2, 1e-2)


def t14(lib):
    solve = fn(lib, "solve", P, P, I, I)
    for r, c in [(1, 1), (1, 9), (9, 1), (2, 2), (3, 3), (17, 130), (100, 257)]:
        x = torch.empty(r, c).uniform_(-100, 100)
        y = torch.empty_like(x)
        solve(ptr(x), ptr(y), r, c)
        want = x.clone()
        want[1:-1, 1:-1] = 0.25 * (x[:-2, 1:-1] + x[2:, 1:-1] + x[1:-1, :-2] + x[1:-1, 2:])
        check(f"{r}x{c}", y, want, 1e-5, 1e-5)


def t15(lib):
    solve = fn(lib, "solve", P, P, P, I, I, I, I)
    for M, N, K in [(1, 1, 1), (5, 7, 3), (33, 70, 17), (100, 130, 65)]:
        A = torch.empty(M, N).uniform_(-10, 10) * (torch.rand(M, N) > 0.65)
        B = torch.empty(N, K).uniform_(-10, 10)
        C = torch.empty(M, K)
        solve(ptr(A), ptr(B), ptr(C), M, N, K, int((A != 0).sum()))
        check(f"{M}x{N}x{K}", C, A.double() @ B.double(), 1e-3, 1e-3)


def t16(lib):
    solve = fn(lib, "solve", P, P, P, I)

    def ref(v, f):
        vd = v.double()
        excl = torch.zeros_like(vd)
        excl[1:] = torch.cumsum(vd[:-1], 0)
        seg = torch.cumsum(f.long(), 0) - 1
        return excl - excl[f.bool()][seg]

    for n, p in [(1, 0.5), (2, 0.0), (5, 1.0), (4095, 0.01), (4097, 0.0), (9000, 0.3),
                 (50_000, 0.0), (300_000, 1 / 256)]:
        v = torch.empty(n).uniform_(-100, 100)
        f = (torch.rand(n) < p).int()
        f[0] = 1
        out = torch.empty(n)
        solve(ptr(v), ptr(f), ptr(out), n)
        check(f"N={n} p(flag)={p:.3g}", out, ref(v, f), 1e-3, 1e-3)


def t17(lib):
    solve = fn(lib, "solve", P, P, P, P, P, P, I, I, I, F)
    for M, K, N in [(1, 1, 1), (3, 5, 7), (40, 70, 33), (65, 130, 100)]:
        x = torch.empty(M, K).uniform_(-1, 1) - 1000.0
        g, b = torch.empty(K).uniform_(0.5, 1.5), torch.empty(K).uniform_(-0.5, 0.5)
        W = torch.empty(K, N).uniform_(-1, 1) / K ** 0.5
        bias = torch.empty(N).uniform_(-0.5, 0.5)
        y = torch.empty(M, N)
        solve(ptr(x), ptr(g), ptr(b), ptr(W), ptr(bias), ptr(y), M, K, N, 1e-5)
        xd, gd, bd, Wd, biasd = (t.double() for t in (x, g, b, W, bias))
        mu = xd.mean(1, keepdim=True)
        var = ((xd - mu) ** 2).mean(1, keepdim=True)
        h = ((xd - mu) / torch.sqrt(var + 1e-5) * gd + bd) @ Wd + biasd
        check(f"M={M} K={K} N={N}", y, 0.5 * h * (1 + torch.erf(h / math.sqrt(2))), 1e-3, 1e-3)


def t18(lib):
    solve = fn(lib, "solve", P, P, P, P, I, I, I)
    for M, N, d in [(1, 1, 1), (2, 3, 4), (17, 100, 33), (70, 129, 128), (5, 7, 100)]:
        Q, K, V = (torch.empty(n, d).uniform_(-1, 1) for n in (M, N, N))
        out = torch.empty(M, d)
        solve(ptr(Q), ptr(K), ptr(V), ptr(out), M, N, d)
        Qd, Kd, Vd = Q.double(), K.double(), V.double()
        check(f"M={M} N={N} d={d}", out, torch.softmax(Qd @ Kd.T / d ** 0.5, 1) @ Vd, 1e-4, 1e-4)


def t19(lib):
    solve = fn(lib, "solve", P, P, P, P, P, I, I, F)
    for N, C in [(1, 1), (3, 5), (17, 100), (300, 129), (9, 2000)]:
        x = torch.empty(N, C).uniform_(-250, 250)
        w = torch.empty(C).uniform_(-2, 2)
        dy = torch.empty(N, C).uniform_(-10, 10)
        dx, dw = torch.empty_like(x), torch.empty(C)
        solve(ptr(x), ptr(w), ptr(dy), ptr(dx), ptr(dw), N, C, 1e-5)
        xd, wd, dyd = x.double(), w.double(), dy.double()
        rstd = torch.rsqrt((xd * xd).mean(1, keepdim=True) + 1e-5)
        xhat, g = xd * rstd, dyd * wd
        check(f"dx {N}x{C}", dx, rstd * (g - xhat * (g * xhat).mean(1, keepdim=True)), 1e-3, 1e-3)
        check(f"dw {N}x{C}", dw, (dyd * xhat).sum(0), 1e-3, 1e-3)


def t20(lib):
    solve = fn(lib, "solve", P, P, P, P, P, P, P, P, I, I)
    for N, C in [(1, 1), (3, 5), (17, 100), (300, 129), (9, 2000)]:
        x = torch.empty(N, C).uniform_(-250, 250)
        gamma = torch.empty(C).uniform_(-2, 2)
        dy = torch.empty(N, C).uniform_(-10, 10)
        mean = x.mean(1)
        rstd = torch.rsqrt(x.var(1, unbiased=False) + 1e-5)
        dx, dg, db = torch.empty_like(x), torch.empty(C), torch.empty(C)
        solve(ptr(x), ptr(gamma), ptr(mean), ptr(rstd), ptr(dy), ptr(dx), ptr(dg), ptr(db), N, C)
        xd, gd, md, rd, dyd = (t.double() for t in (x, gamma, mean, rstd, dy))
        xhat = (xd - md[:, None]) * rd[:, None]
        g = dyd * gd
        want_dx = rd[:, None] * (g - g.mean(1, keepdim=True) - xhat * (g * xhat).mean(1, keepdim=True))
        check(f"dx {N}x{C}", dx, want_dx, 1e-3, 1e-3)
        check(f"dgamma {N}x{C}", dg, (dyd * xhat).sum(0), 1e-3, 1e-3)
        check(f"dbeta {N}x{C}", db, dyd.sum(0), 1e-3, 1e-3)


def stencil(lib):
    grad = fn(lib, "grad2d", P, P, P, I, I, F, F)
    div = fn(lib, "div2d", P, P, P, I, I, F, F)
    lap = fn(lib, "lap3d", P, P, I, I, I, F, F, F)
    dx, dy, dz = 0.1, 0.25, 0.3
    for ny, nx in [(1, 1), (3, 5), (17, 70), (64, 129)]:
        u = torch.randn(ny + 2, nx + 2)
        gx, gy = torch.zeros_like(u), torch.zeros_like(u)
        grad(ptr(u), ptr(gx), ptr(gy), nx, ny, dx, dy)
        check(f"grad2d {ny}x{nx} d/dx", gx[1:-1, 1:-1], (u[1:-1, 2:] - u[1:-1, :-2]) / (2 * dx), 1e-5, 1e-5)
        check(f"grad2d {ny}x{nx} d/dy", gy[1:-1, 1:-1], (u[2:, 1:-1] - u[:-2, 1:-1]) / (2 * dy), 1e-5, 1e-5)
        vx, vy, out = torch.randn_like(u), torch.randn_like(u), torch.zeros_like(u)
        div(ptr(vx), ptr(vy), ptr(out), nx, ny, dx, dy)
        want = (vx[1:-1, 2:] - vx[1:-1, :-2]) / (2 * dx) + (vy[2:, 1:-1] - vy[:-2, 1:-1]) / (2 * dy)
        check(f"div2d  {ny}x{nx}", out[1:-1, 1:-1], want, 1e-5, 1e-5)
    for nz, ny, nx in [(1, 1, 1), (4, 3, 5), (9, 17, 70), (5, 9, 33)]:
        u = torch.randn(nz + 2, ny + 2, nx + 2)
        out = torch.zeros_like(u)
        lap(ptr(u), ptr(out), nx, ny, nz, dx, dy, dz)
        c = u[1:-1, 1:-1, 1:-1]
        want = ((u[1:-1, 1:-1, 2:] + u[1:-1, 1:-1, :-2] - 2 * c) / dx ** 2
                + (u[1:-1, 2:, 1:-1] + u[1:-1, :-2, 1:-1] - 2 * c) / dy ** 2
                + (u[2:, 1:-1, 1:-1] + u[:-2, 1:-1, 1:-1] - 2 * c) / dz ** 2)
        check(f"lap3d  {nz}x{ny}x{nx}", out[1:-1, 1:-1, 1:-1], want, 1e-4, 1e-4)


def t21(lib):
    f = fn(lib, "solve", P, P, P, I, I, I)
    for M, N, K in [(2, 2, 2), (1, 3, 1), (1, 1, 1), (1, 5, 3), (5, 3, 1), (2, 3, 1), (8, 6, 10),
                    (32, 8, 16), (8, 16, 32), (130, 70, 129), (129, 64, 132), (200, 257, 140),
                    (16, 6144, 132)]:
        A, B = torch.empty(M, N).uniform_(-1, 1), torch.empty(N, K).uniform_(-1, 1)
        C = torch.zeros(M, K)
        f(ptr(A), ptr(B), ptr(C), M, N, K)
        check(f"matmul {M}x{N}x{K}", C, A @ B, 1e-4, 1e-4)


def t22(lib):
    f = fn(lib, "solve", P, P, P, I, I, I)
    for M, N in [(1, 1), (2, 2), (3, 4), (10, 8), (20, 20), (7, 130), (33, 2048), (5, 2049), (9, 4100), (6, 4103)]:
        A = torch.empty(M, N).uniform_(-10, 10) * (torch.rand(M, N) > 0.65)
        x, y = torch.empty(N).uniform_(-5, 5), torch.zeros(M)
        f(ptr(A), ptr(x), ptr(y), M, N, int((A != 0).sum()))
        check(f"spmv {M}x{N}", y, A @ x, 1e-3, 1e-3)


def t23(lib):
    f = fn(lib, "solve", P, P, P, I, I, I, I)
    for Bt, M, N, K in [(2, 2, 2, 3), (1, 3, 3, 3), (4, 2, 2, 2), (8, 4, 3, 2), (16, 16, 16, 16),
                        (2, 8, 6, 4), (3, 70, 65, 33), (2, 64, 64, 64)]:
        A, B = torch.empty(Bt, M, K).uniform_(-1, 1), torch.empty(Bt, K, N).uniform_(-1, 1)
        C = torch.zeros(Bt, M, N)
        f(ptr(A), ptr(B), ptr(C), Bt, M, N, K)
        check(f"bmm {Bt}x{M}x{N}x{K}", C, torch.bmm(A, B), 1e-5, 1e-5)


def warp(lib):
    """notes/warp_primitives.cu: shuffles, votes, match, reduce, cooperative groups."""
    U = ctypes.c_uint
    ws = fn(lib, "warp_sums", P, P, I)
    wm = fn(lib, "warp_max", P, P, I)
    sb = fn(lib, "segment_broadcast", P, P, I, I)
    sc = fn(lib, "warp_scan", P, P, I, I)
    bs = fn(lib, "block_scan", P, P, I)
    vo = fn(lib, "warp_votes", P, P, P, P, I)
    cp = fn(lib, "compact_positive", P, P, P, I)
    hm = fn(lib, "histogram_match", P, P, I, I)
    im = fn(lib, "int_sum_max", P, P, P, I)
    cpos = fn(lib, "count_positive", P, P, I)
    cbs = fn(lib, "cg_block_sum", P, P, I)
    cts = fn(lib, "cg_tile_scan", P, P, P, I)
    nz = fn(lib, "normalize_by_sum", P, P, I)

    def pad(t, m, value):                               # pad to a multiple of m, view as rows
        k = (-t.numel()) % m
        return torch.cat([t, torch.full((k,), value, dtype=t.dtype)]).view(-1, m)

    def seg_scan(t, w):
        return torch.cat([c.cumsum(0) for c in t.split(w)]).to(t.dtype)

    for n in [1, 31, 32, 33, 1000, 4099]:
        x = torch.randn(n)
        out = torch.zeros((n + 31) // 32)
        ws(ptr(x), ptr(out), n)
        check(f"shfl_down warp sums n={n}", out, pad(x, 32, 0.0).sum(1), 1e-5, 1e-5)

        out = torch.zeros(n)
        wm(ptr(x), ptr(out), n)
        check(f"shfl_xor warp max (all lanes) n={n}", out,
              pad(x, 32, -math.inf).amax(1).repeat_interleave(32)[:n], 0, 0)

        for w in [4, 8, 32]:
            sb(ptr(x), ptr(out), n, w)
            check(f"shfl width={w} broadcast n={n}", out, x[(torch.arange(n) // w) * w], 0, 0)

        xi = torch.randint(-5, 6, (n,), dtype=torch.int32)
        oi = torch.zeros(n, dtype=torch.int32)
        for w in [8, 32]:
            sc(ptr(xi), ptr(oi), n, w)
            check(f"shfl_up scan width={w} n={n}", oi, seg_scan(xi, w), 0, 0)

        bs(ptr(xi), ptr(oi), n)
        check(f"block scan n={n}", oi, xi.cumsum(0).to(torch.int32), 0, 0)

        nw = (n + 31) // 32
        xv = torch.randn(n).abs() + 0.1
        xv[torch.rand(n) < 0.02] *= -1                  # a few negatives
        xv[: min(n, 32)] = xv[: min(n, 32)].abs()       # first warp: all positive
        an, al = torch.zeros(nw, dtype=torch.int32), torch.zeros(nw, dtype=torch.int32)
        bal = torch.zeros(nw, dtype=torch.int32)
        vo(ptr(xv), ptr(an), ptr(al), ptr(bal), n)
        rows = pad(xv, 32, 1.0)
        check(f"any_sync n={n}", an, (rows < 0).any(1).int(), 0, 0)
        check(f"all_sync n={n}", al, (rows > 0).all(1).int(), 0, 0)
        bits = (pad(xv > 0, 32, False).long() << torch.arange(32)).sum(1)
        bits = torch.where(bits >= 2 ** 31, bits - 2 ** 32, bits).int()
        check(f"ballot_sync n={n}", bal, bits, 0, 0)

        out, cnt = torch.zeros(n), torch.zeros(1, dtype=torch.int32)
        cp(ptr(x), ptr(out), ptr(cnt), n)
        k = int(cnt.item())
        want = x[x > 0]
        check(f"ballot+popc compaction count n={n}", cnt, torch.tensor([want.numel()]), 0, 0)
        check(f"ballot+popc compaction values n={n}", out[:k].sort().values, want.sort().values, 0, 0)

        nb = 16
        b = (torch.randn(n).abs() * 3).long().clamp(max=nb - 1).int()   # skewed: low bins common
        h = torch.zeros(nb, dtype=torch.int32)
        hm(ptr(b), ptr(h), n, nb)
        check(f"match_any histogram n={n}", h, torch.bincount(b, minlength=nb).int(), 0, 0)

        su, mx = torch.zeros(1, dtype=torch.int32), torch.zeros(1, dtype=torch.int32)
        xr = torch.randint(-1000, 1000, (n,), dtype=torch.int32)
        im(ptr(xr), ptr(su), ptr(mx), n)
        check(f"reduce_add_sync n={n}", su, xr.sum().reshape(1).int(), 0, 0)
        check(f"reduce_max_sync n={n}", mx, xr.max().reshape(1), 0, 0)

        nbk = (n + 255) // 256
        pb = torch.zeros(nbk, dtype=torch.int32)
        cpos(ptr(x), ptr(pb), n)
        check(f"syncthreads_count n={n}", pb, pad(x > 0, 256, False).sum(1).int(), 0, 0)

        pf = torch.zeros(nbk)
        cbs(ptr(x), ptr(pf), n)
        check(f"cg::reduce block sums n={n}", pf, pad(x, 256, 0.0).sum(1), 1e-4, 1e-5)

        inc, exc = torch.zeros(n, dtype=torch.int32), torch.zeros(n, dtype=torch.int32)
        cts(ptr(xi), ptr(inc), ptr(exc), n)
        check(f"cg::inclusive_scan tile16 n={n}", inc, seg_scan(xi, 16), 0, 0)
        check(f"cg::exclusive_scan tile16 n={n}", exc, seg_scan(xi, 16) - xi, 0, 0)

    for n in [1, 1000, 100_003]:
        x = torch.rand(n) + 0.5
        y = torch.zeros(n)
        nz(ptr(x), ptr(y), n)
        check(f"grid.sync normalize n={n}", y, x / x.double().sum(), 1e-6, 1e-4)


def cpp_in_cuda(lib):
    """cpp/src/09_cpp_in_cuda.cu: RAII device memory, views, functor and lambda kernels."""
    sf = fn(lib, "saxpy_functor", P, P, I, F)
    sl = fn(lib, "saxpy_lambda", P, P, I, F)
    sr = getattr(lib, "sum_raii")
    sr.argtypes, sr.restype = [P, I], ctypes.c_double
    for n in [1, 255, 1000, 300_001]:
        x, y0 = torch.randn(n), torch.randn(n)
        for name, f in [("functor", sf), ("lambda", sl)]:
            y = y0.clone()
            f(ptr(x), ptr(y), n, 2.5)
            check(f"saxpy {name} n={n}", y, 2.5 * x + y0, 1e-6, 1e-6)
        xh = x.cpu() if GPU else x
        got = torch.tensor([sr(ptr(xh), n)], dtype=torch.float64)
        check(f"RAII sum n={n}", got, xh.double().sum().reshape(1), 1e-9, 1e-9)



# ---------------------------------------------------------------- classic kernels (device pointers)

def k_activations(lib):
    bg = fn(lib, "bias_gelu", P, P, P, I, I)
    sw = fn(lib, "swiglu", P, P, I, I)
    for M, N in [(1, 1), (3, 5), (7, 64), (33, 1000), (4, 4096)]:
        x, b, y = torch.randn(M, N) * 3, torch.randn(N), torch.zeros(M, N)
        bg(ptr(x), ptr(b), ptr(y), M, N)
        check(f"bias+GELU {M}x{N}", y, torch.nn.functional.gelu(x + b), 1e-5, 1e-5)
    for M, H in [(1, 1), (5, 7), (16, 512)]:
        gu, y = torch.randn(M, 2 * H) * 3, torch.zeros(M, H)
        sw(ptr(gu), ptr(y), M, H)
        check(f"SwiGLU {M}x{H}", y, torch.nn.functional.silu(gu[:, :H]) * gu[:, H:], 1e-5, 1e-5)


def k_rope(lib):
    f = fn(lib, "rope", P, I, I, I, F, I)
    for T, H, D, pos0 in [(1, 1, 2, 0), (7, 3, 8, 0), (64, 4, 64, 0), (5, 2, 128, 1000)]:
        x = torch.randn(T, H, D)
        half = D // 2
        inv = 10000.0 ** (-torch.arange(half, dtype=torch.float64) * 2 / D)
        ang = (torch.arange(T, dtype=torch.float64) + pos0)[:, None] * inv[None, :]
        c, s_ = ang.cos()[:, None, :], ang.sin()[:, None, :]
        a, b = x[..., :half].double(), x[..., half:].double()
        want = torch.cat([a * c - b * s_, b * c + a * s_], -1)
        y = x.clone()
        f(ptr(y), T, H, D, 10000.0, pos0)
        check(f"RoPE T={T} H={H} D={D} pos0={pos0}", y, want, 2e-4, 1e-4)


def k_cross_entropy(lib):
    f = fn(lib, "cross_entropy", P, P, P, I, I)
    for N, V in [(1, 1), (3, 10), (16, 1000), (4, 50257)]:
        logits = torch.randn(N, V) * 4
        tgt = torch.randint(0, V, (N,), dtype=torch.int32)
        loss = torch.zeros(N)
        f(ptr(logits), ptr(tgt), ptr(loss), N, V)
        want = torch.nn.functional.cross_entropy(logits.double(), tgt.long(), reduction="none")
        check(f"cross-entropy {N}x{V}", loss, want, 1e-4, 1e-5)


def k_topk(lib):
    f = fn(lib, "topk", P, P, P, I, I, I)
    for N, V, k in [(1, 1, 1), (3, 10, 3), (8, 1000, 5), (2, 8192, 64)]:
        x = torch.stack([torch.randperm(V).float() / V for _ in range(N)])   # distinct values
        vals, idx = torch.zeros(N, k), torch.zeros(N, k, dtype=torch.int32)
        f(ptr(x), ptr(vals), ptr(idx), N, V, k)
        wv, wi = torch.topk(x, k, dim=1)
        check(f"top-k values {N}x{V} k={k}", vals, wv, 0, 0)
        check(f"top-k indices {N}x{V} k={k}", idx, wi.int(), 0, 0)


def k_bitonic(lib):
    f = fn(lib, "bitonic_sort", P, I)
    for n in [1, 2, 7, 512, 513, 1000, 4096, 70_001]:
        x = torch.randn(n)
        x[: n // 10] = 0.5                                   # duplicates
        y = x.clone()
        f(ptr(y), n)
        check(f"bitonic sort n={n}", y, torch.sort(x).values, 0, 0)


def k_radix(lib):
    f = fn(lib, "radix_sort_pairs", P, P, I)
    for n in [1, 2, 255, 256, 1000, 65_537]:
        keys = torch.randint(0, 2 ** 31 - 1, (n,), dtype=torch.int64)
        keys[: n // 3] = keys[: n // 3] % 50                  # many duplicates: tests stability
        k32 = keys.to(torch.int32)                           # bit pattern of uint32 < 2^31
        vals = torch.arange(n, dtype=torch.int32)            # original positions
        kk, vv = k32.clone(), vals.clone()
        f(ptr(kk), ptr(vv), n)
        order = torch.argsort(keys, stable=True)
        check(f"radix sort keys n={n}", kk, k32[order], 0, 0)
        check(f"radix sort values (stable) n={n}", vv, vals[order], 0, 0)


def _philox_py(c, k):
    M0, M1, W0, W1 = 0xD2511F53, 0xCD9E8D57, 0x9E3779B9, 0xBB67AE85
    c, k = list(c), list(k)
    for _ in range(10):
        p0, p1 = M0 * c[0], M1 * c[2]
        c = [((p1 >> 32) ^ c[1] ^ k[0]) & 0xFFFFFFFF, p1 & 0xFFFFFFFF,
             ((p0 >> 32) ^ c[3] ^ k[1]) & 0xFFFFFFFF, p0 & 0xFFFFFFFF]
        k = [(k[0] + W0) & 0xFFFFFFFF, (k[1] + W1) & 0xFFFFFFFF]
    return c


def k_dropout(lib):
    raw = fn(lib, "philox_raw", P, P, P)
    kat = [((0, 0, 0, 0), (0, 0), (0x6627e8d5, 0xe169c58d, 0xbc57ac4c, 0x9b00dbd8)),
           ((0xffffffff,) * 4, (0xffffffff,) * 2, (0x408f276d, 0x41c83b0e, 0xa20bc7c6, 0x6d5451fd)),
           ((0x243f6a88, 0x85a308d3, 0x13198a2e, 0x03707344), (0xa4093822, 0x299f31d0),
            (0xd16cfe09, 0x94fdcceb, 0x5001e420, 0x24126ea1))]
    for ctr, key, want in kat:
        c = torch.tensor([x - 2 ** 32 if x >= 2 ** 31 else x for x in ctr], dtype=torch.int32, device="cpu")
        kk = torch.tensor([x - 2 ** 32 if x >= 2 ** 31 else x for x in key], dtype=torch.int32, device="cpu")
        out = torch.zeros(4, dtype=torch.int32, device="cpu")
        if GPU:
            c, kk, out = c.cuda(), kk.cuda(), out.cuda()
        raw(ptr(c), ptr(kk), ptr(out))
        w = torch.tensor([x - 2 ** 32 if x >= 2 ** 31 else x for x in want], dtype=torch.int32)
        check(f"Philox4x32-10 known answer {ctr[0]:#x}", out.cpu(), w, 0, 0)
    f = getattr(lib, "dropout")
    f.argtypes, f.restype = [P, P, P, ctypes.c_longlong, F, ctypes.c_ulonglong], None
    seed, p = 0x1234_5678_9abc, 0.3
    for n in [1, 5, 4096, 100_003]:
        x = torch.randn(n)
        y, m = torch.zeros(n), torch.zeros(n, dtype=torch.uint8)
        f(ptr(x), ptr(y), ptr(m), n, p, seed)
        keep = m.bool()
        check(f"dropout scale n={n}", y[keep], x[keep] / (1 - p), 1e-6, 1e-6)
        check(f"dropout zeros n={n}", y[~keep], torch.zeros(int((~keep).sum())), 0, 0)
        if n <= 4096:                                        # bit-exact against pure Python
            want = []
            for g in range((n + 3) // 4):
                r = _philox_py((g & 0xFFFFFFFF, g >> 32, 0, 0), (seed & 0xFFFFFFFF, seed >> 32))
                want += [(v >> 8) / 16777216.0 >= p for v in r]
            check(f"dropout mask = Python Philox n={n}", m.cpu().bool(), torch.tensor(want[:n]), 0, 0)
        if n > 10_000:
            frac = keep.float().mean().item()
            ok = abs(frac - (1 - p)) < 0.01
            print(f"  {'ok  ' if ok else 'FAIL'} {'dropout keep fraction':40s} {frac:.4f} (want {1 - p})")
            if not ok:
                FAILED.append("dropout keep fraction")
            y2, m2 = torch.zeros(n), torch.zeros(n, dtype=torch.uint8)
            f(ptr(x), ptr(y2), ptr(m2), n, p, seed)
            check("dropout same seed -> same mask", m2, m, 0, 0)
            f(ptr(x), ptr(y2), ptr(m2), n, p, seed + 1)
            differ = (m2 != m).float().mean().item()
            ok = differ > 0.3
            print(f"  {'ok  ' if ok else 'FAIL'} {'dropout new seed -> new mask':40s} {differ:.3f} differ")
            if not ok:
                FAILED.append("dropout new seed")


def k_int8(lib):
    qf = fn(lib, "quantize_int8", P, P, P, I, I)
    df = fn(lib, "dequantize_int8", P, P, P, I, I)
    for M, N in [(1, 1), (3, 7), (16, 1000), (9, 4096)]:
        x = torch.randn(M, N) * 5
        x[0] = 0.0                                           # an all-zero row
        q, sc, y = torch.zeros(M, N, dtype=torch.int8), torch.zeros(M), torch.zeros(M, N)
        qf(ptr(x), ptr(q), ptr(sc), M, N)
        amax = x.abs().amax(1)
        inv = torch.where(amax > 0, torch.tensor(127.0) / amax, torch.zeros(()))
        want_q = torch.round(x * inv[:, None]).clamp(-127, 127).to(torch.int8)
        check(f"int8 quantize {M}x{N}", q, want_q, 0, 0)
        check(f"int8 scale {M}x{N}", sc, amax / 127, 0, 1e-7)
        df(ptr(q), ptr(sc), ptr(y), M, N)
        check(f"int8 dequantize {M}x{N}", y, want_q.float() * (amax / 127)[:, None], 1e-6, 1e-6)
        err = (y - x).abs().max().item() <= (amax.max() / 127 / 2).item() * 1.0001
        print(f"  {'ok  ' if err else 'FAIL'} {'int8 round-trip error <= scale/2':40s}")
        if not err:
            FAILED.append("int8 round trip")


def k_embedding_bag(lib):
    f = fn(lib, "embedding_bag", P, P, P, P, I, I, I)
    for V, D, B in [(1, 1, 1), (10, 3, 4), (1000, 64, 32), (500, 300, 7)]:
        table = torch.randn(V, D)
        sizes = torch.randint(0, 6, (B,))
        sizes[0] = 0                                         # an empty bag
        offsets = torch.cat([torch.zeros(1, dtype=torch.int64), sizes.cumsum(0)])
        nnz = int(offsets[-1])
        indices = torch.randint(0, V, (max(nnz, 1),))[:nnz]
        idx32 = indices.int().contiguous() if nnz else torch.zeros(1, dtype=torch.int32)   # keep alive:
        off32 = offsets.int().contiguous()                   # ptr() takes a raw address
        for mode, mflag in [("sum", 0), ("mean", 1)]:
            out = torch.zeros(B, D)
            f(ptr(table), ptr(idx32), ptr(off32), ptr(out), B, D, mflag)
            want = torch.nn.functional.embedding_bag(indices, table, offsets, mode=mode, include_last_offset=True)
            check(f"embedding bag {mode} V={V} D={D} B={B}", out, want, 1e-5, 1e-5)


def k_maxpool(lib):
    f = fn(lib, "maxpool2d", P, P, P, I, I, I, I, I, I)
    for N, C, H, W, k, s in [(1, 1, 2, 2, 2, 2), (2, 3, 8, 8, 2, 2), (1, 4, 17, 13, 3, 2), (2, 2, 32, 32, 3, 1)]:
        x = torch.randn(N, C, H, W)
        OH, OW = (H - k) // s + 1, (W - k) // s + 1
        y, idx = torch.zeros(N, C, OH, OW), torch.zeros(N, C, OH, OW, dtype=torch.int32)
        f(ptr(x), ptr(y), ptr(idx), N, C, H, W, k, s)
        wy, wi = torch.nn.functional.max_pool2d(x, k, s, return_indices=True)
        check(f"maxpool {N}x{C}x{H}x{W} k={k} s={s}", y, wy, 0, 0)
        check(f"maxpool indices {N}x{C}x{H}x{W} k={k} s={s}", idx, wi.int(), 0, 0)


# ---------------------------------------------------------------- CUDA crash course
# Entry points take host arrays and do their own copies, so tensors stay on the CPU.

def _cpu(*shape, lo=-1.0, hi=1.0):
    return torch.empty(*shape, device="cpu").uniform_(lo, hi)


def course_vadd(lib):
    for name in ["vadd_baseline", "vadd_pinned", "vadd_unified", "vadd_unified_prefetch"]:
        f = fn(lib, name, P, P, P, I)
        for n in [1, 255, 256, 1000, 1 << 16]:
            a, b, c = _cpu(n), _cpu(n), torch.zeros(n, device="cpu")
            f(ptr(a), ptr(b), ptr(c), n)
            check(f"{name} n={n}", c, a + b, 0, 0)


def course_matmul(lib):
    for name in ["mm_baseline", "mm_no_alias"]:
        f = fn(lib, name, P, P, P, I)
        for N in [16, 32, 64]:
            a, b, c = _cpu(N, N), _cpu(N, N), torch.zeros(N, N, device="cpu")
            f(ptr(a), ptr(b), ptr(c), N)
            check(f"{name} {N}x{N}", c, a @ b, 1e-5, 1e-5)
    f = fn(lib, "mm_non_multiple", P, P, P, I)
    for N in [1, 7, 17, 50]:
        a, b, c = _cpu(N, N), _cpu(N, N), torch.zeros(N, N, device="cpu")
        f(ptr(a), ptr(b), ptr(c), N)
        check(f"mm_non_multiple {N}x{N}", c, a @ b, 1e-5, 1e-5)
    for name in ["mm_rectangular", "mm_tiled"]:
        f = fn(lib, name, P, P, P, I, I, I)
        for M, K, N in [(1, 1, 1), (3, 5, 7), (17, 33, 9), (40, 16, 70), (64, 100, 48)]:
            a, b, c = _cpu(M, K), _cpu(K, N), torch.zeros(M, N, device="cpu")
            f(ptr(a), ptr(b), ptr(c), M, K, N)
            check(f"{name} {M}x{K}x{N}", c, a @ b, 1e-5, 1e-5)


def course_reduction(lib):
    for name in ["sum_diverged", "sum_bank_conflicts", "sum_no_conflicts", "sum_reduce_idle", "sum_warp", "sum_cg"]:
        f = getattr(lib, name)
        f.argtypes, f.restype = [P, I], ctypes.c_int
        for n in [1, 255, 256, 257, 1000, 70_001]:
            x = torch.randint(-50, 50, (n,), dtype=torch.int32, device="cpu")
            got = torch.tensor([f(ptr(x), n)])
            check(f"{name} n={n}", got, x.long().sum().reshape(1), 0, 0)


def course_histogram(lib):
    for name in ["histogram_global", "histogram_shared"]:
        f = fn(lib, name, P, I, P, I)
        for n, nb in [(1, 1), (1000, 7), (50_000, 26), (100_000, 256)]:
            x = torch.randint(0, nb, (n,), dtype=torch.int32, device="cpu")
            h = torch.zeros(nb, dtype=torch.int32, device="cpu")
            f(ptr(x), n, ptr(h), nb)
            check(f"{name} n={n} bins={nb}", h, torch.bincount(x, minlength=nb).int(), 0, 0)


def course_convolution(lib):
    f = fn(lib, "conv1d_step", P, I, P, I, P, I)
    for step, label in [(1, "naive"), (2, "constant"), (3, "tiled"), (4, "cache")]:
        for n, m in [(1, 1), (10, 3), (257, 7), (1000, 31), (4096, 5)]:
            x, mask, y = _cpu(n), _cpu(m), torch.zeros(n, device="cpu")
            f(ptr(x), n, ptr(mask), m, ptr(y), step)
            want = torch.nn.functional.conv1d(x.view(1, 1, -1), mask.view(1, 1, -1), padding=m // 2).view(-1)
            check(f"conv1d {label} n={n} m={m}", y, want, 1e-5, 1e-5)
    g = fn(lib, "conv2d", P, I, P, I, P)
    for N, m in [(1, 1), (5, 3), (33, 7), (64, 5)]:
        img, mask, out = _cpu(N, N), _cpu(m, m), torch.zeros(N, N, device="cpu")
        g(ptr(img), N, ptr(mask), m, ptr(out))
        want = torch.nn.functional.conv2d(img.view(1, 1, N, N), mask.view(1, 1, m, m), padding=m // 2).view(N, N)
        check(f"conv2d N={N} m={m}", out, want, 1e-5, 1e-5)


TESTS = {f"{n:02d}": globals()[f"t{n:02d}"] for n in range(1, 24)}
EXTRA = {"docs/notes/stencil_ops.cu": stencil, "docs/notes/warp_primitives.cu": warp,
         "docs/cpp/src/09_cpp_in_cuda.cu": cpp_in_cuda,
         "docs/cuda-course/src/01_vector_add.cu": course_vadd,
         "docs/cuda-course/src/02_matmul.cu": course_matmul,
         "docs/cuda-course/src/03_reduction.cu": course_reduction,
         "docs/cuda-course/src/04_histogram.cu": course_histogram,
         "docs/cuda-course/src/05_convolution.cu": course_convolution,
         "docs/kernels/src/01_activations.cu": k_activations,
         "docs/kernels/src/02_rope.cu": k_rope,
         "docs/kernels/src/03_cross_entropy.cu": k_cross_entropy,
         "docs/kernels/src/04_topk.cu": k_topk,
         "docs/kernels/src/05_bitonic_sort.cu": k_bitonic,
         "docs/kernels/src/06_radix_sort.cu": k_radix,
         "docs/kernels/src/07_dropout.cu": k_dropout,
         "docs/kernels/src/08_int8_quant.cu": k_int8,
         "docs/kernels/src/09_embedding_bag.cu": k_embedding_bag,
         "docs/kernels/src/10_maxpool2d.cu": k_maxpool}


def main():
    torch.manual_seed(0)
    if GPU:
        torch.set_default_device("cuda")                 # every test tensor lives on the GPU
    args = [a for a in sys.argv[1:] if a != "--gpu"]
    pattern = args[0] if args else ""
    files = sorted((ROOT / "docs" / "leetgpu").glob("[0-9][0-9]_*.cu")) + [ROOT / k for k in EXTRA]
    for cu in files:
        if pattern not in str(cu.relative_to(ROOT)):
            continue
        print(cu.relative_to(ROOT))
        test = EXTRA.get(str(cu.relative_to(ROOT))) or TESTS[cu.name[:2]]
        test(build(cu))
    print(f"\n{len(FAILED)} failed" + (": " + ", ".join(FAILED) if FAILED else ""))
    sys.exit(1 if FAILED else 0)


if __name__ == "__main__":
    main()
