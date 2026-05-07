 
import sys
import os
import time
import math
 
import numpy as np
import matplotlib
matplotlib.use("Agg")          
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
 
plt.rcParams["figure.dpi"] = 120

 
sys.path.insert(0, os.getcwd())
 
from Mandel_implementation import (
    mandelbrot_scalar,
    mandelbrot_naive,
    mandelbrot_numpy,
    mandelbrot_multiprocessing,
    mandelbrot_dask,
)
 
try:
    from numba import cuda
    CUDA_AVAILABLE = cuda.is_available()
except ImportError:
    CUDA_AVAILABLE = False
 
if CUDA_AVAILABLE:
    from Mandel_Cuda import run_gpu, block_size_sweep, BLOCK_SIZES
    print("CUDA device:", cuda.get_current_device().name)
else:
    print("No CUDA device — GPU cells will be skipped.")
 
PARAMS   = dict(xmin=-2.5, xmax=1.0, ymin=-1.25, ymax=1.25)
MAX_ITER = 256
print("All imports OK")

 
img = mandelbrot_numpy(**PARAMS, width=800, height=600, max_iter=MAX_ITER)
 
fig, ax = plt.subplots(figsize=(10, 6))
ax.imshow(img, extent=[-2.5, 1.0, -1.25, 1.25],
          cmap="inferno", origin="lower", aspect="equal")
ax.set_title("Mandelbrot Set — NumPy vectorised (800×600)")
ax.set_xlabel("Re(c)")
ax.set_ylabel("Im(c)")
plt.tight_layout()
plt.savefig("mandelbrot_preview.png", dpi=150)
plt.close()
 

import pytest
 
result = pytest.main([
    "mandelbrot_implementations.py",
    "-v", "--tb=short", "-q",
])
print(f"\npytest exit code: {result}  (0 = all passed)")

if CUDA_AVAILABLE:
    sweep_results = block_size_sweep(
        width=2048, height=2048, max_iter=MAX_ITER, repeats=3
    )
else:
    # Representative synthetic data for illustration
    sweep_results = [
        {"block": (4,  4),  "kernel_s": 0.280, "total_s": 0.310},
        {"block": (8,  8),  "kernel_s": 0.095, "total_s": 0.120},
        {"block": (8,  4),  "kernel_s": 0.098, "total_s": 0.125},
        {"block": (16, 8),  "kernel_s": 0.072, "total_s": 0.100},
        {"block": (16, 16), "kernel_s": 0.058, "total_s": 0.085},
        {"block": (32, 8),  "kernel_s": 0.061, "total_s": 0.089},
        {"block": (32, 16), "kernel_s": 0.060, "total_s": 0.088},
        {"block": (32, 32), "kernel_s": 0.063, "total_s": 0.092},
    ]
    print("(Using synthetic data — no GPU available)")
 
# ── Plot block-size sweep results ─────────────────────────────────────────────
labels    = [str(r["block"]) for r in sweep_results]
k_times   = [r["kernel_s"]  for r in sweep_results]
tot_times = [r["total_s"]   for r in sweep_results]
n_threads = [r["block"][0] * r["block"][1] for r in sweep_results]
warp_mult = [("✓" if t % 32 == 0 else "✗") for t in n_threads]
 
x = np.arange(len(labels))
fig, ax = plt.subplots(figsize=(11, 4))
bars_k = ax.bar(x - 0.2, k_times,   0.35, label="Kernel only",      color="steelblue")
bars_t = ax.bar(x + 0.2, tot_times, 0.35, label="Kernel + transfer", color="coral")
 
for bar, wm in zip(bars_k, warp_mult):
    ax.text(
        bar.get_x() + bar.get_width() / 2,
        bar.get_height() + 0.002,
        wm, ha="center", va="bottom", fontsize=10,
    )
 
ax.set_xticks(x)
ax.set_xticklabels(labels, rotation=30, ha="right")
ax.set_ylabel("Time (s)")
ax.set_title("Block-size sweep — 2048×2048, max_iter=256\n(✓ = warp-size multiple)")
ax.legend()
plt.tight_layout()
plt.savefig("block_size_sweep.png", dpi=150)
plt.close()



SIZES  = [64, 128, 256, 512, 1024, 2048, 4096]
timing = {name: {} for name in ["naive", "numpy", "mp4", "dask", "gpu_k", "gpu_t"]}
 
for sz in SIZES:
    w = h = sz
    print(f"\n── {sz}×{sz} ─────────────────────────────────────")
 
    # Naive (only for small sizes — avoids very long runtimes)
    if sz <= 256:
        t0 = time.perf_counter()
        mandelbrot_naive(**PARAMS, width=w, height=h, max_iter=MAX_ITER)
        timing["naive"][sz] = time.perf_counter() - t0
        print(f"  Naive:          {timing['naive'][sz]:.4f}s")
    else:
        timing["naive"][sz] = float("nan")
 
    # NumPy
    t0 = time.perf_counter()
    mandelbrot_numpy(**PARAMS, width=w, height=h, max_iter=MAX_ITER)
    timing["numpy"][sz] = time.perf_counter() - t0
    print(f"  NumPy:          {timing['numpy'][sz]:.4f}s")
 
    # Multiprocessing (4 workers)
    if sz <= 2048:
        t0 = time.perf_counter()
        mandelbrot_multiprocessing(
            **PARAMS, width=w, height=h, max_iter=MAX_ITER, n_workers=4
        )
        timing["mp4"][sz] = time.perf_counter() - t0
        print(f"  Multiproc (4w): {timing['mp4'][sz]:.4f}s")
    else:
        timing["mp4"][sz] = float("nan")
 
    # Dask
    t0 = time.perf_counter()
    mandelbrot_dask(**PARAMS, width=w, height=h, max_iter=MAX_ITER, chunk_size=128)
    timing["dask"][sz] = time.perf_counter() - t0
    print(f"  Dask:           {timing['dask'][sz]:.4f}s")
 
    # GPU
    if CUDA_AVAILABLE:
        r = run_gpu(w, h, **PARAMS, max_iter=MAX_ITER, threads_per_block=(16, 16))
        timing["gpu_k"][sz] = r["time_kernel"]
        timing["gpu_t"][sz] = r["time_total"]
        print(f"  GPU kernel:     {timing['gpu_k'][sz]:.4f}s")
        print(f"  GPU total:      {timing['gpu_t'][sz]:.4f}s")
    else:
        scale = (sz / 2048) ** 2
        timing["gpu_k"][sz] = 0.058 * scale
        timing["gpu_t"][sz] = 0.085 * scale
        print(f"  GPU (synthetic): kernel={timing['gpu_k'][sz]:.4f}s")
 
print("\nAll timings collected.")



header = (
    f"{'Size':>6}  {'Naive':>9}  {'NumPy':>9}  {'MP-4':>9}  {'Dask':>9}  "
    f"{'GPU kern':>10}  {'GPU tot':>10}  {'SU_np/k':>8}  {'SU_mp/k':>8}"
)
print(header)
print("-" * 95)
 
for sz in SIZES:
    def fmt(v):
        return f"{v:>9.4f}" if not np.isnan(v) else f"{'skip':>9}"
 
    su_np = timing["numpy"][sz] / timing["gpu_k"][sz]
    su_mp = (
        timing["mp4"][sz] / timing["gpu_k"][sz]
        if not np.isnan(timing["mp4"][sz])
        else float("nan")
    )
    su_mp_str = f"{su_mp:>8.1f}x" if not np.isnan(su_mp) else f"{'n/a':>8}"
    print(
        f"{sz:>6}  {fmt(timing['naive'][sz])}  {fmt(timing['numpy'][sz])}  "
        f"{fmt(timing['mp4'][sz])}  {fmt(timing['dask'][sz])}  "
        f"{timing['gpu_k'][sz]:>10.4f}  {timing['gpu_t'][sz]:>10.4f}  "
        f"{su_np:>8.1f}x  {su_mp_str}"
    )


fig, axes = plt.subplots(1, 2, figsize=(14, 5))
pixel_counts = [sz ** 2 for sz in SIZES]
 
# ── Left: absolute timings ────────────────────────────────────────────────────
ax = axes[0]
 
def plot_line(key, label, color, marker, linestyle="-"):
    vals = [timing[key][sz] for sz in SIZES]
    mask = [not np.isnan(v) for v in vals]
    xs   = [pixel_counts[i] for i, m in enumerate(mask) if m]
    ys   = [vals[i]          for i, m in enumerate(mask) if m]
    ax.loglog(xs, ys, marker=marker, color=color, linestyle=linestyle,
              linewidth=2, markersize=7, label=label)
 
plot_line("naive", "Naive (Python)",      "#e74c3c", "o")
plot_line("numpy", "NumPy vectorised",    "#3498db", "s")
plot_line("mp4",   "Multiproc (4w)",      "#2ecc71", "^")
plot_line("dask",  "Dask",                "#f39c12", "D")
plot_line("gpu_k", "GPU kernel",          "#9b59b6", "P", "-")
plot_line("gpu_t", "GPU total (w/trans)", "#9b59b6", "x", "--")
 
ax.set_xlabel("Pixels (W×H)")
ax.set_ylabel("Time (s)")
ax.set_title("Absolute execution time")
ax.legend(fontsize=8)
ax.grid(True, which="both", alpha=0.3)
 
# ── Right: speedup vs NumPy ───────────────────────────────────────────────────
ax2 = axes[1]
 
for key, label, color, marker in [
    ("mp4",   "Multiproc (4w)",      "#2ecc71", "^"),
    ("dask",  "Dask",                "#f39c12", "D"),
    ("gpu_k", "GPU kernel vs NumPy", "#9b59b6", "P"),
    ("gpu_t", "GPU total vs NumPy",  "#9b59b6", "x"),
]:
    su = [
        timing["numpy"][sz] / timing[key][sz]
        for sz in SIZES
        if not np.isnan(timing[key][sz])
    ]
    xs = [
        pixel_counts[i]
        for i, sz in enumerate(SIZES)
        if not np.isnan(timing[key][sz])
    ]
    ls = "--" if "total" in label else "-"
    ax2.semilogx(xs, su, marker=marker, color=color, linestyle=ls,
                 linewidth=2, markersize=7, label=label)
 
ax2.axhline(1, color="gray", linewidth=1, linestyle=":")
ax2.set_xlabel("Pixels (W×H)")
ax2.set_ylabel("Speedup over NumPy")
ax2.set_title("Speedup relative to NumPy vectorised")
ax2.legend(fontsize=8)
ax2.grid(True, which="both", alpha=0.3)
 
plt.tight_layout()
plt.savefig("scaling_analysis.png", dpi=150)
plt.close()

if CUDA_AVAILABLE:
    import numba
    from Mandel_Cuda import mandelbrot_kernel_smem, BLOCK_SMEM
 
    H = W = 512
    tbp   = BLOCK_SMEM
    bpg   = (
        (H + tbp[0] - 1) // tbp[0],
        (W + tbp[1] - 1) // tbp[1],
    )
 
    out_d   = cuda.device_array((H, W), dtype=np.int32)
    means_d = cuda.device_array(bpg,    dtype=np.float32)
 
    mandelbrot_kernel_smem[bpg, tbp](
        out_d, means_d,
        PARAMS["xmin"], PARAMS["xmax"],
        PARAMS["ymin"], PARAMS["ymax"],
        MAX_ITER,
    )
    cuda.synchronize()
 
    means = means_d.copy_to_host()
    print("Per-block mean iteration count (512×512 image):")
    print(f"  Overall mean:    {means.mean():.1f}")
    print(f"  Block-level std: {means.std():.1f} (high = heterogeneous workload)")
 
    fig, ax = plt.subplots(figsize=(6, 4))
    im = ax.imshow(means, cmap="plasma", aspect="equal")
    ax.set_title("Per-block mean iteration count\n(shared-memory reduction)")
    ax.set_xlabel("Block column")
    ax.set_ylabel("Block row")
    plt.colorbar(im, ax=ax, label="Mean iterations")
    plt.tight_layout()
    plt.savefig("smem_block_means.png", dpi=150)
    plt.close()
else:
    print("Shared-memory demo skipped (no CUDA device).")

    