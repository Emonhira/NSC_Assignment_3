 
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
 
