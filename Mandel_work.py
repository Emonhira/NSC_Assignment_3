 
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

