import time
import math
import unittest
import argparse
import numpy as np
from functools import partial

def mandelbrot_scalar(c: complex, max_iter: int = 256) -> int:
     z = 0.0 + 0.0j
     for n in range(max_iter):
        if z.real * z.real + z.imag * z.imag > 4.0:
            return n
        z = z * z + c
     return max_iter


# Implementation 1 – Naive (pure Python)
# ─────────────────────────────────────────────────────────────────────────────
 
def mandelbrot_naive(
    xmin: float, xmax: float,
    ymin: float, ymax: float,
    width: int, height: int,
    max_iter: int = 256,
) -> np.ndarray:
     xs = np.linspace(xmin, xmax, width)
     ys = np.linspace(ymin, ymax, height)
     result = np.empty((height, width), dtype=np.int32)
     for i in range(height):
        for j in range(width):
            result[i, j] = mandelbrot_scalar(xs[j] + 1j * ys[i], max_iter)
     return result
 
# Implementation 2 – NumPy vectorised
# ─────────────────────────────────────────────────────────────────────────────
 
def mandelbrot_numpy(
    xmin: float, xmax: float,
    ymin: float, ymax: float,
    width: int, height: int,
    max_iter: int = 256,
) -> np.ndarray:
    
    xs = np.linspace(xmin, xmax, width)
    ys = np.linspace(ymin, ymax, height)
    C  = xs[np.newaxis, :] + 1j * ys[:, np.newaxis]   # (H, W) complex
 
    Z      = np.zeros_like(C)
    count  = np.zeros((height, width), dtype=np.int32)
    active = np.ones((height, width),  dtype=bool)
 
    for _ in range(max_iter):
        Z[active]      = Z[active] ** 2 + C[active]
        escaped        = active & (np.abs(Z) > 2.0)
        count[escaped] = _ + 1
        active        &= ~escaped
        if not active.any():
            break
 
    count[active] = max_iter   # pixels that never escaped
    return count


# Implementation 3 – Multiprocessing
# ─────────────────────────────────────────────────────────────────────────────
 
def _compute_row_chunk(args):
   
    row_start, row_end, xs, ys, max_iter = args
    local_ys = ys[row_start:row_end]
    band = np.empty((len(local_ys), len(xs)), dtype=np.int32)
    for i, y in enumerate(local_ys):
        for j, x in enumerate(xs):
            band[i, j] = mandelbrot_scalar(complex(x, y), max_iter)
    return row_start, band
 
 
def mandelbrot_multiprocessing(
    xmin: float, xmax: float,
    ymin: float, ymax: float,
    width: int, height: int,
    max_iter: int = 256,
    n_workers: int = 4,
) -> np.ndarray:
    
    import multiprocessing as mp
 
    xs = np.linspace(xmin, xmax, width)
    ys = np.linspace(ymin, ymax, height)
 
    # Build (row_start, row_end) chunks
    chunk_size = math.ceil(height / n_workers)
    chunks = [
        (i, min(i + chunk_size, height), xs, ys, max_iter)
        for i in range(0, height, chunk_size)
    ]
 
    result = np.empty((height, width), dtype=np.int32)
    with mp.Pool(processes=n_workers) as pool:
        for row_start, band in pool.map(_compute_row_chunk, chunks):
            result[row_start: row_start + len(band)] = band
    return result 

 # Implementation 4 – Dask
# ─────────────────────────────────────────────────────────────────────────────
 
def mandelbrot_dask(
    xmin: float, xmax: float,
    ymin: float, ymax: float,
    width: int, height: int,
    max_iter: int = 256,
    chunk_size: int = 256,
) -> np.ndarray:
    
    try:
        import dask
        import dask.array as da
    except ImportError:
        print("  [WARNING] Dask not installed – using NumPy vectorised")
        return mandelbrot_numpy(xmin, xmax, ymin, ymax, width, height, max_iter)
 
    xs = np.linspace(xmin, xmax, width)
    ys = np.linspace(ymin, ymax, height)
 
    @dask.delayed
    def _chunk(y_slice):
        ys_local = ys[y_slice]
        return mandelbrot_numpy.__wrapped__ \
            if hasattr(mandelbrot_numpy, '__wrapped__') else \
            _numpy_chunk(xs, ys_local, max_iter)
 
    def _numpy_chunk(xs_arr, ys_arr, mi):
        C  = xs_arr[np.newaxis, :] + 1j * ys_arr[:, np.newaxis]
        Z  = np.zeros_like(C)
        cnt = np.zeros(C.shape, dtype=np.int32)
        active = np.ones(C.shape, dtype=bool)
        for k in range(mi):
            Z[active] = Z[active] ** 2 + C[active]
            esc = active & (np.abs(Z) > 2.0)
            cnt[esc] = k + 1
            active &= ~esc
            if not active.any():
                break
        cnt[active] = mi
        return cnt
 
    slices = [slice(i, min(i + chunk_size, height))
              for i in range(0, height, chunk_size)]
 
    delayed_chunks = [dask.delayed(_numpy_chunk)(xs, ys[sl], max_iter)
                      for sl in slices]
    results = dask.compute(*delayed_chunks)
    return np.vstack(results)
# Benchmarking harness
# ─────────────────────────────────────────────────────────────────────────────
 
DEFAULT_PARAMS = dict(xmin=-2.5, xmax=1.0, ymin=-1.25, ymax=1.25)
SIZES = [(64, 64), (256, 256), (512, 512), (1024, 1024), (2048, 2048), (4096, 4096)]
 
 
def run_benchmarks(sizes=None, max_iter=256):
    if sizes is None:
        sizes = SIZES
 
    implementations = [
        ("Naive",             mandelbrot_naive),
        ("NumPy",             mandelbrot_numpy),
        ("Multiprocessing",   partial(mandelbrot_multiprocessing, n_workers=4)),
        ("Dask",              partial(mandelbrot_dask, chunk_size=128)),
    ]
 
    header = f"{'Impl':<20s}" + "".join(f"  {w}×{h}" for w, h in sizes)
    print("\n" + "=" * len(header))
    print(header)
    print("-" * len(header))
 
    for name, fn in implementations:
        row = f"{name:<20s}"
        for w, h in sizes:
            if name == "Naive" and w > 256:
                row += f"  {'skip':>8s}"
                continue
            t0 = time.perf_counter()
            fn(**DEFAULT_PARAMS, width=w, height=h, max_iter=max_iter)
            elapsed = time.perf_counter() - t0
            row += f"  {elapsed:>7.3f}s"
        print(row)
 
    print("=" * len(header))
     