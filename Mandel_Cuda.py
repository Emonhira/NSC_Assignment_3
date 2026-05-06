
import time
import argparse
import numpy as np
 
try:
    from numba import cuda
    import numba
    CUDA_AVAILABLE = cuda.is_available()
except ImportError:
    CUDA_AVAILABLE = False
    print("[WARNING] Numba not installed. CUDA kernels cannot run.")

if CUDA_AVAILABLE:
    @cuda.jit
    def mandelbrot_kernel(out, xmin, xmax, ymin, ymax, max_iter):
       
        row, col = cuda.grid(2)
        H, W = out.shape
        if row >= H or col >= W:      # guard against out-of-bounds threads
            return
 
        # Map pixel to complex plane
        x = xmin + col * (xmax - xmin) / (W - 1)
        y = ymin + row * (ymax - ymin) / (H - 1)
        cr, ci = x, y
 
        zr = zi = 0.0
        count = 0
        for _ in range(max_iter):
            zr2 = zr * zr
            zi2 = zi * zi
            if zr2 + zi2 > 4.0:
                break
            zi = 2.0 * zr * zi + ci
            zr = zr2 - zi2 + cr
            count += 1
 
        out[row, col] = count
 
 
 
    BLOCK_SMEM = (16, 16)                      # fixed for the smem kernel
 
    @cuda.jit
    def mandelbrot_kernel_smem(out, block_means, xmin, xmax, ymin, ymax, max_iter):
        
        BH, BW = cuda.blockDim.x, cuda.blockDim.y      # block height, width
        tx, ty = cuda.threadIdx.x, cuda.threadIdx.y
        row, col = cuda.grid(2)
        H, W = out.shape
 
        # Shared memory tile (one int32 per thread)
        smem = cuda.shared.array(shape=(16, 16), dtype=numba.int32)
 
        count = 0
        if row < H and col < W:
            x  = xmin + col * (xmax - xmin) / (W - 1)
            y  = ymin + row * (ymax - ymin) / (H - 1)
            cr, ci = x, y
            zr = zi = 0.0
            for _ in range(max_iter):
                zr2 = zr * zr
                zi2 = zi * zi
                if zr2 + zi2 > 4.0:
                    break
                zi = 2.0 * zr * zi + ci
                zr = zr2 - zi2 + cr
                count += 1
            out[row, col] = count
 
        smem[tx, ty] = count
        cuda.syncthreads()          # ← required before reading neighbours
 
        # Simple linear reduction (sufficient for demonstration)
        if tx == 0 and ty == 0:
            total = numba.int32(0)
            for r in range(BH):
                for c in range(BW):
                    total += smem[r, c]
            bx = cuda.blockIdx.x
            by = cuda.blockIdx.y
            block_means[bx, by] = total / (BH * BW)
 
 

 
def run_gpu(
    width: int, height: int,
    xmin=-2.5, xmax=1.0, ymin=-1.25, ymax=1.25,
    max_iter: int = 256,
    threads_per_block=(16, 16),
    include_transfer: bool = True,
) -> dict:
    
    if not CUDA_AVAILABLE:
        raise RuntimeError("CUDA not available")
 
    tbp = threads_per_block
    bpg = (
        (height + tbp[0] - 1) // tbp[0],
        (width  + tbp[1] - 1) // tbp[1],
    )
 
    # ── Warm-up (compile + run tiny kernel) ─────────────────────────────────
    _dummy = cuda.device_array((2, 2), dtype=np.int32)
    mandelbrot_kernel[(1, 1), (2, 2)](_dummy, xmin, xmax, ymin, ymax, max_iter)
    cuda.synchronize()
 
    # ── Timed run ────────────────────────────────────────────────────────────
    t_transfer_start = time.perf_counter()
 
    out_d = cuda.device_array((height, width), dtype=np.int32)
 
    # CUDA event timing (accurate async GPU measurement)
    start_event = cuda.event(timing=True)
    end_event   = cuda.event(timing=True)
 
    start_event.record()
    mandelbrot_kernel[bpg, tbp](out_d, xmin, xmax, ymin, ymax, max_iter)
    end_event.record()
    end_event.synchronize()                         # blocks until GPU done
 
    kernel_ms = cuda.event_elapsed_time(start_event, end_event)
 
    image = out_d.copy_to_host()                    # PCIe transfer back
    t_transfer_end = time.perf_counter()
 
    return {
        "image":            image,
        "time_kernel":      kernel_ms / 1000.0,
        "time_total":       t_transfer_end - t_transfer_start,
        "threads_per_block": tbp,
    }
 
 

 
BLOCK_SIZES = [
    (4,  4),    #   16 threads – sub-warp (bad occupancy)
    (8,  8),    #   64 threads – 2 warps
    (8,  4),    #   32 threads – 1 warp (non-square)
    (16, 8),    #  128 threads – 4 warps
    (16, 16),   #  256 threads – 8 warps  ← typical sweet-spot
    (32, 8),    #  256 threads – 8 warps  (wider)
    (32, 16),   #  512 threads – 16 warps
    (32, 32),   # 1024 threads – 32 warps (hardware maximum on most GPUs)
]
 
 
def block_size_sweep(width=2048, height=2048, max_iter=256, repeats=3):
    
    if not CUDA_AVAILABLE:
        print("  CUDA not available – skipping block-size sweep")
        return []
 
    print(f"\n── Block-size sweep  ({width}×{height}, max_iter={max_iter}) ──────────")
    print(f"  {'Block':>12s}  {'Threads':>8s}  {'Warp mult':>9s}  "
          f"{'Kernel (s)':>12s}  {'Total (s)':>12s}")
    print("  " + "-" * 64)
 
    results = []
    for tbp in BLOCK_SIZES:
        total_threads = tbp[0] * tbp[1]
        is_warp_mult  = "yes" if total_threads % 32 == 0 else "NO "
 
        times_k = []
        times_t = []
        for _ in range(repeats):
            r = run_gpu(width, height, max_iter=max_iter, threads_per_block=tbp)
            times_k.append(r["time_kernel"])
            times_t.append(r["time_total"])
 
        mk = min(times_k)
        mt = min(times_t)
        print(f"  {str(tbp):>12s}  {total_threads:>8d}  {is_warp_mult:>9s}  "
              f"{mk:>12.4f}  {mt:>12.4f}")
        results.append({"block": tbp, "kernel_s": mk, "total_s": mt})
 
    best = min(results, key=lambda x: x["kernel_s"])
    print(f"\n  Best block size: {best['block']}  "
          f"(kernel {best['kernel_s']:.4f}s)")
    return results
 
 

 
def run_full_benchmark(sizes=None, max_iter=256):
    
    from Mandel_implementation import (
        mandelbrot_numpy,
        mandelbrot_multiprocessing,
    )
    from functools import partial
 
    if sizes is None:
        sizes = [64, 128, 256, 512, 1024, 2048, 4096, 8192]
 
    params = dict(xmin=-2.5, xmax=1.0, ymin=-1.25, ymax=1.25)
 
    print(f"\n{'Size':>6s}  {'NumPy (s)':>12s}  {'MP-4 (s)':>12s}  "
          f"{'GPU kernel (s)':>14s}  {'GPU total (s)':>14s}  "
          f"{'Speedup NP':>12s}  {'Speedup MP':>12s}")
    print("-" * 92)
 
    for sz in sizes:
        w = h = sz
        results = {}
 
        # NumPy
        t0 = time.perf_counter()
        mandelbrot_numpy(**params, width=w, height=h, max_iter=max_iter)
        results["numpy"] = time.perf_counter() - t0
 
        # Multiprocessing (skip for large sizes to save time)
        if sz <= 2048:
            t0 = time.perf_counter()
            mandelbrot_multiprocessing(**params, width=w, height=h,
                                       max_iter=max_iter, n_workers=4)
            results["mp"] = time.perf_counter() - t0
        else:
            results["mp"] = float("nan")
 
        # GPU
        if CUDA_AVAILABLE:
            r = run_gpu(w, h, **params, max_iter=max_iter)
            results["gpu_k"] = r["time_kernel"]
            results["gpu_t"] = r["time_total"]
            su_np = results["numpy"] / results["gpu_k"]
            su_mp = results["mp"]    / results["gpu_k"] if not np.isnan(results["mp"]) else float("nan")
        else:
            results["gpu_k"] = results["gpu_t"] = float("nan")
            su_np = su_mp = float("nan")
 
        def fmt(v):
            return f"{v:>12.4f}" if not np.isnan(v) else f"{'n/a':>12s}"
 
        print(f"{sz:>6d}  {fmt(results['numpy'])}  {fmt(results['mp'])}  "
              f"{fmt(results['gpu_k']):>14s}  {fmt(results['gpu_t']):>14s}  "
              f"{su_np:>12.1f}x  {su_mp:>12.1f}x"
              if not (np.isnan(su_np) or np.isnan(su_mp))
              else f"{sz:>6d}  {fmt(results['numpy'])}  {fmt(results['mp'])}  "
                   f"{'n/a':>14s}  {'n/a':>14s}  {'n/a':>12s}  {'n/a':>12s}")
 
 
# ─────────────────────────────────────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────────────────────────────────────
 
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Mandelbrot CUDA benchmark")
    parser.add_argument("--size",       type=int, default=None,
                        help="Single resolution to test (e.g. 2048)")
    parser.add_argument("--blocks-only", action="store_true",
                        help="Run block-size sweep only")
    parser.add_argument("--max-iter",   type=int, default=256)
    args = parser.parse_args()
 
    if not CUDA_AVAILABLE:
        print("No CUDA device found – please run on a GPU machine.")
    else:
        if args.blocks_only:
            sz = args.size or 2048
            block_size_sweep(sz, sz, max_iter=args.max_iter)
        elif args.size:
            r = run_gpu(args.size, args.size, max_iter=args.max_iter)
            print(f"  {args.size}×{args.size}  "
                  f"kernel={r['time_kernel']:.4f}s  "
                  f"total={r['time_total']:.4f}s")
        else:
            block_size_sweep(max_iter=args.max_iter)
            run_full_benchmark(max_iter=args.max_iter)
