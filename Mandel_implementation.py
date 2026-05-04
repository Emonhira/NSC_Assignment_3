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
