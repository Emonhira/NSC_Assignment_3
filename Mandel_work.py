 
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

