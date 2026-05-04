
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