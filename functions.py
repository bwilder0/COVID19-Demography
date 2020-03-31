import numpy as np
from numba import jit
import random

@jit
def categorical_sample(p):
    threshold = np.random.rand()
    current = 0
    for i in range(p.shape[0]):
        current += p[i]
        if current > threshold:
            return i
@jit
#def threshold_exponential(mean):
#    return 1 + np.round(np.random.exponential(mean-1))
def threshold_exponential(mean):
    return np.round(np.random.exponential(mean))

@jit
def threshold_log_normal(mean, sigma):
    x = np.random.lognormal(mean, sigma)
    if x <= 0:
        return 1
    else:
        return np.round(x)
