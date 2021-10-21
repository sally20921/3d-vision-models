import numpy as np
import os, imageio

def normalize(x):
    return x / np.linalg.norm(x)


