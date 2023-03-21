import numpy as np
import os

def get_sorted_counts(name):
    if name.endswith('vals.bin'):
        name = name.split('vals.bin')[0]

    sorted_vals_name = name + 'sorted_vals.bin'
    if os.path.isfile(sorted_vals_name):
        return np.fromfile(sorted_vals_name, dtype=np.int32)
    a = np.fromfile(name + 'vals.bin', dtype=np.int32)
    a[::-1].sort()
    a.tofile(sorted_vals_name)
    return a
