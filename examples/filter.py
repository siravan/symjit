import util

args = util.process_argv()

import numpy as np
from scipy import datasets
from scipy.ndimage import zoom, generic_filter
import matplotlib.pyplot as plt

ascent = zoom(datasets.ascent(), 0.5)

from sympy import symbols, Min, Max
from symjit import compile_func

maximum_filter_result = generic_filter(ascent, np.amax, [5, 5])


def custom_filter(image):
    return np.amax(image) - np.amin(image)


custom_filter_result = generic_filter(ascent, custom_filter, [5, 5])

X = symbols("X[0:25]")
f = compile_func(X, Max(*X) - Min(*X))
symjit_filter = f.callable_filter()
symjit_filter_result = generic_filter(ascent, custom_filter, [5, 5])

fig, axes = plt.subplots(2, 2, figsize=(9, 9))
plt.gray()  # show the filtered result in grayscale

axes[0, 0].set_axis_off()
axes[0, 0].imshow(ascent)
axes[0, 0].set_title("Original image")

axes[1, 0].set_axis_off()
axes[1, 0].imshow(maximum_filter_result)
axes[1, 0].set_title("Maximum filter, Kernel: 5x5")

axes[0, 1].set_axis_off()
axes[0, 1].imshow(custom_filter_result)
axes[0, 1].set_title("Custom filter, Kernel: 5x5")

axes[1, 1].set_axis_off()
axes[1, 1].imshow(symjit_filter_result)
axes[1, 1].set_title("Symjit filter, Kernel: 5x5")

fig.tight_layout()
plt.show()
