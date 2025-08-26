import matplotlib.pyplot as plt

fig, axes = plt.subplots(nrows=3, ncols=4)

plt.sca(axes[0][0])
import mandelbrot

plt.sca(axes[0][1])
import mandelbrot2

plt.sca(axes[0][2])
import mandelbrot3

plt.sca(axes[0][3])
import lorenz

plt.sca(axes[1][0])
import airy

plt.sca(axes[1][1])
import trig

plt.sca(axes[1][2])
import lotka_volterra

plt.sca(axes[1][3])
import van_der_pol

plt.sca(axes[2][0])
import beeler

plt.sca(axes[2][1])
import tentusscher

plt.sca(axes[2][2])
import newton

plt.sca(axes[2][3])
import normal

plt.show()
