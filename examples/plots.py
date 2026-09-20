import matplotlib.pyplot as plt

fig, axes = plt.subplots(nrows=3, ncols=4)

print("plotting mandelbrot:\t")
plt.sca(axes[0][0])
import mandelbrot

print("plotting mandelbrot2:\t")
plt.sca(axes[0][1])
import mandelbrot2

print("plotting mandelbrot3:\t")
plt.sca(axes[0][2])
import mandelbrot3

print("plotting lorenz:\t")
plt.sca(axes[0][3])
import lorenz

print("plotting airy:\t")
plt.sca(axes[1][0])
import airy

print("plotting trig:\t")
plt.sca(axes[1][1])
import trig

print("plotting lotka_volterra:\t")
plt.sca(axes[1][2])
import lotka_volterra

print("plotting van_der_pol:\t")
plt.sca(axes[1][3])
import van_der_pol

print("plotting beeler:\t")
plt.sca(axes[2][0])
import beeler

print("plotting tentusscher:\t")
plt.sca(axes[2][1])
import tentusscher

print("plotting newton:\t")
plt.sca(axes[2][2])
import newton

print("plotting normal:\t")
plt.sca(axes[2][3])
import normal

plt.show()
