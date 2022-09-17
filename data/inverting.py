import numpy as np
from signal_plot import signal_plot
from matplotlib.pyplot import plot, show, figure

x0 = 5
Nx = 201
x = np.linspace(0, 10, Nx)

k1 = 10
k2 = 1

h = k1 / (k2 + x)

J = -k1 / (k2 + x0)**2

h2 = J * (x - x0) + k1 / (k2 + x0)

fig2 = figure()
plot(x, h, label='h')
plot(x, h2, label='h2')
fig2.legend()
show()