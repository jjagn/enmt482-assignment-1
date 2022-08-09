from scipy.optimize import curve_fit
import numpy as np
from matplotlib.pyplot import subplots, show


def model(x, a, b, c, d):
    return -a * np.log(b * x + c) + d


# Load data
filename = "calibration.csv"
data = np.loadtxt(filename, delimiter=',', skiprows=1)

# fuse IR4 and Sonar1

# load data x, z
index, time, distance, velocity_command, raw_ir1, raw_ir2, raw_ir3, raw_ir4, \
    sonar1, sonar2 = data.T

z = raw_ir3

params, cov = curve_fit(model, distance, z)

print(params)

zfit = model(distance, *params)

fig, axes = subplots(1, 3)
fig.suptitle("curve fit")

axes[0].plot(distance, zfit, '.', alpha=0.2)
axes[0].set_title("fitted")

axes[1].plot(distance, z, '.', alpha=0.2)
axes[1].set_title("original")

z_error = z - zfit

axes[2].plot(distance, z_error, '.', alpha=0.2)
axes[2].set_title("error")

show()
