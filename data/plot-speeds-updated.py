from statistics import variance
import numpy as np
from numpy import loadtxt, gradient
from matplotlib.pyplot import subplots, show

def g(u_nm1, dt):
    return u_nm1 * dt

def motion_model(index, v_com, time):
    dt = time[1:] - time[0:-1]
    dt = np.append(dt, 0)

    X = np.zeros(len(index))
    X[0] = 0
    w = np.zeros(len(index))

    # motion model
    for n in index[0:]:
        n = int(n)
        X[n] = X[n-1] + g(v_com[n-1], dt[n])
        w[n] = X[n] - X[n-1] - g(v_com[n-1], dt[n-1])

    var_w = variance(w)

    return X, var_w

# # Load data
# filename = 'data/training1.csv'
# data = loadtxt(filename, delimiter=',', skiprows=1)

# # Split into columns
# index, time, distance, velocity_command, raw_ir1, raw_ir2, raw_ir3, raw_ir4, \
#     sonar1, sonar2 = data.T

# v_est = gradient(distance, time)

# print(f'process noise variance: {var_w}\n')

# fig, axes = subplots(1)
# axes.plot(time, v_com, label='commanded speed')
# axes.plot(time, v_est, label='measured speed')
# axes.plot(time, X, label='motion model')
# axes.set_xlabel('Time')
# axes.set_ylabel('Speed (m/s)')
# axes.legend()

# fig2, axes2 = subplots(1)
# axes2.plot(time, w, label='process noise')
# axes2.set_xlabel('Time')
# axes2.set_ylabel('Speed (m/s)')
# axes2.legend()

# show()

def main():
    # Load data
    filename = 'data/training1.csv'
    data = loadtxt(filename, delimiter=',', skiprows=1)

    # Split into columns
    index, time, distance, velocity_command, raw_ir1, raw_ir2, raw_ir3, raw_ir4, \
        sonar1, sonar2 = data.T

    motion_modeled, motion_model_variance = motion_model(index, velocity_command, time)
