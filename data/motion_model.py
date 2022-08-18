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

def motion_model_iter(Xn_last, velocity_command_last, dt_last):
    return Xn_last + velocity_command_last * dt_last