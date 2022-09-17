from numpy import loadtxt, gradient
from statistics import stdev, variance
import numpy as np
from sensor_fitting import ir4_model, ir_model
from sensor_fitting import fit_sensor, linear_model, \
    inv_linear_model, inv_ir_model, inv_ir4_model, inv_ir_model_2_linearised,\
        ir_model_2
from matplotlib.pyplot import plot, show, figure
from numpy import zeros
from motion_model import motion_model, motion_model_iter

do_plot = False

# Load data
filename = 'data/training2.csv'
data = loadtxt(filename, delimiter=',', skiprows=1)

# Split into columns
index, time, distance, velocity_command, raw_ir1, raw_ir2, raw_ir3, raw_ir4, \
    sonar1, sonar2 = data.T

var_ir1, std_ir1, params_ir1 = fit_sensor('ir1', ir_model_2, 2, do_plot)
var_ir2, std_ir2, params_ir2 = fit_sensor('ir2', ir_model_2, 2, do_plot)
var_ir3, std_ir3, params_ir3 = fit_sensor('ir3', ir_model_2, 2, do_plot)

