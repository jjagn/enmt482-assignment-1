from numpy import loadtxt, gradient
from statistics import stdev, variance
import numpy as np
from sensor_fitting import ir4_model, ir_model
from sensor_fitting import fit_sensor, linear_model, \
    inv_linear_model, inv_ir_model, inv_ir4_model

# Load data
filename = 'data/training1.csv'
data = loadtxt(filename, delimiter=',', skiprows=1)

# Split into columns
index, time, distance, velocity_command, raw_ir1, raw_ir2, raw_ir3, raw_ir4, \
    sonar1, sonar2 = data.T

v_com = velocity_command

dt = time[1:] - time[0:-1]
print(len(dt))
dt = np.append(dt, 0)
print(len(dt))

v_est = gradient(distance, time)

motion_modeled = v_com * dt * 15 #???

process_noise_error = v_est - motion_modeled

std_W = stdev(motion_modeled)
var_W = variance(motion_modeled)

# need sensor variances
var_ir1, std_ir1, params_ir1 = fit_sensor('ir1', ir_model, 0.2, False)
print(var_ir1)
print(std_ir1)
print(params_ir1)

var_ir2, std_ir2, params_ir2 = fit_sensor('ir2', ir_model, 0.2, False)
print(var_ir2)
print(std_ir2)
print(params_ir2)

var_ir3, std_ir3, params_ir3 = fit_sensor('ir3', ir_model, 0.2, False)
print(var_ir3)
print(std_ir3)
print(params_ir3)

var_ir4, std_ir4, params_ir4 = fit_sensor('ir4', ir4_model, 0.2, False)
print(var_ir4)
print(std_ir4)
print(params_ir4)

var_sonar1, std_sonar1, params_sonar1 = fit_sensor('sonar1', linear_model, \
    0.2, False)
print(var_sonar1)
print(std_sonar1)
print(params_sonar1)

var_sonar2, std_sonar2, params_sonar2 = fit_sensor('sonar2', linear_model, \
    0.2, False)
print(var_sonar2)
print(std_sonar2)
print(params_sonar2)


print(f'process noise variance: {var_W}\n')

mean_X_posterior = distance[0]
var_X_posterior = 0.01
previous_estimate = 0

for n in index:
    n = int(n)
    mean_X_prior = mean_X_posterior + v_com[n] * dt[n]
    var_X_prior = var_X_posterior + var_W

    # ML estimate of position from measurement using sensor model
    # this is wrong, need to invert the sensor models
    # also will need to linearise the nonlinear models for the IR sensors?
    Xir1 = inv_ir_model(raw_ir1[n], params_ir1[0], params_ir1[1], \
        params_ir1[2], previous_estimate)

    print(f'Xir1 = {Xir1}')

    Xir2 = inv_ir_model(raw_ir2[n], params_ir2[0], params_ir2[1], \
        params_ir2[2], previous_estimate)

    print(f'Xir2 = {Xir2}')

    Xir3 = inv_ir_model(raw_ir3[n], params_ir3[0], params_ir3[1], \
        params_ir3[2], previous_estimate)

    print(f'Xir3 = {Xir3}')

    Xir4 = inv_ir4_model(raw_ir4[n], params_ir4[0], params_ir4[1],\
         params_ir4[2], params_ir4[3]) 

    print(f'Xir4 = {Xir4}')

    Xsonar1 = inv_linear_model(sonar1[n], params_sonar1[0], params_sonar1[1])

    print(f'Xsonar1 = {Xsonar1}')

    Xsonar2 = inv_linear_model(sonar2[n], params_sonar2[0], params_sonar2[1])


    # returning a NAN every time
    x_hat_blue = (1/var_ir1 * Xir1 + 1/var_ir2 * Xir2 + 1/var_ir3 * Xir3 + \
        1/var_ir4 * Xir4 + 1/var_sonar1 * Xsonar1 + 1/var_sonar2 * Xsonar2)\
            /(1/var_ir2 + 1/var_ir2 + 1/var_ir3 + 1/var_ir4 + 1/var_sonar1 \
                + 1/var_sonar2)
    print(f'estimated position: {x_hat_blue}')

# 1. predict the robot's position using the previous estimated position and your motion model

# 2. Determine the variance of the robot's predicted position

# 3 invert sensor model h(x) so that given a measurement z you can have an estimate x_hat for x. may need to use interpolation to find x_hat given z. could also use a bisection algorithm h(x) - z to find the root

# 4. determine sensor noise variance at the current best estimate for z

# 5. convert the noise variance into variance in terms of estimator X. this requires linearisation for non-linear sensor models

# 6. combine the estimates from each sensor using a BLUE. plot the weights for eachs sensor at each time step. combine the prediction from the motion model with the estimates from the sensors using a BLUE. 

# 7. determine the variance of the BLUE

# hints:
# test the filter with just the motion model
# test filter with the motion model and best sensor and plot the BLUE weights
# test the filter with the motion model and the two best sensors