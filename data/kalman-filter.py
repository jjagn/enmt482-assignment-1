from numpy import loadtxt, gradient
from statistics import stdev, variance
import numpy as np
from sensor_fitting import ir4_model, ir_model
from sensor_fitting import fit_sensor, linear_model, \
    inv_linear_model, inv_ir_model, inv_ir4_model
from matplotlib.pyplot import plot, show, figure
from numpy import zeros
from motion_model import motion_model, motion_model_iter

do_plot = False

# Load data
filename = 'data/training1.csv'
data = loadtxt(filename, delimiter=',', skiprows=1)

# Split into columns
index, time, distance, velocity_command, raw_ir1, raw_ir2, raw_ir3, raw_ir4, \
    sonar1, sonar2 = data.T

v_com = velocity_command

dt = time[1:] - time[0:-1]
# print(len(dt))
dt = np.append(dt, 0)
# print(len(dt))

# v_est = gradient(distance, time)

motion_modeled, motion_model_noise = motion_model(index, velocity_command, time)

std_W = stdev(motion_modeled)
var_W = variance(motion_modeled)

# need sensor variances
# var_ir1, std_ir1, params_ir1 = fit_sensor('ir1', ir_model, 0.2, do_plot)
# print(var_ir1)
# print(std_ir1)
# print(params_ir1)

# var_ir2, std_ir2, params_ir2 = fit_sensor('ir2', ir_model, 0.2, do_plot)
# print(var_ir2)
# print(std_ir2)
# print(params_ir2)

# var_ir3, std_ir3, params_ir3 = fit_sensor('ir3', ir_model, 0.2, do_plot)
# print(var_ir3)
# print(std_ir3)
# print(params_ir3)

# for now don't bother using ir4 since we can't get a nice model out of it
# var_ir4, std_ir4, params_ir4 = fit_sensor('ir4', ir4_model, 0.2, False)
# print(var_ir4)
# print(std_ir4)
# print(params_ir4)

var_sonar1, std_sonar1, params_sonar1 = fit_sensor('sonar1', linear_model, \
    0.2, do_plot)
print(var_sonar1)
print(std_sonar1)
print(params_sonar1)

var_sonar2, std_sonar2, params_sonar2 = fit_sensor('sonar2', linear_model, \
    0.2, do_plot)
print(var_sonar2)
print(std_sonar2)
print(params_sonar2)

length = len(index)
K = zeros(length)
var_W = motion_model_noise # determined previously, probably wrong
Xir1 = zeros(length)
Xir2 = zeros(length)
Xir3 = zeros(length)
Xsonar1 = zeros(length)
Xsonar2 = zeros(length)

# xs
x_hat_prior = zeros(length)
x_hat = zeros(length)
x_hat_post = zeros(length)
x_hat_blue = zeros(length)

# variances
var_x_hat_prior = zeros(length)
var_x_hat = zeros(length)
var_x_hat_post = zeros(length)

# sensor weights
weight_sonar1 = zeros(length)
weight_sonar2 = zeros(length)

var_sonar1 = 0.4
var_sonar2 = 0.4

for n in index[0:]:
    # convert n to an int to keep everyone happy
    n = int(n)

    # simple motion model
    x_hat_prior[n] = motion_model_iter(x_hat_prior[n-1], v_com[n-1], dt[n-1])

    # Var_X is the variance of the motion model which increases over time
    var_x_hat_prior[n] = var_x_hat_prior[n-1] + var_W

    # mean_X_prior = mean_X_posterior + v_com[n] * dt[n]
    # var_X_prior = var_X_posterior + var_W

    # # ML estimate of position from measurement using sensor model
    # # also will need to linearise the nonlinear models for the IR sensors?
    # Xir1[n] = inv_ir_model(raw_ir1[n], params_ir1[0], params_ir1[1], \
    #     params_ir1[2], previous_estimate)

    # print(f'Xir1 = {Xir1[n]}')

    # Xir2[n] = inv_ir_model(raw_ir2[n], params_ir2[0], params_ir2[1], \
    #     params_ir2[2], previous_estimate)

    # print(f'Xir2 = {Xir2[n]}')

    # Xir3[n] = inv_ir_model(raw_ir3[n], params_ir3[0], params_ir3[1], \
    #     params_ir3[2], previous_estimate)

    # print(f'Xir3 = {Xir3[n]}')

    # # Xir4 = inv_ir4_model(raw_ir4[n], params_ir4[0], params_ir4[1],\
    # #      params_ir4[2], params_ir4[3]) 

    # # print(f'Xir4 = {Xir4}')

    # potentially experiment with some outlier rejection here?
    Xsonar1[n] = inv_linear_model(sonar1[n], params_sonar1[0], params_sonar1[1])

    # print(f'Xsonar1 = {Xsonar1[n]}')

    Xsonar2[n] = inv_linear_model(sonar2[n], params_sonar2[0], params_sonar2[1])

    # fuse the sensors here
    # X_hat[n] = Xsonar2[n]
    weight_sonar1[n] = 1 / var_sonar1
    weight_sonar2[n] = 1 / var_sonar2

    # fusing sonar1 and sonar2
    x_hat[n] = (weight_sonar1[n] * Xsonar1[n] + weight_sonar2[n] * Xsonar2[n])/\
        (weight_sonar1[n] + weight_sonar2[n])

    # var_x_prior will be the fused variance of the sensors as we add more
    var_x_hat[n] = 1 / (weight_sonar1[n] + weight_sonar2[n])

    # KALMAN FILTER
    K[n] = 1/var_x_hat[n] / (1/var_x_hat[n] + 1/var_x_hat_prior[n])
    x_hat_post[n] = K[n] * x_hat[n] + (1-K[n]) * x_hat_prior[n]

    var_x_hat_post = 1/(1/var_x_hat + 1/var_x_hat_prior)

# kalman gain broken

fig = figure()
# plot(time, distance, label='distance')
# plot(time, Xir1, label='Xir1')
# plot(time, Xir2, label='Xir2')
# plot(time, Xir3, label='Xir3')
plot(time, Xsonar1, label='Xsonar1')
plot(time, Xsonar2, label='Xsonar2')
# plot(time, x_hat_blue, label='x hat BLUE')
plot(time, x_hat, label='fused sensor estimate')
plot(time, x_hat_prior, label='motion model')
plot(time, x_hat_post, label='final position estimate')
plot(time, K, label='kalman gain')
fig.legend()

fig2 = figure()
plot(time, var_x_hat_prior, label='var x hat prior')
plot(time, var_x_hat, label='var x hat')
plot(time, var_x_hat_post, label='var x hat post')
plot(time, weight_sonar1, label='sonar 1 weight')
plot(time, weight_sonar2, label='sonar 2 weight')
plot(time, K, label='kalman gain')
fig2.legend()
show()
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