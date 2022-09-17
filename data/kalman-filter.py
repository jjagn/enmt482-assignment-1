from numpy import loadtxt, gradient
from statistics import stdev, variance
import numpy as np
from sensor_fitting import ir4_model, ir_model, linearised_ir_search_invert
from sensor_fitting import fit_sensor, linear_model, \
    inv_linear_model, inv_ir_model, inv_ir4_model, inv_ir_model_2_linearised,\
        ir_model_2
from sensor_fitting import ir_var
from matplotlib.pyplot import plot, show, figure
from numpy import zeros
from motion_model import motion_model, motion_model_iter

do_plot = False
rejection = True

# Load data
filename = 'training1.csv'
data = loadtxt(filename, delimiter=',', skiprows=1)

# Split into columns
# index, time, velocity_command, raw_ir1, raw_ir2, raw_ir3, raw_ir4, \
#     sonar1, sonar2 = data.T

index, time, distance, velocity_command, raw_ir1, raw_ir2, raw_ir3, raw_ir4, \
    sonar1, sonar2 = data.T

v_com = velocity_command

dt = time[1:] - time[0:-1]
# print(len(dt))
dt = np.append(dt, 0)
# print(len(dt))

robot_max_speed_estimate = 15 # let's assume the robot can't move faster than \
# 10 ms -> would probably be better do do this based off acceleration too
robot_max_acc_estimate = 10
# v_est = gradient(distance, time)

motion_modeled, motion_model_noise = motion_model(index, velocity_command, time)

# need sensor variances
# var_ir1, std_ir1, params_ir1 = fit_sensor('ir1', ir_model_2, 0.2, do_plot)
# print(var_ir1)
# print(std_ir1)
# print(params_ir1)

# var_ir2, std_ir2, params_ir2 = fit_sensor('ir2', ir_model_2, 0.2, do_plot)
# print(var_ir2)
# print(std_ir2)
# print(params_ir2)

# var_ir3, std_ir3, params_ir3 = fit_sensor('ir3', ir_model_2, 0.2, do_plot)
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

# var_sonar2, std_sonar2, params_sonar2 = fit_sensor('sonar2', linear_model, \
#     0.2, do_plot)

var_ir1, std_ir1, params_ir1 = fit_sensor('ir1', ir_model_2, 2, do_plot)


var_ir2, std_ir2, params_ir2 = fit_sensor('ir2', ir_model_2, 2, do_plot)


var_ir3, std_ir3, params_ir3 = fit_sensor('ir3', ir_model_2, 2, do_plot)


var_W = motion_model_noise# determined previously, probably wrong
# var_ir1 = 1e-3
# var_ir2 = 1e-3
# var_ir3 = 1e-3
# var_sonar1 = 1e-3
# var_W = 1e-3

length = len(index)
K = zeros(length)
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

# linearised sensor variances
var_ir1_est = zeros(length)
var_ir2_est = zeros(length)
var_ir3_est = zeros(length)

# sensor weights
weight_sonar1 = zeros(length)
weight_sonar2 = zeros(length)
weight_ir1 = zeros(length)
weight_ir2 = zeros(length)
weight_ir3 = zeros(length) 

for n in index[0:]:
    # convert n to an int to keep everyone happy
    n = int(n)

    # simple motion model
    x_hat_prior[n] = motion_model_iter(x_hat_post[n-1], v_com[n-1], dt[n-1])

    # Var_X is the variance of the motion model which increases over time
    var_x_hat_prior[n] = var_x_hat_post[n-1] + var_W

    previous_estimate = x_hat_prior[n-1] 

    Xir1[n] = linearised_ir_search_invert(previous_estimate, params_ir1, raw_ir1[n])

    print(f'Xir1 = {Xir1[n]}')

    Xir2[n] = linearised_ir_search_invert(previous_estimate, params_ir2, raw_ir2[n])

    print(f'Xir2 = {Xir2[n]}')

    Xir3[n] = linearised_ir_search_invert(previous_estimate, params_ir3, raw_ir3[n])

    # potentially experiment with some outlier rejection here?
    Xsonar1[n] = inv_linear_model(sonar1[n], params_sonar1[0], params_sonar1[1])
    # Xsonar2[n] = inv_linear_model(sonar2[n], params_sonar2[0], params_sonar2[1])

    furthest_possible_move = robot_max_speed_estimate * dt[n]
    # print(f'robot could not have moved further than {furthest_possible_move}')
    if rejection:
        if abs(Xsonar1[n] - x_hat_post[n-1]) > furthest_possible_move:
            print('it ran')
            print(f'n: {n}')
            Xsonar1[n] = x_hat_post[n-1]

        if abs(Xir1[n] - x_hat_post[n-1]) > furthest_possible_move:
            print('it ran')
            print(f'n: {n}')
            Xir1[n] = x_hat_post[n-1]

        if abs(Xir2[n] - x_hat_post[n-1]) > furthest_possible_move:
            print('it ran')
            print(f'n: {n}')
            Xir2[n] = x_hat_post[n-1]

        if abs(Xir3[n] - x_hat_post[n-1]) > furthest_possible_move:
            print('it ran')
            print(f'n: {n}')
            Xir3[n] = x_hat_post[n-1]

    # make ir variances greatly increase with range
    var_ir1_est[n] = ir_var(x_hat_post[n-1], var_ir1) / ((-params_ir1[0]/(params_ir1[1]+x_hat_post[n-1])**2+ params_ir1[2]))**2
    var_ir2_est[n] = ir_var(x_hat_post[n-1], var_ir2)/ ((-params_ir2[0]/(params_ir2[1]+x_hat_post[n-1])**2+ params_ir2[2]))**2
    var_ir3_est[n] = ir_var(x_hat_post[n-1], var_ir3) / ((-params_ir3[0]/(params_ir3[1]+x_hat_post[n-1])**2+ params_ir3[2]))**2

    # fuse the sensors here
    weight_sonar1[n] = 1 / var_sonar1
    weight_ir1[n] = 1 / var_ir1_est[n]
    weight_ir2[n] = 1 / var_ir2_est[n]
    weight_ir3[n] = 1 / var_ir3_est[n]

    # cap weights
    if weight_ir1[n] > 1e3:
        weight_ir1[n] = 1e3
    if weight_ir2[n] > 1e3:
        weight_ir2[n] = 1e3
    if weight_ir3[n] > 1e3:
        weight_ir3[n] = 1e3
    if weight_sonar1[n] > 1e3:
        weight_sonar1[n] = 1e3
    

 # fusing sensors
    x_hat[n] = (weight_sonar1[n] * Xsonar1[n]\
            + weight_ir1[n] + Xir1[n] + weight_ir2[n] * Xir2[n] + weight_ir3[n]\
            * Xir3[n]) / (weight_sonar1[n] + weight_ir1[n]\
            + weight_ir2[n] + weight_ir3[n])

    # var_x_prior will be the fused variance of the sensors as we add more
    var_x_hat[n] = 1 / (weight_sonar1[n] + weight_ir1[n] + weight_ir2[n] + weight_ir3[n]) 

    # KALMAN FILTER -> K = 0, trust the sensors
    K[n] = 1/var_x_hat[n] / (1/var_x_hat[n] + 1/var_x_hat_prior[n])
    x_hat_post[n] = K[n] * x_hat[n] + (1-K[n]) * x_hat_prior[n]

    var_x_hat_post[n] = 1/(1/var_x_hat[n] + 1/var_x_hat_prior[n])

# kalman gain broken

fig = figure()
plot(time, distance, label='distance')
# plot(time, Xir1, label='Xir1')
# plot(time, Xir2, label='Xir2')
# plot(time, Xir3, label='Xir3')
# plot(time, Xsonar1, label='Xsonar1')
# plot(time, Xsonar2, label='Xsonar2')
# plot(time, x_hat_blue, label='x hat BLUE')
plot(time, x_hat, label='fused sensor estimate')
plot(time, x_hat_prior, label='motion model')
plot(time, x_hat_post, label='final position estimate')
# plot(time, K, label='kalman gain')
fig.legend()

fig2 = figure()
plot(time, var_x_hat_prior, label='var x hat prior')
plot(time, var_x_hat, label='var x hat')
plot(time, var_x_hat_post, label='var x hat post')
plot(time, weight_sonar1, label='sonar 1 weight')
plot(time, weight_ir1, label='ir1 weight')
plot(time, weight_ir2, label='ir2 weight')
plot(time, weight_ir3, label='ir3 weight')
plot(time, K*1000, label='kalman gain')
fig2.legend()

fig3 = figure()
plot(time, Xsonar1, label='Xsonar1')
plot(time, Xir1, label='Xir1')
plot(time, Xir2, label='Xir2')
plot(time, Xir3, label='Xir3')
fig3.legend()

fig4 = figure()
plot(time, sonar1, label='raw sonar 1')
plot(time, raw_ir1, label='raw ir1')
plot(time, raw_ir2, label='raw ir2')
plot(time, raw_ir3, label='raw ir3')
fig4.legend()

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
