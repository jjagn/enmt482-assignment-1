from statistics import variance
import numpy as np
from numpy import loadtxt, gradient
from matplotlib.pyplot import subplots, show

# Load data
filename = 'data/training1.csv'
data = loadtxt(filename, delimiter=',', skiprows=1)

# Split into columns
index, time, distance, velocity_command, raw_ir1, raw_ir2, raw_ir3, raw_ir4, \
    sonar1, sonar2 = data.T

v_com = velocity_command

dt = time[1:] - time[0:-1]
print(dt)
print(len(dt))
dt = np.append(dt, 0)
print(len(dt))

v_est = gradient(distance, time)

motion_modeled = v_com / dt

process_noise_error = v_est - motion_modeled
var_w = variance(motion_modeled)
print(f'process noise variance: {var_w}\n')
# process_noise_error = np.append(process_noise_error, 0)

# # motion model
# motion_modeled = []
# velocity = 0
# prev_time = 0
# for command, time_entry in zip(v_com, time):
#     time_step = time_entry - prev_time
#     print(time_step)
#     # ??? why does it need to be scaled?
#     motion_modeled.append(command * time_step)
#     prev_time = time_entry

fig, axes = subplots(1)
axes.plot(time, v_com, label='commanded speed')
axes.plot(time, v_est, label='measured speed')
axes.plot(time, motion_modeled, label='motion model')
axes.set_xlabel('Time')
axes.set_ylabel('Speed (m/s)')
axes.legend()

fig2, axes2 = subplots(1)
axes2.plot(time, process_noise_error, label='process noise')
axes2.set_xlabel('Time')
axes2.set_ylabel('Speed (m/s)')
axes2.legend()

show()
