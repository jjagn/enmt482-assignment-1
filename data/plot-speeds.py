import numpy as np
from matplotlib.pyplot import subplots, show
import matplotlib.pyplot as pyplot

# Load data
filename = 'data/training2.csv'
data = np.loadtxt(filename, delimiter=',', skiprows=1)

# Split into columns
index, time, distance, velocity_command, raw_ir1, raw_ir2, raw_ir3, raw_ir4, \
    sonar1, sonar2 = data.T

speeds = []
prev_time = 0
prev_pos = distance[0]
for pos, time_entry in zip(distance, time):
    time_step = time_entry - prev_time
    print(time_step)
    speeds.append((pos-prev_pos)/time_step)
    prev_pos = pos
    prev_time = time_entry

fig, axes = subplots(2, 3)
fig.suptitle('Calibration data')

axes[0, 0].plot(distance, raw_ir1, '.', alpha=0.2)
axes[0, 0].set_title('IR1')

axes[0, 1].plot(distance, raw_ir2, '.', alpha=0.2)
axes[0, 1].set_title('IR2')

axes[0, 2].plot(distance, raw_ir3, '.', alpha=0.2)
axes[0, 2].set_title('IR3')

axes[1, 0].plot(distance, raw_ir4, '.', alpha=0.2)
axes[1, 0].set_title('IR4')

axes[1, 1].plot(distance, sonar1, '.', alpha=0.2)
axes[1, 1].set_title('Sonar1')

axes[1, 2].plot(distance, sonar2, '.', alpha=0.2)
axes[1, 2].set_title('Sonar2')

pyplot.figure()
pyplot.plot(time, speeds)
pyplot.plot(time, velocity_command)

# motion model
motion_modeled = []
velocity = 0
prev_time = 0
for command, time_entry in zip(velocity_command, time):
    time_step = time_entry - prev_time
    print(time_step)
    # ??? why does it need to be scaled?
    motion_modeled.append(command * time_step)
    prev_time = time_entry

pyplot.plot(time, motion_modeled)
show()
