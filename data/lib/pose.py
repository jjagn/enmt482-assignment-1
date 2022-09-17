# Michael P. Hayes UCECE, Copyright 2018--2019

import numpy as np

class Pose(object):

    def __init__(self, x=0, y=0, theta=0):

        self.x = x
        self.y = y
        self.theta = theta

    def draw_axes(self, axes, linestyle='-'):

        opt = {'head_width': 0.15, 'head_length': 0.15, 'width': 0.05,
               'length_includes_head': True}

        xdx = np.cos(self.theta)
        xdy = np.sin(self.theta)

        ydx = np.cos(self.theta + np.pi / 2)
        ydy = np.sin(self.theta + np.pi / 2)        
        
        axes.arrow(self.x, self.y, xdx, xdy, **opt, color='red',
                   linestyle=linestyle)
        axes.arrow(self.x, self.y, ydx, ydy, **opt, color='green',
                   linestyle=linestyle)                        
        
