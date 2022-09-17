# Michael P. Hayes UCECE, Copyright 2018--2019

import numpy as np

def rect(t):
    """Rectangle function"""

    return (abs(t) <= 0.5) * 1


def tri(t):
    """Triangle function"""

    return (1 - abs(t)) * rect(t / 2)


def triangle_wave(t, period=1.0):
    """Triangle wave with symmetry of cosine"""

    t /= period
    cycles = np.round(t)
    y = t - cycles
    return 2 * (tri(2 * y) - 0.5)


def sinc(t):
    """Normalised cardinal sine function sin(pi t) / (pi t)"""
    return np.sinc(t)


def gauss(t, mu=0, sigma=1):
    """Gaussian function"""

    return np.exp(-0.5 * ((t - mu) / sigma)**2) / (np.sqrt(2 * np.pi) * sigma)


def trap(t, alpha):
    """Trapezoid function; alpha is the normalised rise/fall time.
    trap(t, 0) = rect(t); trap(t, 1) = tri(t).  The top of the trapezoid
    has width 1 - alpha and the bottom has width 1 + alpha."""

    if alpha == 0:
        return rect(t)
    if alpha == 1:
        return tri(t)

    w = rect(t / (1 + alpha)) - rect(t / (1 - alpha))

    return rect(t / (1 - alpha)) + w * (1 - (abs(t) - 0.5 * (1 - alpha)) / alpha)


def trap2(t, top, base):
    """Trapezoid function; top is the width of the top, base is the width
    of the base"""

    T = 0.5 * (base + top)
    alpha = (base - top) / (base + top)
    return trap(t / T, alpha)


def mgauss(vx, vmu, K):

    N = vx.shape[-1]

    Kinv = np.linalg.inv(K)
    Kdet = np.linalg.det(K)

    arg = np.dot(np.dot((vx - vmu).T, Kinv), (vx - vmu))
    
    return np.exp(-0.5 * arg) / np.sqrt((2 * np.pi) ** N * Kdet)


def mgauss2(x, y, vmu, vsigma, rho):

    K = np.array(((vsigma[0]**2, rho * vsigma[0] * vsigma[1]), (rho * vsigma[0] * vsigma[1], vsigma[1]**2)))
        
    mg = np.zeros((len(y), len(x)))
    
    for n, y1 in enumerate(y):
        for m, x1 in enumerate(x):
            
            vx = np.array((x1, y1))
            mg[n, m] = mgauss(vx, vmu, K)

    return mg


distributions = ['gaussian', 'uniform']


def pdf(x, muX, sigmaX, distribution):

    if distribution == 'gaussian':
        return gauss(x, muX, sigmaX)
    elif distribution == 'uniform':
        xmin = muX - np.sqrt(12) * sigmaX / 2
        xmax = muX + np.sqrt(12) * sigmaX / 2
        u = 1.0 * ((x >= xmin) & (x <= xmax))
        u /= np.trapz(u, x)
        return u
    raise ValueError('Unknown distribution %s' % distribution)


###############################################################################
# Functions for working with angles (in radians)

def wrapto2pi(angle):
    """Convert angle into range [0, 2 * pi)."""
    return angle % (2 * np.pi)


def wraptopi(angle):
    """Convert angle into range [-pi, pi)."""
    return wrapto2pi(angle + np.pi) - np.pi


def angle_difference(angle1, angle2):
    """Return principal difference angle, angle2 - angle1  with return value in range [-pi, pi)."""
    return ((((angle2 - angle1) % (2 * np.pi)) + (3 * np.pi)) % (2 * np.pi)) - np.pi


def abs_angle_difference(angle1, angle2):

    d = angle_difference(angle1, angle2)
    if (d < 0):
        d += np.pi * 2
    return
    
