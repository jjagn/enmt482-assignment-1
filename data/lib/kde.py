# Michael P. Hayes UCECE, Copyright 2018--2019

from numpy import zeros_like, ones_like, exp, sqrt, pi, average, trapz

class KDE(object):
    """1-D Kernel density estimation (KDE)"""

    def __init__(self, samples, weights=None):

        self.samples = samples
        self.weights = weights
        if self.weights is None:
            self.weights = ones_like(self.samples)

        Ns = len(self.samples)

        mean = average(self.samples, weights=self.weights)
        # Note, this is a biased estimate of the variance
        var = average((self.samples - mean)**2, weights=self.weights)
        sd = sqrt(var)

        # Use Silverman's approximation for std dev of kernel.  Note,
        # this can yield widely inaccurate estimates when the density
        # is not close to being normal.
        if sd == 0.0:
            sd = 1e-6
        self.sigmak = 1.06 * sd * Ns ** (-1.0 / 5)

            
    def estimate(self, values):
        """`values' is an array of values to estimate PDF at."""

        pdf = zeros_like(values)

        Ns = len(self.samples)        

        # Could perform convolution with FFT.
        for m in range(Ns):
            pdf += self.weights[m] * exp(-(values - self.samples[m])**2 / (2 * self.sigmak**2)) / (self.sigmak * sqrt(2 * pi)) / Ns

        pdf /= trapz(pdf, values)
        return pdf
