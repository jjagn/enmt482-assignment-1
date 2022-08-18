from statistics import stdev, variance, mean
from scipy.optimize import curve_fit
from scipy.special import lambertw
import numpy as np
from matplotlib.pyplot import subplots, show

do_plot = True

# superpose functions together
# taylor series
# iteratively trim

def log_model(x, a, b, c, d):
    return -a * np.log(b * x + c) + d


def ir_model(x, a, b, c):
    return 1/(x + a) + b * x + c


def linear_model(x, a, b):
    return a * x + b


def ir4_model(x, a, b):
    return (np.log(a * x)) / (x * x) + b


def quad_model(x, a, b, c):
    return a * x + b * x * x + c


def cubic_model(x, a, b, c, d):
    return a * (x * x * x) + b * (x * x) + c * (x) + d


def inv_linear_model(z, a, b):
    return (z - b) / a


def inv_log_model(z, a, b, c, d):
    return 1 / b * (np.exp((z - d) / - a) - c)


def inv_cubic_model(x, a, b, c, d):
    p = -b / (3 * a)
    q = p * p * p + (b * c - 3 * a * d) / (6 * a * a)
    r = c / (3 * a)
    x = (q + (q * q + (r - p * p) ** 3) ** (1/2)) ** (1/3) + \
        (q - (q * q + (r - p * p) ** 3) ** (1/2)) ** (1/3) + p
    return x


def inv_ir_model(x, a, b, c, est):
    comp1 = -(c - x + a*b + (a**2*b**2 - 2*a*b*c + 2*a*b*x - 4*b + c**2 - 2*c*x + x**2)**(1/2))/(2*b)
    err1 = abs(comp1-est)
    comp2 = -(c - x + a*b - (a**2*b**2 - 2*a*b*c + 2*a*b*x - 4*b + c**2 - 2*c*x + x**2)**(1/2))/(2*b)
    err2 = abs(comp2-est)

    if err1 > err2:
        return comp2
    else:
        return comp1    

def inv_ir4_model(x, a, b):
    return np.exp(-lambertw(0, (2*b - 2*x)/a^2)/2)/a

def fit_sensor(sensor='ir1', model=linear_model, standard_deviations=3, plot=False, filename='data/calibration.csv'):
    # spits out parameters, variances for sensor
    # Load data
    data = np.loadtxt(filename, delimiter=',', skiprows=1)

    # load data x, z
    index, time, distance, velocity_command, raw_ir1, raw_ir2, raw_ir3, raw_ir4, \
        sonar1, sonar2 = data.T

    if sensor == 'ir1':
        z = raw_ir1
    elif sensor == 'ir2':
        z = raw_ir2
    elif sensor == 'ir3':
        z = raw_ir3
    elif sensor == 'ir4':
        z = raw_ir4
    elif sensor == 'sonar1':
        z = sonar1
    elif sensor == 'sonar2':
        z = sonar2
    else:
        print(f'sensor {sensor} not recognised')
        quit()

    print(f'sensor: {sensor}')

    iterate = True
    stds = 4
    while stds >= 2:
        print(f'standard deviations: {stds}')
        params, cov = curve_fit(model, distance, z)

        zfit = model(distance, *params)
        z_error = z - zfit

        var_V = variance(z_error)
        std_V = stdev(z_error)
        print(f'sensor variance: {var_V}')
        print(f'sensor std: {std_V}')
        mean_V = mean(z_error)
        print(f'sensor noise mean = {mean_V}')

        max_allowable_error = std_V * stds
        print(f'maximum allowable error: {max_allowable_error}')

        # removing outliers
        m = abs(z - zfit) < max_allowable_error
        distance = distance[m]
        z = z[m]

        stds -= 0.25

    zfit = model(distance, *params)
    z_error = z - zfit
    var_V = variance(z_error)
    std_V = stdev(z_error)
    print(f'sensor variance: {var_V}')
    print(f'sensor std: {std_V}')
    mean_V = mean(z_error)
    print(f'sensor noise mean = {mean_V}')

    if plot:
        fig, axes = subplots(1, 2)
        fig.suptitle(f'curve fit, sensor: {sensor}')

        axes[0].plot(distance, zfit, '.', alpha=0.2, label='fit')

        axes[0].plot(distance, z, '.', alpha=0.2, label='original')
        axes[0].set_title("original + fitted")
        axes[0].legend()

        axes[1].plot(distance, z_error, '.', alpha=0.2)
        axes[1].set_title("error")

        # fig2, axes2 = subplots(1, 2)
        # fig2.suptitle(f'corrected fit, sensor: {sensor}')

        # axes2[0].plot(distance, zfit, '.', alpha=0.2)

        # axes2[0].plot(distance, z, '.', alpha=0.2)
        # axes2[0].set_title("original + fitted")
        # axes2[0].legend()
        show()

    # if plot:
    #     axes2[1].plot(distance, z_error, '.', alpha=0.2)
    #     axes2[1].set_title("error")
    #     show()

        
    return(var_V, std_V, params)

def main():
    # every house should have one
    var_ir1, std_ir1, params_ir1 = fit_sensor('ir1', ir_model, 1, do_plot)
    print(var_ir1)
    print(std_ir1)
    print(params_ir1)

    var_ir2, std_ir2, params_ir2 = fit_sensor('ir2', ir_model, 1, do_plot)
    print(var_ir2)
    print(std_ir2)
    print(params_ir2)

    var_ir3, std_ir3, params_ir3 = fit_sensor('ir3', ir_model, 1, do_plot)
    print(var_ir3)
    print(std_ir3)
    print(params_ir3)

    var_ir4, std_ir4, params_ir4 = fit_sensor('ir4', ir4_model, 1, do_plot)
    print(var_ir4)
    print(std_ir4)
    print(params_ir4)

    var_sonar1, std_sonar1, params_sonar1 = fit_sensor('sonar1', linear_model, 1, do_plot)
    print(var_sonar1)
    print(std_sonar1)
    print(params_sonar1)

    var_sonar2, std_sonar2, params_sonar2 = fit_sensor('sonar2', linear_model, 1, do_plot)
    print(var_sonar2)
    print(std_sonar2)
    print(params_sonar2)
