import numpy as np
from scipy.stats import norm
from scipy.integrate import cumulative_trapezoid


class PotentialCalculator:
    """Calculates potential values that can be plugged into GibbsDistribution"""
    def __init__(self, x_values, contributions):
        """x_values: array of x values
           contributions: dictionary containing contributions to the potential
           The keys of the dictionary are the names of the contributions
           The values are dictionaries that contain the following keys:
           - 'covariate': an array of input values corresponding to the x_values
           - 'coefficient': a scalar defining the slope of the contribution
        """
        self._xx = x_values
        self._contributions = contributions
    
    def plot_contributions(self, ax, *args, **kwargs):
        for name, contribution in self._contributions.items():
            ax.plot(self._xx, contribution['coefficient'] * contribution['covariate'], *args, label=name, **kwargs)
        
        # The total potential is the sum of the contributions
        potential = self.compute_potential()
        ax.plot(self._xx, potential, label='Total', **kwargs)

        ax.legend()
    
    def compute_potential(self):
        potential = np.zeros(self._xx.size)
        for name, contribution in self._contributions.items():
            potential += contribution['coefficient'] * contribution['covariate']
        return potential
    

class GibbsDistribution:
    def __init__(self, rv_values, potential_values):
        self._xx = rv_values
        self._potential = potential_values
        self._cdf, self._pdf = self._compute_cdf(self._potential)
    
    def _compute_cdf(self, potential):
        # Compute cumulative distribution function
        pdf_unnormalized = np.exp(-potential)
        cdf_unnormalized = cumulative_trapezoid(pdf_unnormalized, self._xx, initial=0)
        normalization_constant = cdf_unnormalized[-1]
        cdf = cdf_unnormalized / normalization_constant
        pdf = pdf_unnormalized / normalization_constant
        return cdf, pdf

    def draw(self):
        # Sample from the distribution
        u = np.random.uniform()
        return np.interp(u, self._cdf, self._xx)
    
    def plot_cdf(self, ax, *args, **kwargs):
        ax.plot(self._xx, self._cdf, *args, **kwargs)
    
    def plot_pdf(self, ax, *args, **kwargs):
        ax.plot(self._xx, self._pdf, *args, **kwargs)


# A probability distribution that concentrates probability mass at topographic lows
class TopographicLowDistribution:
    def __init__(self, depth, x_min, x_max, object_width, gamma=1.0):
        """depth_function: function defining depth as a function of x
           x_min, x_max: limits of the domain of the distribution
           n: number of points in discretization of [x_min, x_max]
        """
        self._x_min = x_min
        self._x_max = x_max
        self._n = depth.size
        
        self._xx = np.linspace(x_min, x_max, self._n)

        self._potential_contributions = {'topography': {'covariate': depth, 'coefficient': -gamma}}
        self._potential_calculator = PotentialCalculator(self._xx, self._potential_contributions)
        
        potential = self._potential_calculator.compute_potential()

        # Convolve potential with a Gaussian kernel
        # to capture effect of nonzero object extent
        potential = self._convolve_potential(potential, object_width)

        self._distribution = GibbsDistribution(self._xx, potential)
    
    def _convolve_potential(self, potential, object_width):
        sigma = object_width / 3
        kernel = norm(scale=sigma)
        return np.convolve(potential, kernel.pdf(self._xx), mode='same')

    def draw(self):
        return self._distribution.draw()

    def plot_cdf(self, ax, *args, **kwargs):
        self._distribution.plot_cdf(ax, *args, **kwargs)
    
    def plot_pdf(self, ax, *args, **kwargs):
        self._distribution.plot_pdf(ax, *args, **kwargs)
