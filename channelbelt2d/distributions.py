import numpy as np
from scipy.stats import norm
from scipy.integrate import cumulative_trapezoid

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
        self._gamma = gamma
        
        self._xx = np.linspace(x_min, x_max, self._n)
        self._zz = depth
        
        potential = self._compute_potential()

        # Convolve potential with a Gaussian kernel
        # to capture effect of nonzero object extent
        potential = self._convolve_potential(potential, object_width)

        cdf, pdf = self._compute_cdf(potential)
        self._cdf = cdf
        self._pdf = pdf
    
    def _compute_potential(self):
        # Compute potential values at each point in discretization
        # negative sign (potential decreases as depth increases)
        return -self._gamma * self._zz 

    def _convolve_potential(self, potential, object_width):
        sigma = object_width / 3
        kernel = norm(scale=sigma)
        return np.convolve(potential, kernel.pdf(self._xx), mode='same')

    def _compute_cdf(self, potential):
        # Compute cumulative distribution function
        pdf_unnormalized = np.exp(-potential)
        cdf_unnormalized = cumulative_trapezoid(pdf_unnormalized, self._xx, initial=0)
        normalization_constant = cdf_unnormalized[-1]
        cdf = cdf_unnormalized / normalization_constant
        pdf = pdf_unnormalized / normalization_constant
        #assert(np.all(np.diff(cdf) > 0))
        return cdf, pdf

    def draw(self):
        # Sample from the distribution
        u = np.random.uniform()
        return np.interp(u, self._cdf, self._xx)
    
    def plot_cdf(self, ax, *args, **kwargs):
        ax.plot(self._xx, self._cdf, *args, **kwargs)
    
    def plot_pdf(self, ax, *args, **kwargs):
        ax.plot(self._xx, self._pdf, *args, **kwargs)
