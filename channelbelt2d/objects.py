# Channel belt object for meandering river deposition
import numpy as np
from scipy.interpolate import UnivariateSpline, LSQUnivariateSpline

class MeanderBelt:
    def __init__(self, x, y, w, h):
        self._position = x
        self._width = w
        self._base_depth = y
        self._thickness = h
    
    def get_top_depth(self):
        return self._base_depth - self._thickness
    
    def plot(self, ax):
        x = np.array([self._position - self._width/2, self._position + self._width/2])
        y = np.array([self._base_depth, self.get_top_depth()])
        ax.fill_between(x, y[0], y[1], color='goldenrod', edgecolor='black')


# Channel belt object for braided river deposition
class BraidedBelt:
    def __init__(self, x, b, w, ha, he):
        """x: x-coordinate of center
           b: base depth
           w: full width
           ha: accretion thickness
           he: erosion thickness
        """
        n = b.size
        self._position = x
        self._x = np.linspace(x - w/2, x + w/2, n)
        self._base_depth = b
        self._smooth_base_depth = self.compute_smooth_base_depth()
        self._width = w
        self._accretion_thickness = ha
        self._erosion_thickness = he
        self._top_depth = self._smooth_base_depth - self.accretion_function(self._x)
        self._base_depth = self._smooth_base_depth + self.erosion_function(self._x)
    
    def shape_round(self, x):
        return 1 - (2 * (x - self._position) / self._width)**2
    
    def shape_flattened(self, x):
        return 1 - (2 * (x - self._position) / self._width)**4

    def accretion_function(self, x):
        return self._accretion_thickness * self.shape_flattened(x)
    
    def erosion_function(self, x):
        return self._erosion_thickness * self.shape_flattened(x)
    
    def get_top_depth(self):
        return self._top_depth.mean()

    def compute_smooth_base_depth(self):
        # Smooth base depth using spline interpolation,
        # making sure the endpoints are interpolated exactly
        spl = UnivariateSpline(self._x, self._base_depth, s=self._x.size * 1.5)
        t = spl.get_knots()
        wts = np.ones_like(self._x)
        wts[0] = 1e6
        wts[-1] = 1e6
        spl_weighted = LSQUnivariateSpline(self._x, self._base_depth, t[1:-1], w=wts)
        return spl_weighted(self._x)

    def plot(self, ax):
        ax.fill_between(self._x, self._top_depth, self._base_depth,
                        color='goldenrod', edgecolor='black')


# Probability distributions of parameters controlling channel belts
class BeltParameters:
    def __init__(self, aggradation, channel_depth, belt_elevation, belt_width):
        self._aggradation = aggradation
        self._channel_depth = channel_depth
        self._belt_elevation = belt_elevation
        self._belt_width = belt_width
    
    def aggradation(self):
        return self._aggradation

    def channel_depth(self):
        return self._channel_depth

    def belt_elevation(self):
        return self._belt_elevation
    
    def belt_width(self):
        return self._belt_width
