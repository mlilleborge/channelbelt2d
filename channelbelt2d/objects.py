# Channel belt object for meandering river deposition
import numpy as np
from scipy.stats import norm, truncnorm
from scipy.interpolate import UnivariateSpline, LSQUnivariateSpline
from dataclasses import dataclass

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


@dataclass
class WingedBeltParameterRealization:
    left_wing_width: float
    right_wing_width: float
    base_belt_width: float
    top_belt_width: float
    belth_thickness: float
    superelevation: float


class WingedBeltObject:
    def __init__(self, center_location, floodplain_elevation, params: WingedBeltParameterRealization):
        self._center_location = center_location
        self._floodplain_elevation = floodplain_elevation
        self._left_wing_width = params.left_wing_width
        self._right_wing_width = params.right_wing_width
        self._base_belt_width = params.base_belt_width
        self._top_belt_width = params.top_belt_width
        self._belt_thickness = params.belth_thickness
        self._superelevation = params.superelevation
    
    def plot(self, ax, n=100):
        self._plot_belt_center(ax, n)

        self._plot_left_slope(ax, n)
        self._plot_right_slope(ax, n)

        self._plot_left_wing(ax, n)
        self._plot_right_wing(ax, n)
        
    def _plot_belt_center(self, ax, n):
        x_left = self._center_location - self._base_belt_width / 2
        x_right = self._center_location + self._base_belt_width / 2
        x_top = np.linspace(x_left, x_right, n)
        top_depth = self._floodplain_elevation - self._superelevation
        base_depth = self._floodplain_elevation + self._belt_thickness - self._superelevation
        y_top = np.full(n, top_depth)
        y_base = np.full(n, base_depth)

        ax.fill_between(x_top, y_top, y_base, color='goldenrod', edgecolor='black')
    
    def _plot_left_slope(self, ax, n):
        x_left = self._center_location - self._top_belt_width / 2
        x_right = self._center_location - self._base_belt_width / 2
        x = np.linspace(x_left, x_right, n)
        top_depth = self._floodplain_elevation - self._superelevation
        left_base_depth = self._floodplain_elevation
        right_base_depth = self._floodplain_elevation + self._belt_thickness - self._superelevation
        y_top = np.full(n, top_depth)
        y_base = np.linspace(left_base_depth, right_base_depth, n)

        ax.fill_between(x, y_top, y_base, color='goldenrod', edgecolor='black')
    
    def _plot_right_slope(self, ax, n):
        x_left = self._center_location + self._base_belt_width / 2
        x_right = self._center_location + self._top_belt_width / 2
        x = np.linspace(x_left, x_right, n)
        top_depth = self._floodplain_elevation - self._superelevation
        left_base_depth = self._floodplain_elevation + self._belt_thickness - self._superelevation
        right_base_depth = self._floodplain_elevation
        y_top = np.full(n, top_depth)
        y_base = np.linspace(left_base_depth, right_base_depth, n)

        ax.fill_between(x, y_top, y_base, color='goldenrod', edgecolor='black')
    
    def _plot_left_wing(self, ax, n):
        x_left = self._center_location - self._top_belt_width / 2 - self._left_wing_width
        x_right = self._center_location - self._top_belt_width / 2
        x = np.linspace(x_left, x_right, n)
        left_depth = self._floodplain_elevation
        right_depth = self._floodplain_elevation - self._superelevation
        y_top = np.linspace(left_depth, right_depth, n)
        y_base = np.full(n, left_depth)

        ax.fill_between(x, y_top, y_base, color='goldenrod', edgecolor='black')

    def _plot_right_wing(self, ax, n):
        x_left = self._center_location + self._top_belt_width / 2
        x_right = self._center_location + self._top_belt_width / 2 + self._right_wing_width
        x = np.linspace(x_left, x_right, n)
        left_depth = self._floodplain_elevation - self._superelevation
        right_depth = self._floodplain_elevation
        y_top = np.linspace(left_depth, right_depth, n)
        y_base = np.full(n, right_depth)

        ax.fill_between(x, y_top, y_base, color='goldenrod', edgecolor='black')
    
    def get_top_surface(self, xx):
        zz = np.full(xx.size, self._floodplain_elevation)
        x_left_ramp_start = self._center_location - self._top_belt_width / 2 - self._left_wing_width
        x_left_ramp_end = self._center_location - self._top_belt_width / 2
        x_right_ramp_start = self._center_location + self._top_belt_width / 2
        x_right_ramp_end = self._center_location + self._top_belt_width / 2 + self._right_wing_width

        for i, x_i in enumerate(xx):
            if x_left_ramp_start <= x_i <= x_left_ramp_end:
                zz[i] -= (x_i - x_left_ramp_start) / (x_left_ramp_end - x_left_ramp_start) * self._superelevation
            elif x_left_ramp_end <= x_i <= x_right_ramp_start:
                zz[i] -= self._superelevation
            elif x_right_ramp_start <= x_i <= x_right_ramp_end:
                zz[i] -= (x_right_ramp_end - x_i) / (x_right_ramp_end - x_right_ramp_start) * self._superelevation
        
        return zz
    
    def get_x_limits(self):
        x_left = self._center_location - self._top_belt_width / 2 - self._left_wing_width
        x_right = self._center_location + self._top_belt_width / 2 + self._right_wing_width
        return x_left, x_right


class WingedBeltParameterDistribution:
    def __init__(self,
                    left_wing_width,
                    right_wing_width,
                    base_belt_width,
                    top_belt_width,
                    belth_thickness,
                    superelevation):
            self._left_wing_width = left_wing_width
            self._right_wing_width = right_wing_width
            self._base_belt_width = base_belt_width
            self._top_belt_width = top_belt_width
            self._belt_thickness = belth_thickness
            self._superelevation = superelevation
        
    def draw_realization(self):
        # The top width distribution is truncated below at the base width
        base_belt_width = self._base_belt_width.rvs()
        truncation_threshold = (base_belt_width - self._base_belt_width.mean()) / self._base_belt_width.std()
        top_belt_width_distribution_truncated = truncnorm(a=truncation_threshold,
                                                          b=np.inf,
                                                          loc=self._top_belt_width.mean(),
                                                          scale=self._top_belt_width.std())
        top_belt_width = top_belt_width_distribution_truncated.rvs()

        return WingedBeltParameterRealization(
            left_wing_width=self._left_wing_width.rvs(),
            right_wing_width=self._right_wing_width.rvs(),
            base_belt_width=base_belt_width,
            top_belt_width=top_belt_width,
            belth_thickness=self._belt_thickness.rvs(),
            superelevation=self._superelevation.rvs()
        )


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
