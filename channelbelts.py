import numpy as np
from scipy.stats import norm
from scipy.integrate import cumulative_trapezoid
from scipy.interpolate import UnivariateSpline, LSQUnivariateSpline
from copy import copy

###########################################################################
# Modeling channel belts to recreate figures 11 and 12 on p. 222 of
# Boggs, S. (2014). Principles of Sedimentology and Stratigraphy
# (5. ed., Pearson New International Edition). Pearson Education Limited.
#
# From the text:
#
# Lateral migration of braided rivers leaves sheetlike or wedge-shaped
# deposits of channel and bar complexes (Cant, 1982). Lateral migration
# combined with aggradation leads to deposition of sheet sandstones or
# conglomerates that enclose very thin, nonpersistent shales within
# coarser sediments (Fig. 11).
# 
# Migration of meandering streams, which
# are confined within narrow, sandy meander belts of stream floodplains,
# generate linear “shoestring” sand bodies oriented parallel to the
# river course. These shoestring sands are surrounded by finer grained,
# overbank floodplain sediments. Periodic stream avulsion may create new
# channels over time, leading to formation of several linear sand bodies
# within a major stream valley (Fig. 12).
#
# Reference cited:
# Cant, D.J. 1982. Fluvial facies models and their application.
# In Scholle, P. A., and D. Spearing (eds.).
# Sandstone Depositional Environments.
# Amer Assoc. Petroleum Geologists Mem. 31. 115–138.
########################################################################


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


# Channel belt object for meandering river deposition
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


# Valley confining the river
class Valley:
    def __init__(self, valley_parameters, n=100):
        self._width = valley_parameters._get_width()
        self._depth = valley_parameters._get_depth()
        self._level = valley_parameters._get_initial_level()

        w = self._width
        d = self._depth
        self._xx = np.linspace(-w/2, w/2, n)
        self._y0 = self.shape(self._xx, w, d)
        
    def shape(self, x, w, d):
        return d * (1 - (2 * x / w)**4)

    def get_x(self):
        return self._xx

    def get_y(self):
        return np.minimum(self._y0, self._level * np.ones_like(self._xx))
    
    def get_y_point(self, x):
        return np.minimum(self.shape(x, self._width, self._depth), self._level)
    
    def get_x_min(self):
        return -self._width/2
    
    def get_x_max(self):
        return self._width/2

    def plot(self, ax):
        ax.plot(self._xx, self._y0, color='black')                          # initial valley
        ax.fill_between(self._xx, self._y0, self.get_y(), color='yellowgreen')    # fill to current level

    def set_level(self, newlevel):
        self._level = newlevel
    
    def aggrade(self, aggradation):
        self._level -= aggradation
    
    def plot_background(self, ax):
        x_min, x_max = ax.get_xlim()
        y_min, y_max = ax.get_ylim()
        x = np.concatenate(([x_min], self._xx, [x_max]))
        y_top = np.concatenate(([y_min], self._y0, [y_min]))
        y_bottom = np.repeat(y_max, x.size)
        ax.fill_between(x, y_top, y_bottom, color='darkgray')
    
    def get_minimum_depth(self):
        return self._y0.min()


# Parameters defining valley geometry
class ValleyParameters:
    def __init__(self, width, depth, initial_level=None):
        self._width = width
        self._depth = depth
        # initial level equals depth by default
        self._initial_level = depth if initial_level is None else initial_level
    
    def _get_width(self):
        return self._width
    
    def _get_depth(self):
        return self._depth
    
    def _get_initial_level(self):
        return self._initial_level


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


# Meandering river depositional process
class MeanderingRiverDeposition:
    def __init__(self, valley_parameters, belt_parameters):
        self._valley = Valley(valley_parameters)
        self._belt_parameters = belt_parameters
        self._belts = []
        self._max_number_of_failed_attempts = 100

    def draw_belt(self):
        surface = self._valley.get_y()

        previous_belt = None
        if len(self._belts) > 0:
            previous_belt = self._belts[-1]
        
        aggradation = self._draw_aggradation()
        channel_depth = self._draw_channel_depth()
        belt_elevation = self._draw_belt_elevation()

        new_belt_thickness = aggradation + channel_depth + belt_elevation
        
        width_and_position_successfully_drawn = False
        number_of_failed_attempts = 0
        while not width_and_position_successfully_drawn:
            new_belt_width = self._draw_belt_width()
            try:
                new_belt_position = self._draw_belt_position(previous_belt, new_belt_width)
                width_and_position_successfully_drawn = True
            except ValueError:
                number_of_failed_attempts += 1
                if number_of_failed_attempts > self._max_number_of_failed_attempts:
                    raise ValueError("Failed to draw a new meander belt. Try changing the belt width distribution.")

        new_belt_base_depth = self._valley.get_y_point(new_belt_position) + channel_depth
        
        new_belt = MeanderBelt(new_belt_position, new_belt_base_depth, new_belt_width, new_belt_thickness)
        self._belts.append(new_belt)

        self._valley.aggrade(aggradation)
    
    def _draw_aggradation(self):
        return self._belt_parameters.aggradation().rvs()
    
    def _draw_channel_depth(self):
        return self._belt_parameters.channel_depth().rvs()
    
    def _draw_belt_elevation(self):
        return self._belt_parameters.belt_elevation().rvs()
    
    def _draw_belt_width(self):
        return self._belt_parameters.belt_width().rvs()
    

    # Sample position uniformly on unobstructed valley floor
    # - avoid previous belt
    # - don't get too close to valley edges or previous belt
    def _draw_belt_position(self, previous_belt, new_belt_width):
        # To find endpoints of flat region,
        # look for intersections between valley surface and horizontal line at level
        x = self._valley._xx
        y = self._valley.get_y()
        level = self._valley._level
        intersections = np.where(np.diff(np.sign(y - level)))[0]
        x_l = x[intersections[0]] # left intersection
        x_r = x[intersections[1]] # right intersection
        a = x_l + new_belt_width/2
        d = x_r - new_belt_width/2

        if not previous_belt: # Sample uniformly on (a, d)
            x_out = np.random.uniform(a, d)
        else: # Sample uniformly on union of (a, b) and (c, d)
            # To find b and c, find left and right limits of previous belt
            x_cl = previous_belt._position - previous_belt._width/2
            x_cr = previous_belt._position + previous_belt._width/2
            b = x_cl - new_belt_width/2
            c = x_cr + new_belt_width/2

            # Find lengths of intervals (a, b) and (c, d)
            ab = np.maximum(0, b - a)
            cd = np.maximum(0, d - c)

            if ab + cd == 0:
                raise ValueError(f"Proposed meander belt is too wide to fit (width = {new_belt_width}).")

            # Choose interval randomly with probability proportional to interval length
            p_ab = ab / (ab + cd)
            p_cd = cd / (ab + cd)
            interval = np.random.choice([0, 1], p=[p_ab, p_cd])

            # Sample x_out uniformly on the chosen interval
            if interval == 0:
                x_out = np.random.uniform(a, b)
            else:
                x_out = np.random.uniform(c, d)
    
        return x_out

    def _plot_valley(self, ax):
        self._valley.plot(ax)
    
    def _plot_belts(self, ax):
        for belt in self._belts:
            belt.plot(ax)


# Braided river depositional process
class BraidedRiverDeposition:
    def __init__(self, valley, belt_parameters, gamma=1.0):
        self._valley = Valley(valley)
        self._belt_parameters = belt_parameters
        self._belts = []
        self._active_surface = copy(self._valley.get_y())
        self._topographic_relief_sensitivity = gamma

    def draw_belt(self):
        previous_belt = None
        if len(self._belts) > 0:
            previous_belt = self._belts[-1]
        
        aggradation = self._draw_aggradation()
        channel_depth = self._draw_channel_depth()

        new_belt_width = self._draw_belt_width()
        new_belt_position = self._draw_belt_position(new_belt_width)
        new_belt_elevation = self._draw_belt_elevation()

        n_pts_new_belt = 100
        xx_new_belt = np.linspace(new_belt_position - new_belt_width/2, new_belt_position + new_belt_width/2, n_pts_new_belt)

        corrected_active_surface = self.correct_active_surface(ignore_valley_edges=True)
        
        # Check if valley is full
        shallowest_depth_reached = corrected_active_surface.min()
        if np.any(shallowest_depth_reached < self._valley.get_minimum_depth() + aggradation):
            raise ValueError("No room left for new belt. Halting deposition.")
        
        base_new_belt = np.interp(xx_new_belt, self._valley._xx, corrected_active_surface)

        new_belt = BraidedBelt(new_belt_position, base_new_belt, new_belt_width, new_belt_elevation, channel_depth)
        self._belts.append(new_belt)
        
        # new_belt_top_depth = new_belt.get_top_depth()
        # valley_floor_level = new_belt_top_depth + new_belt_elevation
        # self._valley.set_level(valley_floor_level)
        self._valley.aggrade(aggradation)
        self._active_surface = self._update_active_surface(new_belt)
        
    def correct_active_surface(self, ignore_valley_edges=False):
        # Compute base surface of new belt, ignoring valley edges if needed
        # Find intersections between valley surface and horizontal line at level
        x = self._valley._xx
        y = self._valley.get_y()
        level = self._valley._level
        intersections = np.where(np.diff(np.sign(y - level)))[0]
        x_l = x[intersections[0]]
        x_r = x[intersections[1]]

        # Set corrected_active_surface to active_surface
        corrected_active_surface = copy(self._active_surface)
        
        # If ignore_valley_edges is True, set active_surface to level outside of [x_l, x_r]
        if ignore_valley_edges:
            corrected_active_surface[x <= x_l] = level
            corrected_active_surface[x >= x_r] = level
        
        return corrected_active_surface

    def _draw_aggradation(self):
        return self._belt_parameters.aggradation().rvs()
    
    def _draw_channel_depth(self):
        return self._belt_parameters.channel_depth().rvs()
    
    def _draw_belt_elevation(self):
        return self._belt_parameters.belt_elevation().rvs()

    def _draw_belt_position(self, belt_width):
        # Use topographic low distribution based on current valley geometry
        topo_distr = TopographicLowDistribution(self._active_surface,
                                                self._valley.get_x_min(),
                                                self._valley.get_x_max(),
                                                belt_width,
                                                gamma=self._topographic_relief_sensitivity)
        return topo_distr.draw()
    
    def _draw_belt_width(self):
        return self._belt_parameters.belt_width().rvs()
    
    def _plot_valley(self, ax):
        self._valley.plot(ax)

    def _plot_belts(self, ax):
        for belt in self._belts:
            belt.plot(ax)
        
        self._valley.plot_background(ax)
    
    def _update_active_surface(self, new_belt):
        surface = np.minimum(self._active_surface, self._valley.get_y())
        x_valley = self._valley._xx
        x_belt = new_belt._x
        x_belt_min = x_belt[0]
        x_belt_max = x_belt[-1]
        for i, x in enumerate(x_valley):
            if x_belt_min <= x <= x_belt_max:
                top_belt_i = np.interp(x, x_belt, new_belt._top_depth)
                surface[i] = min(surface[i], top_belt_i)
        return surface
