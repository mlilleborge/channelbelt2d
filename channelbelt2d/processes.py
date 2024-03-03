from copy import copy
import numpy as np

from channelbelt2d.environments import Valley
from channelbelt2d.objects import MeanderBelt, BraidedBelt, WingedBeltObject
from channelbelt2d.distributions import TopographicLowDistribution, PotentialCalculator, GibbsDistribution


class Event:
    """An event is an object together with its associated parameters
       and the top surface after deposition
    """
    def __init__(self, object, surface):
        self._object = object
        self._top_surface = surface

    def plot(self, ax, xx, *args, **kwargs):
        self._object.plot(ax, *args, **kwargs)
        ax.plot(xx, self._top_surface, *args, **kwargs)


class FluvialDepositionalProcess:
    """Base class for fluvial depositional processes.
       Currently works for winged belt objects and Gibbs distributions.
       Should be made to also work for existing meandering and braided river systems.
    """
    def __init__(self,
                 grid_parameters,
                 initial_topography_parameters,
                 object_behavior,
                 process_behavior):
        self._xx = self._initialize_xaxis(grid_parameters)

        self._zz = np.full_like(self._xx, initial_topography_parameters['floodplain_depth'])
        self._zz_initial = copy(self._zz)

        self._object_type = object_behavior['object_type']
        self._object_parameter_distribution = object_behavior['parameter_distribution']

        self._process_behavior = process_behavior

        self._events = []

    def plot_system(self, ax):
        for event in self._events:
            event.plot(ax, self._xx)

    def draw_next_object(self):
        potential_contributions = self._make_potential_contributions()
        potential_calculator = PotentialCalculator(self._xx, potential_contributions)
        potential = potential_calculator.compute_potential()
        location_distribution = GibbsDistribution(potential_calculator._xx, potential)

        # Draw all parameters needed to define the object
        location = location_distribution.draw()
        object_parameters = self._object_parameter_distribution.draw_realization()

        new_object = self._make_object(location, object_parameters, self._object_type)

        # Update topography
        self._zz = np.maximum(self._zz, new_object.get_top_surface(self._xx))

        self._events.append(Event(new_object, self._zz))

    def _initialize_xaxis(self, grid_parameters):
        """Initialize the x-axis for the process grid.
           grid_parameters: dictionary with keys 'x_min', 'x_max', 'n_x'
        """
        x_min = grid_parameters['x_min']
        x_max = grid_parameters['x_max']
        n_x = grid_parameters['n_x']        
        return np.linspace(x_min, x_max, n_x)

    def _make_potential_contributions(self):
        potential_contributions = {}
        for key, value in self._process_behavior.items():
            if key == 'topography':
                covariate = self._zz
            elif key == 'erodibility':
                covariate = self._current_erodibility()
            else:
                raise ValueError(f"Unknown potential contribution type: {key}")

            coefficient = value['coefficient']
            potential_contributions[key] = {'covariate': covariate, 'coefficient': coefficient}
        return potential_contributions

    def _current_erodibility(self):
        erodibility = np.zeros_like(self._xx)

        if self._events:
            most_recent_event = self._events[-1]
            x_left, x_right = most_recent_event._object.get_x_limits()
            erodibility[np.logical_and(x_left <= self._xx, self._xx <= x_right)] = 1.0
        
        return erodibility
    
    def _make_object(self, location, object_parameters, object_type):
        if object_type == 'winged_belt':
            x_left = location - object_parameters.base_belt_width / 2
            x_right = location + object_parameters.base_belt_width / 2
            local_floodplain_elevation = self._local_depth(x_left, x_right)
            return WingedBeltObject(center_location=location,
                                    floodplain_elevation=local_floodplain_elevation,
                                    params=object_parameters)
        elif object_type == 'meander_belt':
            raise NotImplementedError("Meander belt objects not yet supported by this process.")
        elif object_type == 'braided_belt':
            raise NotImplementedError("Braided belt objects not yet supported by this process.")
        else:
            raise ValueError(f"Unknown object type: {object_type}")
    
    def _local_depth(self, x_left, x_right):
        return self._zz[np.logical_and(x_left <= self._xx, self._xx <= x_right)].mean()


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
