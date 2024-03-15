from copy import copy
import numpy as np
from scipy.stats import norm

from channelbelt2d.environments import Valley
from channelbelt2d.objects import MeanderBelt, BraidedBelt, WingedBeltObject
from channelbelt2d.distributions import (
    TopographicLowDistribution,
    PotentialCalculator,
    GibbsDistribution,
)


class Event:
    """An event is an object together with its associated top surface after deposition"""

    def __init__(self, object, surface):
        self._object = object
        self._top_surface = surface

    def plot_object(
        self,
        ax,
        color_table,
        include_wings=True,
        include_outline=True,
    ):
        self._object.plot(ax, color_table, include_wings, include_outline)

    def plot_surface(self, ax, xx, *args, **kwargs):
        ax.plot(xx, self._top_surface, *args, **kwargs)


class FluvialDepositionalProcess:
    """Base class for fluvial depositional processes.
    Currently works for winged belt objects and Gibbs distributions.
    Should be made to also work for existing meandering and braided river systems.
    """

    def __init__(
        self,
        grid_parameters,
        initial_topography_parameters,
        object_behavior,
        process_behavior,
        visual_settings,
    ):
        self._xx = self._initialize_xaxis(grid_parameters)

        self._zz = np.full_like(
            self._xx, initial_topography_parameters["floodplain_depth"]
        )
        self._zz_initial = copy(self._zz)

        self._object_type = object_behavior["object_type"]
        self._object_parameter_distribution = object_behavior["parameter_distribution"]
        self._migration_displacement_distribution = object_behavior[
            "migration_displacement_distribution"
        ]

        self._location_factors = process_behavior["location_factors"]
        self._avulsion_parameters = process_behavior["avulsion_parameters"]
        self._floodplain_aggradation_parameters = process_behavior[
            "floodplain_aggradation_parameters"
        ]
        self._erodibility_determination = process_behavior["erodibility_determination"]

        self._color_table = self._parse_color_table(visual_settings)

        self._events = []

    def plot_objects(self, ax, include_wings=True, include_outline=True):
        for event in self._events:
            event.plot_object(ax, self._color_table, include_wings, include_outline)

    def plot_surfaces(self, ax, *args, **kwargs):
        for event in self._events:
            event.plot_surface(ax, self._xx, *args, **kwargs)

    def plot_topography(self, ax, *args, **kwargs):
        ax.plot(self._xx, self._zz, *args, **kwargs)

    def plot_background(self, ax, set_limits=True, aspect_ratio=0.001):
        # get x limits for process
        x_min = self._xx.min()
        x_max = self._xx.max()

        # Find suitable y limits for the plot
        y_min = self._zz.min()
        mean_thickness = self._object_parameter_distribution._belt_thickness.mean()
        y_max = self._zz_initial.max() + mean_thickness
        plot_height_without_margin = y_max - y_min
        plot_width = x_max - x_min
        target_plot_height = plot_width * aspect_ratio
        margin = target_plot_height / plot_height_without_margin - 1.0
        y_min -= margin
        y_max += margin

        # plot a filled rectangle for the background
        ax.fill_between(
            [x_min, x_max],
            [y_min, y_min],
            [y_max, y_max],
            facecolor=self._color_table["background"],
            edgecolor="face",
        )

        if set_limits:
            ax.set_xlim(x_min, x_max)
            ax.set_ylim(y_min, y_max)

    def plot_floodplain(self, ax):
        ax.fill_between(
            self._xx,
            self._zz_initial,
            self._zz,
            facecolor=self._color_table["floodplain"],
            edgecolor="face",
        )

    def plot_next_object_location_pdf(
        self,
        ax,
        *args,
        include_avulsion=True,
        include_migration=True,
        include_combined=True,
        **kwargs,
    ):
        avulsion_distribution = self._make_avulsion_location_distribution()
        migration_density = self._migration_location_distribution_pdf_values()

        avulsion_probability = self._avulsion_parameters["avulsion_probability"]

        if include_avulsion:
            avulsion_distribution.plot_pdf(ax, *args, label="Avulsion", **kwargs)

        if include_migration:
            ax.plot(
                self._xx,
                migration_density,
                *args,
                label="Migration",
                **kwargs,
            )

        if include_combined:
            combined_pdf = (
                avulsion_probability * avulsion_distribution.pdf_values()
                + (1 - avulsion_probability) * migration_density
            )
            ax.plot(self._xx, combined_pdf, *args, label="Combined", **kwargs)

    def draw_next_object(self):
        # Choose whether an avulsion occurs
        avulsion = (
            np.random.rand() < self._avulsion_parameters["avulsion_probability"]
            or not self._events
        )
        if avulsion:
            # Choose whether the avulsion occurs outside the grid
            outside = (
                np.random.rand() < self._avulsion_parameters["outside_probability"]
            )
            if outside:
                # Outside avulsion case:
                # - Update topography: Aggrade using avulsion aggradation
                self._update_topography_with_aggradation(
                    self._floodplain_aggradation_parameters["avulsion_aggradation"]
                )
                # - Don't add a new object

            else:
                # Inside avulsion case:
                # Update topography: Aggrade using avulsion aggradation
                self._update_topography_with_aggradation(
                    self._floodplain_aggradation_parameters["avulsion_aggradation"]
                )

                # - Add a new object normally
                potential_contributions = self._make_potential_contributions()
                potential_calculator = PotentialCalculator(
                    self._xx, potential_contributions
                )
                potential = potential_calculator.compute_potential()
                location_distribution = GibbsDistribution(
                    potential_calculator._xx, potential
                )

                # Draw all parameters needed to define the object
                location = location_distribution.draw()
                object_parameters = (
                    self._object_parameter_distribution.draw_realization()
                )

                new_object = self._make_object(
                    location, object_parameters, self._object_type
                )

                # Update topography part 1: Make it shallower where top surface of new object is above old topography
                self._update_topography_with_object(new_object)

                self._events.append(Event(new_object, self._zz))

        else:
            # Non-avulsion (migration) case:
            # - Add a new object by drawing parameters and location with
            #   the previous object's parameters as the mean
            # - Update topography using non-avulsion aggradation

            # Get the previous object's parameters
            if self._events:
                previous_object = self._events[-1]._object

                # Update topography: Aggrade using non-avulsion aggradation
                self._update_topography_with_aggradation(
                    self._floodplain_aggradation_parameters["non_avulsion_aggradation"]
                )

                # Draw new parameters
                new_object_parameters = (
                    self._object_parameter_distribution.draw_realization()
                )

                # Get the location and floodplain elevation of the previous object
                previous_location = previous_object._center_location
                previous_floodplain_elevation = previous_object._floodplain_elevation

                # Draw a 2-element vector with a horizontal and vertical displacement
                # for the location and floodplain elevation of the new object
                displacement = self._migration_displacement_distribution.draw()
                new_location = previous_location + displacement[0]
                new_floodplain_elevation = (
                    previous_floodplain_elevation - displacement[1]
                )

                # Make new object
                new_object = self._make_object(
                    new_location, new_object_parameters, self._object_type
                )

                # Set the floodplain elevation of the new object
                new_object._floodplain_elevation = new_floodplain_elevation

                # Update topography part 1:
                self._update_topography_with_object(new_object)

                self._events.append(Event(new_object, self._zz))

    def _update_topography_with_object(self, new_object):
        """Make topography shallower where top surface of new object is above old topography
        """
        self._zz = np.minimum(self._zz, new_object.get_top_surface(self._xx))

    def _update_topography_with_aggradation(self, aggradation: float):
        """Make topography shallower by filling out the deepest areas
        """
        self._zz = np.minimum(
            self._zz,
            self._zz.max() - aggradation,
        )

    def _initialize_xaxis(self, grid_parameters):
        """Initialize the x-axis for the process grid.
        grid_parameters: dictionary with keys 'x_min', 'x_max', 'n_x'
        """
        x_min = grid_parameters["x_min"]
        x_max = grid_parameters["x_max"]
        n_x = grid_parameters["n_x"]
        return np.linspace(x_min, x_max, n_x)

    def _make_potential_contributions(self):
        potential_contributions = {}
        for key, value in self._location_factors.items():
            if key == "topography":
                covariate = self._zz
            elif key == "erodibility":
                covariate = self._current_erodibility()
            else:
                raise ValueError(f"Unknown potential contribution type: {key}")

            coefficient = value["coefficient"]
            potential_contributions[key] = {
                "covariate": covariate,
                "coefficient": coefficient,
            }
        return potential_contributions

    def _depth_to_sand(self):
        depth_to_sand = np.full_like(self._xx, np.inf)

        for event in self._events:
            x_left, x_right = event._object.get_x_limits(include_wings=False)

            # Update depth_to_sand in the belt region of the object
            inside_event = np.logical_and(x_left <= self._xx, self._xx <= x_right)

            depth_to_sand[inside_event] = np.minimum(
                depth_to_sand[inside_event],
                event._object.get_top_surface(self._xx[inside_event])
                - self._zz[inside_event],
            )

        return depth_to_sand

    def _current_erodibility(self, sensing_depth=None):
        if sensing_depth is None:
            mean_thickness = self._object_parameter_distribution._belt_thickness.mean()
            mean_superelevation = (
                self._object_parameter_distribution._superelevation.mean()
            )
            mean_incision_depth = mean_thickness - mean_superelevation
            sensing_depth = mean_incision_depth

        depth_to_sand = self._depth_to_sand()
        erodibility = np.maximum(
            0, self._erodibility_determination["mapping"](depth_to_sand / sensing_depth)
        )

        return erodibility

    def _make_object(self, location, object_parameters, object_type):
        if object_type == "winged_belt":
            # Procedure to determine depth of object base:
            x_left = location - object_parameters.base_belt_width / 2
            x_right = location + object_parameters.base_belt_width / 2
            local_floodplain_elevation = self._local_depth_in_range(x_left, x_right)
            # Procedure to determine depth of each wing: shallower than floodplain iff. wing rests on previous object
            wing_elevation_lhs = min(local_floodplain_elevation, self._local_depth(x_left))
            wing_elevation_rhs = min(local_floodplain_elevation, self._local_depth(x_right))
            return WingedBeltObject(
                center_location=location,
                floodplain_elevation=local_floodplain_elevation,
                params=object_parameters,
                left_wing_elevation=wing_elevation_lhs,
                right_wing_elevation=wing_elevation_rhs
            )
        elif object_type == "meander_belt":
            raise NotImplementedError(
                "Meander belt objects not yet supported by this process."
            )
        elif object_type == "braided_belt":
            raise NotImplementedError(
                "Braided belt objects not yet supported by this process."
            )
        else:
            raise ValueError(f"Unknown object type: {object_type}")

    def _local_depth_in_range(self, x_left, x_right):
        inside_event = np.logical_and(x_left <= self._xx, self._xx <= x_right)

        if not inside_event.any():
            print(
                f"Warning: no overlap between object and process grid at x = {x_left} to x = {x_right}."
            )
            # Fall back to using the closest grid point to the center of the object
            return self._local_depth(0.5 * (x_left + x_right))

        return self._zz[inside_event].mean()

    def _local_depth(self, x):
        closest_index = np.argmin(np.abs(self._xx - x))
        return self._zz[closest_index]

    def _parse_color_table(self, visual_settings):
        """Parse the color table from the visual settings.
        The input color table has values between 0 and 255, but the
        plotting functions expect values between 0 and 1.
        """
        color_table = visual_settings["color_table"]
        for key, value in color_table.items():
            color_table[key] = tuple([x / 255 for x in value])
        return color_table

    def _make_avulsion_location_distribution(self):
        potential_contributions = self._make_potential_contributions()
        potential_calculator = PotentialCalculator(self._xx, potential_contributions)
        potential = potential_calculator.compute_potential()
        location_distribution = GibbsDistribution(potential_calculator._xx, potential)
        return location_distribution

    def _migration_location_distribution_pdf_values(self):
        previous_object = self._events[-1]._object
        mu_h_left = (
            previous_object._center_location
            - self._migration_displacement_distribution._distribution.mean[0]
        )
        mu_h_right = (
            previous_object._center_location
            + self._migration_displacement_distribution._distribution.mean[0]
        )
        sigma_h = (
            self._migration_displacement_distribution._distribution.cov[0, 0] ** 0.5
        )

        left_density = norm.pdf(self._xx, loc=mu_h_left, scale=sigma_h)
        right_density = norm.pdf(self._xx, loc=mu_h_right, scale=sigma_h)

        return (left_density + right_density) * 0.5


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
                new_belt_position = self._draw_belt_position(
                    previous_belt, new_belt_width
                )
                width_and_position_successfully_drawn = True
            except ValueError:
                number_of_failed_attempts += 1
                if number_of_failed_attempts > self._max_number_of_failed_attempts:
                    raise ValueError(
                        "Failed to draw a new meander belt. Try changing the belt width distribution."
                    )

        new_belt_base_depth = (
            self._valley.get_y_point(new_belt_position) + channel_depth
        )

        new_belt = MeanderBelt(
            new_belt_position, new_belt_base_depth, new_belt_width, new_belt_thickness
        )
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
        x_l = x[intersections[0]]  # left intersection
        x_r = x[intersections[1]]  # right intersection
        a = x_l + new_belt_width / 2
        d = x_r - new_belt_width / 2

        if not previous_belt:  # Sample uniformly on (a, d)
            x_out = np.random.uniform(a, d)
        else:  # Sample uniformly on union of (a, b) and (c, d)
            # To find b and c, find left and right limits of previous belt
            x_cl = previous_belt._position - previous_belt._width / 2
            x_cr = previous_belt._position + previous_belt._width / 2
            b = x_cl - new_belt_width / 2
            c = x_cr + new_belt_width / 2

            # Find lengths of intervals (a, b) and (c, d)
            ab = np.maximum(0, b - a)
            cd = np.maximum(0, d - c)

            if ab + cd == 0:
                raise ValueError(
                    f"Proposed meander belt is too wide to fit (width = {new_belt_width})."
                )

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
        xx_new_belt = np.linspace(
            new_belt_position - new_belt_width / 2,
            new_belt_position + new_belt_width / 2,
            n_pts_new_belt,
        )

        corrected_active_surface = self.correct_active_surface(ignore_valley_edges=True)

        # Check if valley is full
        shallowest_depth_reached = corrected_active_surface.min()
        if np.any(
            shallowest_depth_reached < self._valley.get_minimum_depth() + aggradation
        ):
            raise ValueError("No room left for new belt. Halting deposition.")

        base_new_belt = np.interp(
            xx_new_belt, self._valley._xx, corrected_active_surface
        )

        new_belt = BraidedBelt(
            new_belt_position,
            base_new_belt,
            new_belt_width,
            new_belt_elevation,
            channel_depth,
        )
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
        topo_distr = TopographicLowDistribution(
            self._active_surface,
            self._valley.get_x_min(),
            self._valley.get_x_max(),
            belt_width,
            gamma=self._topographic_relief_sensitivity,
        )
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
