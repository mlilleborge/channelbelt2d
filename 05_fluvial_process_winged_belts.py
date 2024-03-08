import matplotlib.pyplot as plt
from scipy.stats import norm, multivariate_normal
import numpy as np

from channelbelt2d.processes import FluvialDepositionalProcess
from channelbelt2d.objects import WingedBeltParameterDistribution
from channelbelt2d.distributions import MigrationDisplacementDistribution


# Define grid parameters
grid_params = {
    "x_min": -5000.0,
    "x_max": 5000.0,
    "n_x": 200,
}

# Define initial topography parameters
init_topo_params = {
    "floodplain_depth": 0.0,
}

# Define object parameter distribution
obj_param_distr = WingedBeltParameterDistribution(
    left_wing_width=norm(loc=5000, scale=500),
    right_wing_width=norm(loc=5000, scale=500),
    base_belt_width=norm(loc=500, scale=150),
    top_belt_width=norm(loc=600, scale=150),
    belth_thickness=norm(loc=4, scale=0.3),
    superelevation=norm(loc=1, scale=0.2),
)

migr_distr = MigrationDisplacementDistribution(
    parameters={
        "mean": {"horizontal": 400, "vertical": 1.0},
        "standard_deviation": {"horizontal": 50, "vertical": 0.05},
        "correlation": 0.0,
    }
)

# Define object behavior
obj_behavior = {
    "object_type": "winged_belt",
    "parameter_distribution": obj_param_distr,
    "migration_displacement_distribution": migr_distr,
}

# Define process behavior
proc_behavior = {
    "location_factors": {
        "topography": {"coefficient": -1},
        "erodibility": {"coefficient": -1},
    },
    "avulsion_parameters": {
        "avulsion_probability": 0.5,
        "outside_probability": 0.1,
    },
    "floodplain_aggradation_parameters": {
        "non_avulsion_aggradation": 1.0,
        "avulsion_aggradation": 3.0,
    },
}

visual_settings = {
    "color_table": {
        "channel_belt": (255, 157, 0),
        "wings": (217, 255, 0),
        "floodplain": (56, 207, 104),
        "background": (150, 150, 150),
    }
}

# Create the fluvial process
process = FluvialDepositionalProcess(
    grid_parameters=grid_params,
    initial_topography_parameters=init_topo_params,
    object_behavior=obj_behavior,
    process_behavior=proc_behavior,
    visual_settings=visual_settings,
)

# Draw a few objects
n_objects = 40
for i in range(n_objects):
    process.draw_next_object()


# Plot the state of the process
fig, ax = plt.subplots(1, 1, figsize=(8, 4))

process.plot_background(ax, set_limits=True)

process.plot_objects(ax, include_wings=True, include_outline=False)
process.plot_objects(ax, include_wings=False, include_outline=True)

# process.plot_surfaces(ax, color="black", linewidth=0.5)
process.plot_topography(ax, linewidth=2, color="black")

ax.invert_yaxis()
ax.set_xlabel("x")
ax.set_ylabel("Depth")

plt.show()
