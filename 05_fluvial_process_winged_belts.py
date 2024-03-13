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
        "mean": {"horizontal": 300, "vertical": 1.0},
        "standard_deviation": {"horizontal": 150, "vertical": 0.05},
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
        "outside_probability": 0.0,
    },
    "floodplain_aggradation_parameters": {
        "non_avulsion_aggradation": 1.0,
        "avulsion_aggradation": 3.0,
    },
    "erodibility_determination": {
        "mapping": lambda x: 1 - x,
        "input variable": "depth to sand / sensing depth",
        "output variable": "erodibility",
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

# Draw objects
n_objects = 150
for i in range(n_objects):
    process.draw_next_object()


# Plot the state of the process
fig, ax = plt.subplots(1, 1, figsize=(8, 4))

process.plot_background(ax, set_limits=True, aspect_ratio=0.025)

process.plot_objects(ax, include_wings=True, include_outline=False)
process.plot_objects(ax, include_wings=False, include_outline=True)

# process.plot_surfaces(ax, color="black", linewidth=0.5)
process.plot_topography(ax, linewidth=2, color="black")

ax.invert_yaxis()
ax.set_xlabel("x")
ax.set_ylabel("Depth")

# Plot the current erodibility
if True:
    fig, ax = plt.subplots(1, 1, figsize=(8, 4))

    erodibility = process._current_erodibility()
    ax.plot(process._xx, erodibility, color="black", linewidth=2)
    ax.set_xlabel("x")
    ax.set_ylabel("Erodibility")
    ax.set_xlim(grid_params["x_min"], grid_params["x_max"])

# Plot the PDF of the location of the next object
if True:
    fig, ax = plt.subplots(1, 1, figsize=(8, 4))
    process.plot_next_object_location_pdf(
        ax, include_avulsion=True, include_migration=True, include_combined=True
    )
    ax.set_xlabel("x")
    ax.set_ylabel("Probability density")
    ax.legend()
    ax.set_xlim(grid_params["x_min"], grid_params["x_max"])

plt.show()
