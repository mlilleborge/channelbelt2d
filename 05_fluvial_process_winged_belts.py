import matplotlib.pyplot as plt
from scipy.stats import norm

from channelbelt2d.processes import FluvialDepositionalProcess
from channelbelt2d.objects import WingedBeltParameterDistribution


# Define grid parameters
grid_params = {
    "x_min": -1500.0,
    "x_max": 1500.0,
    "n_x": 100,
}

# Define initial topography parameters
init_topo_params = {
    "floodplain_depth": 0.0,
}

# Define object parameter distribution
obj_param_distr = WingedBeltParameterDistribution(
    left_wing_width=norm(loc=250, scale=50),
    right_wing_width=norm(loc=250, scale=50),
    base_belt_width=norm(loc=500, scale=60),
    top_belt_width=norm(loc=600, scale=80),
    belth_thickness=norm(loc=8, scale=0.6),
    superelevation=norm(loc=2, scale=0.5)
)

# Define object behavior
obj_behavior = {
    "object_type": "winged_belt",
    "parameter_distribution": obj_param_distr
}

# Define process behavior
proc_behavior = {
    'topography': {'coefficient': -2},
    'erodibility': {'coefficient': -3}
}

# Create the fluvial process
process = FluvialDepositionalProcess(
    grid_parameters=grid_params,
    initial_topography_parameters=init_topo_params,
    object_behavior=obj_behavior,
    process_behavior=proc_behavior
)
    
# Draw a few objects
n_objects = 50
for i in range(n_objects):
    process.draw_next_object()


# Plot the state of the process
fig, ax = plt.subplots(1, 1, figsize=(8, 4))
process.plot_system(ax)
ax.invert_yaxis()
ax.set_xlabel('x')
ax.set_ylabel('Depth')

ax.set_ylim(20, -20)
plt.show()
