# Make a distribution based on topography and erodibility

import numpy as np
import matplotlib.pyplot as plt

from channelbelt2d.distributions import PotentialCalculator, GibbsDistribution

# Define the x-axis
x_min = -1000.0
x_max = 1000.0
n_x = 100
x = np.linspace(x_min, x_max, n_x)

depth = np.zeros(n_x)
erodibility = np.zeros(n_x)

# Define regions
x_left_ramp = [-500, -200]
x_hi = [-200, 300]
x_right_ramp = [300, 550]
erodibility[np.logical_and(x_hi[0] <= x, x <= x_hi[1])] = 1.0

# Make the high-erodibility region a topographic high (lower depth)
# Make continuous ramps on either side of the high-erodibility region
for i, x_i in enumerate(x):
    if x_left_ramp[0] <= x_i <= x_left_ramp[1]:
        depth[i] = -(x_i - x_left_ramp[0]) / (x_left_ramp[1] - x_left_ramp[0])
    elif x_right_ramp[0] <= x_i <= x_right_ramp[1]:
        depth[i] = -(x_right_ramp[1] - x_i) / (x_right_ramp[1] - x_right_ramp[0])
    elif x_hi[0] <= x_i <= x_hi[1]:
        depth[i] = -1


# Define the potential contributions
gamma = 2.0
beta = -3.0

potential_contributions = {'topography': {'covariate': depth, 'coefficient': -gamma},
                            'erodibility': {'covariate': erodibility, 'coefficient': beta}}

potential_calculator = PotentialCalculator(np.linspace(x_min, x_max, 100), potential_contributions)
potential = potential_calculator.compute_potential()
distribution = GibbsDistribution(potential_calculator._xx, potential)


# plot the topography and erodibility
fig, ax = plt.subplots(1, 1, figsize=(8, 4))
ax.plot(x, depth, label='Depth', color='black')
ax.set_ylim(-5, 1)
ax.invert_yaxis()
ax.set_xlabel('x')
ax.set_ylabel('Depth')

# Plot erodibility with its own y-axis
ax2 = ax.twinx()
ax2.plot(x, erodibility, label='Erodibility', color='red')
ax2.set_xlabel('x')
ax2.set_ylabel('Erodibility')

# Add a legend
ax.legend(loc='upper left')
ax2.legend(loc='upper right')


# Plot the CDF, PDF, and histogram of samples
fig, ax = plt.subplots(1, 3, figsize=(12, 4))

distribution.plot_cdf(ax[0])
ax[0].set_title('CDF')

distribution.plot_pdf(ax[1])
ax[1].set_title('PDF')

samples = [distribution.draw() for _ in range(1000)]
ax[2].hist(samples, bins=20, density=True)

plt.show()
