import numpy as np
import matplotlib.pyplot as plt

from channelbelt2d.objects import BraidedBelt
from channelbelt2d.distributions import TopographicLowDistribution
from channelbelt2d.environments import Valley, ValleyParameters

#################################################################
# Testing: Braided river
# Draw a sample from the channel center position distribution,
# then create a braided belt at that position and plot it.
#################################################################
valley_parameters = ValleyParameters(width=100, depth=100)
valley = Valley(valley_parameters)

# Initialize topo low distribution using valley.get_y_point as the depth function
topo_distr_02 = TopographicLowDistribution(valley.get_y(), -50, 50, 20, gamma=0.2)
topo_distr_1 = TopographicLowDistribution(valley.get_y(), -50, 50, 20, gamma=1.0)
topo_distr_5 = TopographicLowDistribution(valley.get_y(), -50, 50, 20, gamma=5.0)

# Plot the CDF and PDF of the distribution
fig, ax = plt.subplots(3, 1, figsize=(6, 6))
topo_distr_02.plot_cdf(ax[0], label=r'$\gamma = 0.2$')
topo_distr_02.plot_pdf(ax[1], label=r'$\gamma = 0.2$')
topo_distr_1.plot_cdf(ax[0], label=r'$\gamma = 1$')
topo_distr_1.plot_pdf(ax[1], label=r'$\gamma = 1$')
topo_distr_5.plot_cdf(ax[0], label=r'$\gamma = 5$')
topo_distr_5.plot_pdf(ax[1], label=r'$\gamma = 5$')

ax[0].legend()
ax[1].legend()

# Draw a sample from the distribution
x_bb1 = topo_distr_1.draw()
w_bb1 = 20

xx_bb1 = np.linspace(x_bb1 - w_bb1/2, x_bb1 + w_bb1/2, 100)
base_bb1 = np.interp(xx_bb1, valley.get_x(), valley.get_y())
ha_bb1 = 10
he_bb1 = 5

el = 4
init_lvl = 100
lvl = init_lvl - ha_bb1 + el

valley.set_level(lvl)
valley.plot(ax[2])

braided_belt_1 = BraidedBelt(x_bb1, base_bb1, w_bb1, ha_bb1, he_bb1)
braided_belt_1.plot(ax[2])

ax[2].invert_yaxis()
plt.show()