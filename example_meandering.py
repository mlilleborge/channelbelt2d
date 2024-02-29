import matplotlib.pyplot as plt
from channelbelts import MeanderingRiverDeposition, ValleyParameters, BeltParameters
from scipy.stats import norm

####################################################################
# Example usage: Meandering river
# Recreate fig. 12 on p. 222 of Boggs (2014).
#
# Figure caption:
# FIGURE 12 Schematic representation of the fluvial architecture of
# meandering-river deposits.
# [After Walker, R. G., and D. J. Cant, 1984, Sandy fluvial systems,
# in R. G. Walker (ed.), Facies models:
# Geoscience Canada Reprint Ser. 1, Fig. 9, p. 77,
# reprinted by permission of Geological Association of Canada.]
#
# Corresponding caption in Walker and Cant (1984):
# Figure 8A
# Block diagram of flood-plain aggradation with very sinuous
# rivers. Shoestring sands are preserved, and are surrounded by
# vertical accretion siltstones and mudstones. Vertical scale is
# highly exaggerated.
####################################################################
fig, ax = plt.subplots(figsize=(8, 4))

# Define valley geometry
valley_parameters = ValleyParameters(width=100, depth=100, initial_level=90)

# Create a meandering river deposit
belt_parameters = BeltParameters(aggradation = norm(loc=10, scale=0.5),
                                 channel_depth = norm(loc=0.5, scale=0.05),
                                 belt_elevation = norm(loc=1.0, scale=0.13),
                                 belt_width = norm(loc=18, scale=6))

# Initialize depositional process
depositional_process = MeanderingRiverDeposition(valley_parameters, belt_parameters)

# Let the process run
number_of_belts = 5
for i in range(number_of_belts):
    depositional_process.draw_belt()

# Plot resulting deposit
depositional_process._plot_valley(ax)   # Plot the valley
depositional_process._plot_belts(ax)    # Plot all the meander belts

ax.invert_yaxis() # Reverse the y-axis
plt.show()