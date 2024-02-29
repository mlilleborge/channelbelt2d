import matplotlib.pyplot as plt
from channelbelts import BraidedRiverDeposition, ValleyParameters, BeltParameters
from scipy.stats import norm

####################################################################
# Example usage: Braided river
# Recreate fig. 11 in Boggs (2014).
#
# Figure caption:
# FIGURE 11 Schematic representation of the fluvial architecture of
# braided-river deposits.
# [After Walker, R.G., and D. J. Cant, 1984,
# Sandy fluvial systems, in R. G. Walker (ed.), Facies models:
# Geoscience Canada Reprint Ser. 1, Fig. 9, p. 77,
# reprinted by permission of Geological Association of Canada.]
#
# Corresponding caption in Walker and Cant (1984):
# Figure 8B
# Block diagram of a braided sandy system with low sinuosity
# channels. Vertical accretion can occur during flood stage,
# for aexample on the vegetated island, but deposits are rarely
# preserved. Diagrams modified from those in Allen (1965).
#
# Reference cited:
# Allen, J. R. (1965). A review of the origin and characteristics of
# recent alluvial sediments. Sedimentology, 5(2), 89-191.
# DOI: 10.1111/j.1365-3091.1965.tb01561.x
# (see fig. 35, p. 164)
####################################################################
fig, ax = plt.subplots(figsize=(12, 4))

# Define valley geometry
valley_parameters = ValleyParameters(width=100, depth=100, initial_level=90)

# Create a meandering river deposit
belt_parameters = BeltParameters(aggradation = norm(loc=6, scale=1),
                                 channel_depth = norm(loc=5, scale=1),
                                 belt_elevation = norm(loc=0.01, scale=0.001),
                                 belt_width = norm(loc=20, scale=5))

# Initialize depositional process
depositional_process = BraidedRiverDeposition(valley_parameters, belt_parameters, gamma=0.1)

# Prepare to make an animation
depositional_process._plot_valley(ax)   # Plot the valley

# Let the process run
number_of_belts = 20
try:
    for i in range(number_of_belts):
        depositional_process.draw_belt()
        
        ax.clear() # Clear the axis
        depositional_process._plot_valley(ax)   # Plot the valley
        depositional_process._plot_belts(ax)    # Plot all the meander belts deposited so far

        ax.invert_yaxis() # Reverse the y-axis

        plt.xlim(-50, 50)
        plt.ylim(110, 0)
        ax.set_axis_off()

        # Save frame
        plt.savefig(f"braided_{i:03d}.png")
except ValueError:
    print(f"Process halted after {i} events because valley is full.")


# Combine frames to make an animation
actual_number_of_belts = i
import os
import imageio
filenames = [f"braided_{i:03d}.png" for i in range(actual_number_of_belts)]
with imageio.get_writer('braided.gif', mode='I') as writer:
    for filename in filenames:
        image = imageio.imread(filename)
        for j in range(5):
            writer.append_data(image)
        

# Clean up
for filename in set(filenames):
    os.remove(filename)


# Plot resulting deposit
#depositional_process._plot_valley(ax)   # Plot the valley
#depositional_process._plot_belts(ax)    # Plot all the meander belts

#ax.invert_yaxis() # Reverse the y-axis

#plt.xlim(-50, 50)
#plt.ylim(110, 0)
# turn axis off
#ax.set_axis_off()

#plt.show()