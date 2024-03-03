import matplotlib.pyplot as plt
from scipy.stats import norm

from channelbelt2d.objects import WingedBeltObject, WingedBeltParameterRealization, WingedBeltParameterDistribution

distribution = WingedBeltParameterDistribution(
    left_wing_width=norm(loc=250, scale=50),
    right_wing_width=norm(loc=250, scale=50),
    base_belt_width=norm(loc=500, scale=60),
    top_belt_width=norm(loc=600, scale=80),
    belth_thickness=norm(loc=8, scale=0.6),
    superelevation=norm(loc=2, scale=0.5)
)

realization = distribution.draw_realization()

winged_belt = WingedBeltObject(center_location=0,
                               floodplain_elevation=0,
                               params=realization)

# Plot the winged belt object
fig = plt.figure()
ax = fig.add_subplot(111)

winged_belt.plot(ax, n=100)

ax.set_xlim(-1000, 1000)
ax.set_ylim(-25, 50)

ax.invert_yaxis()

plt.show()