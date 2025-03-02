import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# Data (Year, Value1, Value2)
data = np.array([
    [2022, 1424117394, 1896047335],
    [2021, 1525486540, 1893143269],
    [2020, 1609229857, 1805655202],
    [2019, 1535046668, 2124916982],
    [2018, 1551441077, 2172490874],
    [2017, 1584877006, 2159883548],
    [2016, 1602218919, 2012774683],
    [2015, 1632646065, 1954766502],
    [2014, 1650323202, 1890155436],
    [2013, 1632305132, 1927600374],
    [2012, 1663571950, 1862786430]
])

# New z-axis data points
z_new = np.array([37.2, 30.0, 33.9, 33.9, 31.1, 30.0, 31.1, 32.2, 27.8, 
                  31.1, 27.2])

# Extracting x, y, and using new z values
x = data[:, 0]  # Year
y = data[:, 1]  # Value 1
z = z_new       # New z values from the provided list

# Prepare 3D plot
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

# Scatter plot for actual data
ax.scatter(x, y, z, color='b', label='Data')

# Labels and grid
ax.set_xlabel('Year')
ax.set_ylabel('Value 1')
ax.set_zlabel('New Value (z)')
ax.grid(True)
plt.legend()

# Show plot
plt.show()
