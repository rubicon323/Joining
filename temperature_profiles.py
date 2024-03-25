import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

# Constants
t0 = 296  # Initial temperature of the workpiece (Kelvin)
q = 1200  # Effective arc power (Watts)
k = 6.3   # Thermal conductivity (W/(m*K))
v = 0.005  # Welding speed (m/s)
a = 2.7 * 10e-6  # Thermal diffusivity (m^2/s)
temp_cap = 1933.15

# Grid size
grid_size = 1000  # Number of points along each axis
grid_range = 0.3  # Range of the grid in meters

# Source location (at the center of the grid)
source_x = 0.1  # Source x-coordinate (m)
source_y = 0.0  # Source y-coordinate (m)

# Create grid
x = np.linspace(-grid_range/2, grid_range/2, grid_size)
y = np.linspace(-grid_range/3, grid_range/3, grid_size)
X, Y = np.meshgrid(x, y)

# Compute temperature for each point on the grid
R = np.sqrt((X - source_x)**2 + (Y - source_y)**2)  # Distance from the point heat source
temperature = t0 + (q / (2 * np.pi * k * R)) * np.exp(-(v / (2 * a)) * (X - source_x + R))
temperature_capped = np.clip(temperature, None, temp_cap)

# Plot the 2D heatmap
plt.figure(figsize=(8, 6))
plt.imshow(temperature_capped, cmap='turbo', extent=[x.min(), x.max(), y.min(), y.max()], origin='lower')
levels = [400, 600, 800, 1000, 1200, 1400, 1600, 1800]
cbar = plt.colorbar()
cbar.ax.tick_params(axis='y', which='both', direction='in', right=True, labelright=True)
contour = plt.contour(X, Y, temperature_capped, levels=levels, colors='black', linewidths=1)
plt.xlabel('X Distance (m)')
plt.ylabel('Y Distance (m)')
plt.title('Weld Thermal Profile Heatmap (P = 1200 W)')
plt.savefig('Heatmap', dpi=300)
plt.show()
