import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
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
grid_range = 0.15  # Range of the grid in meters

# Source location (at the center of the grid)
source_x = 0.05  # Source x-coordinate (m)
source_y = 0.0  # Source y-coordinate (m)

# Create grid
x = np.linspace(-grid_range/2, grid_range/2, grid_size)
y = np.linspace(-grid_range/3, grid_range/3, grid_size)
X, Y = np.meshgrid(x, y)

# Compute temperature for each point on the grid
R = np.sqrt((X - source_x)**2 + (Y - source_y)**2)  # Distance from the point heat source
temperature = t0 + (q / (2 * np.pi * k * R)) * np.exp(-(v / (2 * a)) * (X - source_x + R))
temperature_capped = np.clip(temperature, None, temp_cap)
temperature_capped -= 273.15

# # Plot the 2D heatmap
# imratio = temperature_capped.shape[0]/temperature_capped.shape[1]
# plt.figure(0, figsize=(8, 6))
# plt.imshow(temperature_capped, cmap='gnuplot', extent=[x.min(), x.max(), y.min(), y.max()], origin='lower')
# levels = [200, 400, 600, 800, 1000, 1200, 1400, 1600]
# # divider = make_axes_locatable(plt.gca())
# # cax = divider.append_axes('right', size='5%', pad=0.05)
# cbar = plt.colorbar(label=r'Temperature (C$^o$)', fraction=0.0308*imratio)
# cbar.ax.tick_params(axis='y', which='both', direction='in', right=True, labelright=True)
# contour = plt.contour(X, Y, temperature_capped, levels=levels, colors='w', linewidths=0.75)
# plt.clabel(contour, inline=True, fontsize=8, colors='k')
# plt.gca().xaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: '{:.0f}'.format(x * 1000)))
# plt.gca().yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: '{:.0f}'.format(y * 1000)))
# plt.xlabel('X Distance (mm)', fontsize=9)
# plt.ylabel('Y Distance (mm)', fontsize=9)
# plt.xticks(fontsize=9)
# plt.yticks(fontsize=9)
# # plt.title('Weld Thermal Profile Heatmap (P = 1200 W)')
# plt.tight_layout()
# # plt.savefig('Heatmap', dpi=300)

source_x_index = source_x / (grid_range * 0.5) * 500 + 500
first_x = np.where(temperature_capped[500, :] == 1933.15-273.15)[0][0]
last_x = np.where(temperature_capped[500, :] == 1933.15-273.15)[0][-1]
first_y = np.where(temperature_capped[:, 790] == 1933.15-273.15)[0][0]
last_y = np.where(temperature_capped[:, 790] == 1933.15-273.15)[0][-1]

pool_length = X[0, last_x] - X[0, first_x]
pool_width = Y[last_y, 0] - Y[first_y, 0]

# plt.figure(1, figsize=(8, 6))
# centerline_temperature = temperature_capped[500, :]
# plt.plot(X[0, :], centerline_temperature, color='k', label='Temperature Profile')
# plt.vlines(x=[X[0, first_x], X[0, last_x]],
#            ymin=-100,
#            ymax=np.max(centerline_temperature+200),
#            label='Melt Pool Boundary',
#            linestyle='--',
#            color='k')
# plt.ylim(0, np.max(centerline_temperature+100))
# plt.grid()
# plt.xlabel('X Distance (mm)')
# plt.ylabel(r'Temperature (C$^o$)')
# plt.legend()
# plt.savefig('Longitudinal Temperature Profile')
#
# plt.figure(2, figsize=(8, 6))
# transverse_temperature = temperature_capped[:, 790].reshape(-1, 1)
# plt.plot(Y[:, 0], transverse_temperature, color='k', label='Temperature Profile')
# ylines = [Y[first_y, 0], Y[last_y, 0]]
# plt.vlines(x=ylines,
#            ymin=-100,
#            ymax=np.max(transverse_temperature)+200,
#            label='Melt Pool Boundary',
#            linestyle='--',
#            color='k')
# plt.ylim(0, np.max(centerline_temperature+100))
# plt.grid()
# plt.xlabel('Y Distance (mm)')
# plt.ylabel(r'Temperature (C$^o$)')
# plt.legend()
# plt.savefig('Transverse Temperature Profile')

t = np.linspace(0, 50, 1000)
T = t0 + q / (2 * np.pi * k * v * t) - 273.15
min_index = np.argmin(np.abs(T-900))
max_index = np.argmin(np.abs(T-950))
plt.plot(t, T, color='k', label='Temperature')
plt.ylim(0, temp_cap)
plt.xlim(0, 50)
plt.hlines(y=950,
           xmin=-10,
           xmax=t[max_index],
           label=r'$\beta$ Transus Point',
           linestyle='--',
           color='k')
plt.vlines(x=t[max_index],
           ymin=-10,
           ymax=950,
           linestyle='--',
           color='k')
plt.text(t[max_index], -95, f'{round(t[max_index], 2)}', ha='center')
plt.xlabel('Time (s)')
plt.ylabel(r'Temperature (C$^o$)')
plt.grid()
plt.tight_layout()
plt.legend()
# plt.savefig('Cooling Rate', dpi=300)

plt.show()

debug = 0
