import numpy as np
from scipy.fftpack import fft2, fftshift
import matplotlib.pyplot as plt
import matplotlib as mpl

# Distances are supposed to be in mm (650 nm for red laser)
shape      = (2000, 2000)
sheet_Lx   = 36
sheet_Ly   = 36
radius     = 2
distance   = 1000
wavelength = 6.50e-4
wavenumber = 2 * np.pi / wavelength

x = np.linspace(-sheet_Lx / 2, +sheet_Lx / 2, shape[0] + 1)[:-1, None]
y = np.linspace(-sheet_Ly / 2, +sheet_Ly / 2, shape[1] + 1)[None, :-1]

# Fresnel diffraction formula
aperture  = x**2 + y**2 > radius**2
kernel    = np.exp(1j * wavenumber / (2 * distance) * (x**2 + y**2) )
amplitude = fftshift(fft2(aperture * kernel))
power     = np.abs(amplitude)**2
power    /= power[shape[0]//2, shape[1]//2]

screen_Lx = 0.5 * distance * wavelength * shape[0] / sheet_Lx
screen_Ly = 0.5 * distance * wavelength * shape[1] / sheet_Ly
extent    = ( - screen_Lx, screen_Lx * (1 - 2/shape[0]) , - screen_Ly, screen_Ly * (1 - 2/shape[1]) )

# Plotting
cmap      = mpl.colors.LinearSegmentedColormap.from_list("", [(0.0, 'black'), (1.0, 'red')])

fig, ax = plt.subplots(1,1, figsize = (10,10))
ax.axis('off')
ax.set_xlim(-5,5)
ax.set_ylim(-5,5)

ax.imshow(power.T, cmap = cmap, interpolation = "bilinear", origin = 'lower', extent = extent, vmin = 0, vmax = 1.4)
plt.savefig('MIB0034_AragoSpot.png', dpi = fig.dpi, bbox_inches='tight', pad_inches = 0)
