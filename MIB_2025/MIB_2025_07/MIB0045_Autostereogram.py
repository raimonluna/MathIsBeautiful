import numpy as np
import matplotlib.pyplot as plt

shape        = (1024, 1024)
xrange       = (-1.7, 1.3)
yrange       = (-1.3, 1.7)
cells        = (16, 16)
peak         = 128
variation    = 50 
np.random.seed(42)

x = np.linspace(*xrange, shape[0])[:, None]
y = np.linspace(*yrange, shape[1])[None, :]

def perlin_noise(cells = (5,5), npix = 32):
    xy = np.mgrid[0:cells[0]:1/npix, 0:cells[1]:1/npix].transpose(1, 2, 0) % 1
    sm = 6*xy**5 - 15*xy**4 + 10*xy**3 # Smoothstep function
    
    angles = 2 * np.pi * np.random.rand(*cells)
    grads  = np.dstack([np.cos(angles), np.sin(angles)])
    cshape = np.ones((npix, npix, 1))
    
    perlin = 0
    for di, dj in [(di, dj) for di in (0, 1) for dj in (0, 1)]:
        dot_product = np.kron(np.roll(grads, (-di, -dj), axis = (0, 1)), cshape) * (xy - [di, dj])
        smoothing   = sm + (1 - 2 * sm) * [1 - di, 1 - dj]
        perlin     += np.sqrt(2) * np.prod(smoothing, axis = 2) * np.sum(dot_product, axis = 2)

    return perlin

def fractal_noise(cells = (5,5), npix = 32, octaves = 5):
    return np.sum([perlin_noise((2**n * cells[0], 2**n * cells[1]), npix // 2**n) / 2**n for n in range(octaves)], axis = 0)

# Extract the shape from implicit equation via Newton-Raphson
f = 10 * np.ones(shape)
for _ in range(500):
    numerator   = (x**2 + 9 * f**2 / 4 + y**2 -1)**3 - x**2 * y**3 - 9 * f**2 * y**3 / 80
    denominator = 81 * f * (x**2 + 9 * f**2 / 4 + y**2 -1)**2 / 2 - 18 * f * y**3 / 80
    correction  = numerator / denominator
    f = np.abs(f - correction)
f[np.abs(correction) > 1e-5] = 0

# Perform periodicity adjustments over fractal Perlin noise background
canvas    = fractal_noise(cells = cells, npix = shape[1] // cells[1])
all_range = range(shape[0])
for col in range(shape[1]):
    canvas[col, all_range] = canvas[col - peak + np.int64(variation * f[col, :]), all_range]

# Do the actual plotting
fig, ax = plt.subplots(1,1, figsize = (8,8))

ax.axis('off')
ax.imshow(canvas.T, cmap = 'tab20b', origin = 'lower')
plt.savefig('MIB0045_Autostereogram.png', dpi = 1000, bbox_inches='tight', pad_inches = 0)
