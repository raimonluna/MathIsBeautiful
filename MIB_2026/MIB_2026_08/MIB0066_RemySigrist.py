import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from tqdm.auto import tqdm
np.random.seed(42)

print('Now creating sequence...')

series, highest, used_bits = [0], [0], [0] * 2**13
for n in tqdm(range(2, 2**13)):
    
    chosen = 0
    while (used_bits[chosen] & n) != 0:
        chosen += 1
        
    series.append(chosen)
    used_bits[chosen] |= n

    if chosen > highest[-1]:
        highest.append(chosen)
    else:
        highest.append(highest[-1])

series = np.array(series)

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
    
noise = fractal_noise(npix = 128)

print('Now creating animation...')
fig, ax = plt.subplots(1,1, figsize = (6,6))
fig.subplots_adjust(left=0, bottom=0, right=1, top=1, wspace=None, hspace=None)
ax.axis('off')
plt.close()

for k in np.linspace(0, 100, 20):
    ax.plot(np.arange(len(series)) + k, series + 0.001 * k * (k - 500), '.', color = 'black', ms = .5, alpha = 0.3*(1 - k/100))

ax.fill(list(range(1, 2**13)) + [2**13], highest + [0], color = 'ghostwhite', alpha = 0.5)
ax.fill(list(range(1, 2**13)) + [2**13], np.minimum(highest[::-1] + [highest[-1]], highest + [0]), color = 'white', alpha = 0.5)

ish = ax.imshow(noise, cmap = 'binary', extent = (0, 2**13, 0, 3404))
ax.set_xlim(0, 2**13)
ax.set_ylim(0, 3404)
ax.set_aspect(2**13/3404)

animation_fig = animation.FuncAnimation(fig, lambda i: ish.set_data(np.roll(noise, i)), frames = len(noise), interval = 50)
animation_fig.save("MIB0066_RemySigrist.mp4", dpi = 200)
