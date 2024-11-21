import numpy as np
from tqdm.auto import tqdm
import matplotlib.pyplot as plt
from scipy.ndimage import sobel

shape        = (2000, 2000)
max_radius   = 10
max_iter     = 100
xrange       = (-1.9, 0.9)
yrange       = (-1.4, 1.4)

buddha_iters = 10000
buddha_batch = 10000
buddha_lifes = (5000, 500, 50)

np.random.seed(42)

##### Some useful functions

def mandelbrot(c, max_iter, max_radius):
    z = 0 * c
    diverged = z == 2
    for i in range(max_iter):
        z = z**2 + c
        diverged = diverged | (np.abs(z) > max_radius)
        z[diverged] = 0
    return ~diverged

def new_batch(edges, size, dr, di):
    batch_ind = np.random.randint(0, len(edges), size = size)
    rand_real = np.random.uniform(-dr/2, dr/2,   size = size)
    rand_imag = np.random.uniform(-di/2, di/2,   size = size)
    return edges[batch_ind] + rand_real + 1j * rand_imag

def add_to_hist(histogram, history, successful, channel):
    to_plot = history[successful, :].ravel()
    idx1    = np.int32( shape[0] * (to_plot.real - xrange[0]) / (xrange[1] - xrange[0]) )
    idx2    = np.int32( shape[1] * (to_plot.imag - yrange[0]) / (yrange[1] - yrange[0]) )
    inside  = (0 <= idx1) & (idx1 < shape[0]) & (0 <= idx2) & (idx2 < shape[0])
    histogram[idx1[inside], idx2[inside], channel] += 1

##### Extract boundary points

rec    = np.linspace(*xrange, shape[0])[:, None]
imc    = np.linspace(*yrange, shape[1])[None, :]
dr, di = rec[1,0] - rec[0,0], imc[0,1] - imc[0,0]
c      = rec + 1j * imc
mandel = mandelbrot(c, max_iter, max_radius)

edges  = np.abs(sobel(mandel, axis=0)) + np.abs(sobel(mandel, axis=1))
edges  = c[edges].ravel()

##### Do the sampling

histogram = np.zeros(shape + (3,))

for channel in range(3):
    print('Sampling ', ('red', 'green', 'blue')[channel], '...')
    
    batch     = new_batch(edges, buddha_batch, dr, di)
    buddha_z  = np.zeros_like(batch, dtype = np.complex128)
    survival  = np.zeros(buddha_batch, dtype = np.int32)
    history   = 10 * np.ones((buddha_batch, buddha_lifes[channel] + 1),
                             dtype = np.complex128)
    
    for it in tqdm(range(buddha_iters)):
        buddha_z  = buddha_z**2 + batch
        history[range(buddha_batch), survival] = buddha_z
        survival += 1
        
        successful = np.abs(buddha_z) > 2
        lived_long = survival >= buddha_lifes[channel]
        discarded  = successful | lived_long    
    
        add_to_hist(histogram, history, successful, channel)
    
        buddha_z[discarded] = survival[discarded] = 0
        history[discarded, :] = 10
        batch[discarded] = new_batch(edges, np.sum(discarded), dr, di)

##### Plotting the Buddhabrot

rescaled_hist  = histogram / np.max(histogram, axis = (0,1))
rescaled_hist  = np.log(rescaled_hist + np.array([1e-1, 2e-2, 1e-3]))
rescaled_hist -= np.min(rescaled_hist, axis = (0,1))
rescaled_hist /= np.max(rescaled_hist, axis = (0,1))
rescaled_hist = np.minimum(1.6 * rescaled_hist, 1.0)

plt.axis('off')
plt.imshow(rescaled_hist, interpolation = 'bilinear')
plt.savefig('MIB0018_Buddhabrot.png', dpi = 1000, bbox_inches='tight', pad_inches = 0)
