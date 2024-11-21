import numpy as np
from tqdm.auto import tqdm
import matplotlib.pyplot as plt

shape        = (2000, 2000)
max_radius   = 10
max_iter     = 100
xrange       = (-1.9, 0.9)
yrange       = (-1.4, 1.4)

buddha_iters = 10000
buddha_batch = 1000000

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

def add_to_hist(histogram, buddha_z):
    to_plot = buddha_z
    idx1    = np.int32( shape[0] * (to_plot.real - xrange[0]) / (xrange[1] - xrange[0]) )
    idx2    = np.int32( shape[1] * (to_plot.imag - yrange[0]) / (yrange[1] - yrange[0]) )
    inside  = (0 <= idx1) & (idx1 < shape[0]) & (0 <= idx2) & (idx2 < shape[0])
    histogram[idx1[inside], idx2[inside]] += 1

##### Extract boundary points

rec    = np.linspace(*xrange, shape[0])[:, None]
imc    = np.linspace(*yrange, shape[1])[None, :]
dr, di = rec[1,0] - rec[0,0], imc[0,1] - imc[0,0]
c      = rec + 1j * imc
mandel = mandelbrot(c, max_iter, max_radius)
edges  = c[mandel].ravel()

##### Do the sampling

histogram = np.float64(mandel)
batch     = new_batch(edges, buddha_batch, dr, di)
buddha_z  = np.zeros_like(batch, dtype = np.complex128)

for it in tqdm(range(buddha_iters)):

    add_to_hist(histogram, buddha_z)
    buddha_z  = buddha_z**2 + batch

    discarded = np.abs(buddha_z) > 2
    buddha_z[discarded] = 0
    batch[discarded]    = new_batch(edges, np.sum(discarded), dr, di)

plt.axis('off')
plt.imshow(histogram + 200, norm = 'log', cmap = 'hot', interpolation = 'bilinear')
plt.savefig('MIB0019_AntiBuddhabrot.png', dpi = 1000, bbox_inches='tight', pad_inches = 0)
