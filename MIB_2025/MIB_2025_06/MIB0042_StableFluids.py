import numpy as np
from numpy.fft import fft2, ifft2, fftfreq
from scipy.interpolate import RegularGridInterpolator
from tqdm.auto import tqdm
import matplotlib.pyplot as plt
import matplotlib.animation as animation

shape  = (600, 600)
xrange = (0, 1)
yrange = (0, 1)
eta    = 0.0001
dt     = 0.005
iters  = 750

x  = np.linspace(*xrange, shape[0] + 1)[:-1, None]
y  = np.linspace(*yrange, shape[1] + 1)[None, :-1]

kx = shape[0] * fftfreq(shape[0])[:, None]
ky = shape[1] * fftfreq(shape[1])[None, :]
k  = np.sqrt(kx**2 + ky**2)

kernel   = 1 / (1 + dt * eta * k**2)
force_x  = 100 * np.exp( - ((x - 0.2)**2 + (y - 0.45)**2) / 0.1**2 )
force_x -= 100 * np.exp( - ((x - 0.8)**2 + (y - 0.55)**2) / 0.1**2 )
curl     = np.zeros(shape + (iters,))  

k[k == 0] = 1
nkx, nky  = kx / k, ky / k
vx, vy    = 0*k, 0*k

print('Evolving fluid...')
for it in tqdm(range(iters)):

    #External force
    vx += dt * max(1 - it * dt, 0) * force_x

    #Advection
    interp_x = RegularGridInterpolator((x.ravel(), y.ravel()), vx)
    interp_y = RegularGridInterpolator((x.ravel(), y.ravel()), vy)
    newcoord = np.mod(x - vx * dt, x[-1, 0]), np.mod(y - vy * dt, y[0, -1])
    vx, vy   = interp_x(newcoord), interp_y(newcoord)

    #Diffusion
    vxf, vyf = fft2(vx), fft2(vy)
    vxf, vyf = vxf * kernel, vyf * kernel

    #Divergence removal
    div      = vxf * nkx + vyf * nky
    vxf, vyf = vxf - div * nkx, vyf - div * nky
    vx, vy   = ifft2(vxf).real, ifft2(vyf).real

    #Compute vorticity
    dvx_dy = np.roll(vx, -1, axis = 1) - np.roll(vx, 1, axis = 1)
    dvy_dx = np.roll(vy, -1, axis = 0) - np.roll(vy, 1, axis = 0)
    curl[..., it] = dvx_dy - dvy_dx

def update(i):
    ish.set_data(curl[..., i].T)
    ish.autoscale()

##### Do the plotting
print('Making the animation...')

fig, ax = plt.subplots(frameon = False, figsize = (5, 5))
fig.subplots_adjust(left=0, bottom=0, right=1, top=1, wspace=None, hspace=None)
ax.axis('off')
plt.close()

ish = ax.imshow(curl[..., 0].T, cmap = 'ocean', interpolation = 'bilinear', origin = 'lower', extent = (0, 1, 0, 1))

animation_fig = animation.FuncAnimation(fig, update, frames = iters, interval = 50)
animation_fig.save("MIB0042_StableFluids.mp4", dpi = 200)
