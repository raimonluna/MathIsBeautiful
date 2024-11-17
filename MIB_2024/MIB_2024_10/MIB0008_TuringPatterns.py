import numpy as np
import cupy  as cp
from tqdm.auto import tqdm
import matplotlib.pyplot as plt
import matplotlib.animation as animation

shape        = (600,  600)
xrange       = (-1000, 1000)
yrange       = (-1000, 1000)

CFL          = 0.5
tmax         = 10000
save_every   = 10
(Du, Dv)     = (1.0, 2.0)
(a, b)       = (0.037, 0.06)

###### Prepare the grids
x = cp.linspace(*xrange, shape[0] + 1)[:-1, None]
y = cp.linspace(*yrange, shape[1] + 1)[None, :-1]
dx, dy = x[1, 0] - x[0, 0], y[0, 1] - y[0, 0]
dt     = CFL * cp.minimum(dx, dy)
iters  = cp.int64(cp.asnumpy(tmax // dt)) + 1
frames = (iters - 1)// save_every + 1

F         = cp.zeros(shape + (2,), dtype=cp.float32)
stored    = np.zeros(shape + (frames,))

###### Set random initial conditions
np.random.seed(42)
for i in range(10):
    x0, y0 = 2000 * np.random.rand(2) - 1000
    F[..., 0] += cp.exp(- ((x - x0)**2 + (y - y0)**2) / 10)
F[..., 1] = 1

###### Gray-Scott Equations
def rhs(t, F):
    d2F_dx2 = (cp.roll(F, 1, axis=0) - 2*F + cp.roll(F, -1, axis=0)) / dx**2
    d2F_dy2 = (cp.roll(F, 1, axis=1) - 2*F + cp.roll(F, -1, axis=1)) / dy**2
    laplace = d2F_dx2 + d2F_dy2

    u, v    = F.transpose(2,0,1)
    lu, lv  = laplace.transpose(2,0,1)

    dX_dt         = cp.zeros_like(F)
    dX_dt[..., 0] = Du * lu + u**2 * v - (a + b) * u
    dX_dt[..., 1] = Dv * lv - u**2 * v + a * (1 - v)

    return dX_dt

###### Runge-Kutta 4th order step
def RK4(t, F):
    k1 = rhs(t, F)
    k2 = rhs(t + 0.5 * dt, F + 0.5 * k1 * dt)
    k3 = rhs(t + 0.5 * dt, F + 0.5 * k2 * dt)
    k4 = rhs(t + dt, F + k3 * dt)
    return t + dt, F + dt * (k1 + 2*k2 + 2*k3 + k4) / 6

###### Time evolution
for i in tqdm(range(iters)):
    if i % save_every == 0:
        stored[..., i//save_every] = cp.asnumpy(F[..., 0])
    _, F = RK4(0, F)
print('Simulation finished!')

###### Make the animation

fig, ax = plt.subplots(frameon=False, figsize= (8,8))
fig.subplots_adjust(left=0, bottom=0, right=1, top=1, wspace=None, hspace=None)
ish = ax.imshow(stored[..., 0].T, cmap = 'ocean', interpolation = 'bilinear' , origin = 'lower', extent = xrange + yrange)

def update(i):
    ii = min(i, frames - 1)
    ish.set_data(stored[..., ii].T)
    ish.autoscale()

ax.axis('off')
animation_fig = animation.FuncAnimation(fig, update, frames = frames + 100, interval = 20)
animation_fig.save("MIB0008_TuringPatterns.mp4", dpi = 200)

print('Animation finished!')
