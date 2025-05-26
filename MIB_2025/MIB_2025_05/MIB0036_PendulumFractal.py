import numpy as np
import cupy  as cp
from tqdm.auto import tqdm
import matplotlib.pyplot as plt

shape = (1000, 1000)
tmax  = 1000
dt    = 0.05
check_every = 50

iters   = cp.int64(cp.asnumpy(tmax // dt)) + 1
theta   = cp.linspace(-cp.pi, cp.pi, shape[0] + 1)[:-1, None]
theta   = theta + 0 * theta.T
flip_t  = 0 * theta + tmax
X       = cp.dstack([theta, theta.T, 0*theta, 0*theta])

###### Equations of Motion
def rhs(t, X):
    t1, t2, p1, p2 = X.transpose(2,0,1)

    cosdt = cp.cos(t1 - t2)
    sindt = cp.sin(t1 - t2)
    sint1 = cp.sin(t1)
    sint2 = cp.sin(t2)

    dt1 = 6 * ( 2 * p1 - 3 * cosdt * p2 ) / ( 16 - 9 * cosdt**2 )
    dt2 = 6 * ( 8 * p2 - 3 * cosdt * p1 ) / ( 16 - 9 * cosdt**2 )
    dp1 = - 0.5 * ( 3 * sint1 + dt1 * dt2 * sindt )
    dp2 = - 0.5 * ( 1 * sint2 - dt1 * dt2 * sindt )

    return cp.dstack([dt1, dt2, dp1, dp2])
    
###### Runge-Kutta 4th order step
def RK4(t, X):
    k1 = rhs(t, X)
    k2 = rhs(t + 0.5 * dt, X + 0.5 * k1 * dt)
    k3 = rhs(t + 0.5 * dt, X + 0.5 * k2 * dt)
    k4 = rhs(t + dt, X + k3 * dt)
    return t + dt, X + dt * (k1 + 2*k2 + 2*k3 + k4) / 6

###### Time evolution
t = 0.0
for i in tqdm(range(iters)):
    t, X = RK4(t, X)
    if i % check_every == 0:
        flipped_1 = (X[..., 0] < - cp.pi) | (X[..., 0] > cp.pi) 
        flipped_2 = (X[..., 1] < - cp.pi) | (X[..., 1] > cp.pi) 
        flip_t[(flip_t == tmax) & (flipped_1 | flipped_2)] = t    

###### Make the plot
fig, ax = plt.subplots(1,1, figsize = (10, 10))
ax.axis('off')

ax.imshow(cp.asnumpy(flip_t).T, cmap = 'CMRmap', norm = 'log', interpolation='bicubic',
          origin = 'lower', extent = (-np.pi, np.pi, -np.pi, np.pi), 
          vmin = check_every * dt, vmax = tmax)
plt.savefig('MIB0036_PendulumFractal.png', dpi = fig.dpi, bbox_inches='tight', pad_inches = 0)

