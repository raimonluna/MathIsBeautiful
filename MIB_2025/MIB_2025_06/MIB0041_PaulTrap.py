import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.animation as animation

shape  = (1000, 1000)
xrange = (-1, 1)
yrange = (-1, 1)
n_ions = 100
sigma  = 0.1
w0     = 4.0

frames       = 1000
dt           = 0.005
it_per_frame = 10 
tmax         = frames * it_per_frame * dt 
np.random.seed(42)
print(f'tmax = {tmax:.2f}')

x = np.linspace(*xrange, shape[0])[:, None]
y = np.linspace(*yrange, shape[1])[None, :]
z = x + 1j * y 

Z = np.random.normal(0, sigma, (n_ions, 4))
Z = np.hstack([Z[:, [0]] + 1j * Z[:, [1]], Z[:, [2]] + 1j * Z[:, [3]]])

def phase(t):
    return w0 * t * (1 - t / (2 * tmax))

def rhs(t, Z):
    return np.hstack([Z[:, [1]], - Z[:, [0]].conj() * np.exp(2j * phase(t)) ])
    
def RK4(t, Z):
    k1 = rhs(t, Z)
    k2 = rhs(t + 0.5 * dt, Z + 0.5 * k1 * dt)
    k3 = rhs(t + 0.5 * dt, Z + 0.5 * k2 * dt)
    k4 = rhs(t + dt, Z + k3 * dt)
    return t + dt, Z + dt * (k1 + 2*k2 + 2*k3 + k4) / 6

##### Do the plotting
fig, ax = plt.subplots(frameon = False, figsize = (5, 5))
fig.subplots_adjust(left=0, bottom=0, right=1, top=1, wspace=None, hspace=None)
ax.axis('off')
ax.set_xlim(-1,1)
ax.set_ylim(-1,1)
plt.close()

potential  = ax.imshow(z.real.T, cmap = 'PuOr', origin = 'lower', extent = xrange + yrange, vmin = -0.3, vmax = 0.3)
particles, = ax.plot(Z[:, 1].real, Z[:, 1].imag, 'o', color = 'blue', ms = 2.0)

t = 0
def update(i):
    global t, Z
    for _ in range(it_per_frame):
        t, Z = RK4(t, Z)
    V = 0.5 * z.conj()**2 * np.exp(2j * phase(t))
    potential.set_data(V.real.T)
    particles.set_data(Z[:, 1].real, Z[:, 1].imag)
    
animation_fig = animation.FuncAnimation(fig, update, frames = frames, interval = 50)
animation_fig.save("MIB0041_PaulTrap.mp4", dpi = 200)
