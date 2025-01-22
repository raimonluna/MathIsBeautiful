import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
np.random.seed(42)

# Parameters of the grid
shape        = (2000, 2000)
xrange       = (-4, 4)
yrange       = (-3, 5)

# Parameters of the airfoil
mu    = -0.15 + 0.15j
cut_h = 0.07
V_inf = 3
alpha = 20 * (np.pi / 180)
R     = np.sqrt((1 - mu.real)**2 + mu.imag**2)
gamma = 4 * np.pi * V_inf * R * np.sin( alpha + np.arcsin(mu.imag / R) )
rot   = np.exp(1j * alpha)

# Inverse Joukowsky transform
def z2zeta(z):
    discriminant = np.sqrt(z**2 - 4)
    zeta = (z + discriminant) / 2
    in_unit_circle = (np.abs(zeta) <= 1)
    below_wing = (z.real**2 < 4) & (z.imag > 0) & \
                 (z.imag < cut_h * (4 - z.real**2))
    bad_branch = below_wing ^ in_unit_circle
    zeta[bad_branch] -= discriminant[bad_branch]
    return zeta

# W = vx - i vy
def complex_velocity(z):
    zeta     = z2zeta(z)
    W_tilde  = V_inf * np.exp(-1j * alpha) 
    W_tilde += 1j * gamma / (2 * np.pi * (zeta - mu))
    W_tilde -= V_inf * R**2 * np.exp(1j * alpha) / (zeta - mu)**2
    W = W_tilde / (1 - 1/zeta**2)
    return W.conj()

# Bernoullis's principle
def pressure(z):
    st = complex_velocity(z)
    P = 0.5 * (V_inf**2 - np.abs(st)**2)
    return P

# Fix misplaced points and remove the ones inside the wing
def purge_points(xy):
    xy[xy.real >  xrange[1]] -= (xrange[1] - xrange[0])
    xy[xy.imag < -yrange[0]] += 1j * (yrange[1] - yrange[0])
    xy[xy.imag >  yrange[1]] -= 1j * (yrange[1] - yrange[0])
    in_wing = np.abs(z2zeta(xy * rot) - mu) < R
    xy[in_wing] += 100 + 100j

# Define the complex plane grid
rez  = np.linspace(*xrange, shape[0])[:, None]
imz  = np.linspace(*yrange, shape[1])[None, :]
z    = (rez + 1j * imz)

# Random particles to show flow
x  = xrange[0] + (xrange[1] - xrange[0]) * np.random.rand(5000)
y  = yrange[0] + (yrange[1] - yrange[0]) * np.random.rand(5000)
xy = x + 1j * y 

# Joukowsky airfoil profile
theta     = np.linspace(0, 2*np.pi, 100)
zeta_circ = mu + R * np.exp(1j * theta)
z_airfoil = (zeta_circ + 1/zeta_circ) * rot.conj()

##### Do the plotting #####

fig, ax = plt.subplots(figsize = (8,8))
fig.subplots_adjust(left=0, bottom=0, right=1, top=1, wspace=None, hspace=None)
plt.close()
ax.axis('off')

ax.imshow(pressure(z * rot).T, origin = 'lower', cmap = 'jet_r', 
          extent = xrange + yrange, vmin = -10, vmax = 7)
dots = ax.scatter(x, y, color = 'black', s= .3)
ax.fill(z_airfoil.real, z_airfoil.imag, color = 'white')

def step(i):
    global xy
    for _ in range(10):
        xy += 0.001 * complex_velocity(xy * rot ) * rot.conj()
    purge_points(xy)
    dots.set_offsets(np.c_[xy.real, xy.imag])

animation_fig = animation.FuncAnimation(fig, step, frames = 200, interval = 50)
animation_fig.save("MIB0027_JoukowskyAirfoil.mp4", dpi = 200)
