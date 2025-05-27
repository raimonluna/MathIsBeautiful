import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.animation as animation

n_pendula = 1000
dt        = 0.05

dtheta  = np.linspace(-1e-5, 1e-5, n_pendula)[:, None]
X       = np.hstack([3 + dtheta, -1 + dtheta, 0*dtheta, 0*dtheta])

def rhs(t, X):
    t1, t2, p1, p2 = X.transpose(1,0)

    cosdt = np.cos(t1 - t2)
    sindt = np.sin(t1 - t2)
    sint1 = np.sin(t1)
    sint2 = np.sin(t2)

    dt1 = 6 * ( 2 * p1 - 3 * cosdt * p2 ) / ( 16 - 9 * cosdt**2 )
    dt2 = 6 * ( 8 * p2 - 3 * cosdt * p1 ) / ( 16 - 9 * cosdt**2 )
    dp1 = - 0.5 * ( 3 * sint1 + dt1 * dt2 * sindt )
    dp2 = - 0.5 * ( 1 * sint2 - dt1 * dt2 * sindt )

    return np.hstack([dt1[:, None], dt2[:, None], dp1[:, None], dp2[:, None]])
    
def RK4(t, X):
    k1 = rhs(t, X)
    k2 = rhs(t + 0.5 * dt, X + 0.5 * k1 * dt)
    k3 = rhs(t + 0.5 * dt, X + 0.5 * k2 * dt)
    k4 = rhs(t + dt, X + k3 * dt)
    return t + dt, X + dt * (k1 + 2*k2 + 2*k3 + k4) / 6

def plot_pendula(ax, X, ls = None):
    x_to_plot  = np.vstack([0 * X[:, 0],   np.sin(X[:, 0]),   np.sin(X[:, 0]) + np.sin(X[:, 1])])[None, ...]
    y_to_plot  = np.vstack([0 * X[:, 0], - np.cos(X[:, 0]), - np.cos(X[:, 0]) - np.cos(X[:, 1])])[None, ...]
    xy_to_plot = np.vstack([x_to_plot, y_to_plot])
    if ls is None:
        cmap = mpl.colormaps['jet_r']
        ls = [ax.plot(*xy_to_plot[..., i], '-o', color = cmap(i/n_pendula), alpha = 0.5,
                      markerfacecolor = 'k', markeredgecolor = 'k')[0] for i in range(n_pendula)]
        return ls
    return [ln.set_data(*xy_to_plot[..., i]) for i, ln in enumerate(ls)]

fig, ax = plt.subplots(1,1, figsize = (6, 6))
fig.subplots_adjust(left=0, bottom=0, right=1, top=1, wspace=None, hspace=None)
fig.patch.set_facecolor('wheat')
plt.close()

ls = plot_pendula(ax, X)
ax.set_xlim(-2.1, 2.1)
ax.set_ylim(-2.1, 2.1)
ax.axis('off')

def update(i):
    global X
    _, X = RK4(0, X)
    plot_pendula(ax, X, ls)

# Iteration i = 286 for the thumbnail, with ranges (-1.2, 1.8) and (-2.5, 0.5).
animation_fig = animation.FuncAnimation(fig, update, frames = 500, interval = 50)
animation_fig.save("MIB0037_DoublePendulum.mp4", dpi = 200)

