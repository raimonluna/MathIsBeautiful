import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import matplotlib as mpl
np.random.seed(42)

max_dist = 15
rep_dist = 5 * max_dist
att_dist = max_dist / 2
gravity  = 20
dt       = 1

epsilon   = 1e-10
init_part = 10
radius    = 40

iters      = 700
plot_every = 5
frames     = iters // plot_every

cmap = mpl.colormaps['turbo']

theta   = np.linspace(0, 2 * np.pi, init_part + 1)[:-1]
points  = np.vstack([radius * np.cos(theta), radius * np.sin(theta), 0 * theta, 0 * theta]).T 
points += 10 * (1 - 2 * np.random.rand(init_part, 4))

def insert_points(points, max_dist):
    dists = np.sum((points - np.roll(points, 1, axis = 0))**2, axis = 1)**0.5
    if (dists > max_dist).any():   
        chunks  = np.split(points, np.where(dists > max_dist)[0])[int(dists[0] > max_dist):]
        new_points = chunks[0]
        for chunk in chunks[1:]: # Any cool way to vectorize this?
            new_points = np.vstack([new_points, (new_points[[-1],:] + chunk[[0],:]) / 2, chunk])
        if dists[0] > max_dist:
            new_points = np.vstack([new_points, (new_points[[-1],:] + new_points[[0],:]) / 2])
        return new_points
    return points

def compute_forces(points):
    ind        = np.arange(points.shape[0])
    ind1, ind2 = ind[:, None], ind[None, :]
    
    vecs  = points[None, :, :2] - points[:, None, :2]
    dists = np.sum(vecs**2, axis = 2)**0.5
    dists[ind, ind] = 1 / epsilon
    
    neighbors  = np.abs(ind1 - ind2) % (ind[-1] - 1) == 1
    coef_rep   = - 1.0 * (dists < rep_dist)
    coef_neigh = neighbors * (1.0 - 2 * (dists < att_dist))
    
    forces = np.sum(gravity * vecs * ((coef_rep + coef_neigh) / dists**3)[..., None], axis = 1)
    return forces

fig, ax = plt.subplots(1, 1, figsize = (10, 10) )
fig.patch.set_facecolor('black')
fig.subplots_adjust(left=0, bottom=0, right=1, top=1, wspace=None, hspace=None)
ax.set_xlim(-350, 350)
ax.set_ylim(-350, 350)
ax.axis('off')
plt.close()

def update(i):
    global points
    if i < frames:
        for k in range(plot_every):
            vel = points[:, 2:]
            acc = compute_forces(points)
            points[:, 2:] += acc * dt
            points[:, :2] += vel * dt
            points[:, 2:] *= 0.7 # damping
            points = insert_points(points, max_dist)
    
        closed = np.vstack([points, points[[0], :]])
        ax.plot(*closed[:, :2].T, lw = 1, alpha = i/frames, color = cmap(i/frames))

animation_fig = animation.FuncAnimation(fig, update, frames = frames + 50, interval = 50)
animation_fig.save("MIB0055_DifferentialGrowth.mp4", dpi = 200)
