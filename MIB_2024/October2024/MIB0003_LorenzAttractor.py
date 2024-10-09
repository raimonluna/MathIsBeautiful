import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import matplotlib.animation as animation

s, b, r     = 10, 8/3, 28
total_iters = 50000
frames      = 200
dt          = 0.001

fig  = plt.figure()
fig.subplots_adjust(left=0, bottom=0, right=1, top=1, wspace=None, hspace=None)
ax   = fig.add_subplot(projection='3d')
cmap = mpl.colormaps['ocean']

np.random.seed(42)
for run in range(5):
    X = np.asarray([1, 1, 1 + 0.001*np.random.rand()])
    saved_sols = np.zeros((total_iters, 3))
    for i in range(total_iters):
        saved_sols[i, :] = X
        x, y, z = X
        X += np.asarray([s*(y-x), x*(r-z) - y, x*y - b*z]) * dt
    im = ax.plot(*saved_sols.T, linewidth = 0.3, color = cmap(256 * run // 5))

ax.axis('off')
ax.set_xlim(-20, 20)
ax.set_ylim(-20, 25)
ax.set_zlim(15, 42)
ax.set_facecolor("black")
fig.set_facecolor("black")
animation_fig = animation.FuncAnimation(fig, lambda i: ax.view_init(elev = 20, azim = - 360 * i / frames), frames = frames, interval = 50)
animation_fig.save("MIB0003_LorenzAttractor.mp4", dpi = 200)
