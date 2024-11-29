import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation

theta_ray = np.linspace(-np.pi, np.pi, 200)

fig, ax = plt.subplots(frameon=False, figsize= (6,6))
fig.subplots_adjust(left=0, bottom=0, right=1, top=1, wspace=None, hspace=None)
ax.set_xlim(-1.1, 1.1)
ax.set_ylim(-1.1, 1.1)
ax.set_facecolor("wheat")
ax.xaxis.set_visible(False)
ax.yaxis.set_visible(False)

ax.fill(1.01*np.cos(theta_ray), 1.01*np.sin(theta_ray), color = 'peru', edgecolor = 'black')

lin, = ax.plot([-1], [0], color = 'blue', alpha = 0.5)
def update(i):
    global lin
    if i >= len(theta_ray):
        return
    theta1 = theta_ray[i]
    theta2 = 2 * theta1 + np.pi
    lin.set_color('yellow')
    lin, = ax.plot([-1, np.cos(theta1), np.cos(theta2)], [0, np.sin(theta1), np.sin(theta2)], color = 'blue', alpha = 0.5)

animation_fig = animation.FuncAnimation(fig, update, frames = 250, interval = 100)
animation_fig.save("MIB0022_MugCardioid.mp4", dpi = 200)

