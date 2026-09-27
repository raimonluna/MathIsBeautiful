import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import matplotlib.animation as animation

max_side = 0.1
frames   = 300
side     = 20

base_box = np.array([-1+1j, 1+1j, 1-1j, -1-1j])
angles   = np.pi/2 - 0.57 * np.cos(np.linspace(0, 2 * np.pi, frames))
cmap     = mpl.colors.LinearSegmentedColormap.from_list("", [(0.0, 'maroon'), (0.7, 'darkgreen'), (0.8, 'forestgreen'), (1, 'green')])

def make_tree(ax, box, angle, steps):
    ax.fill(box.real, box.imag, color = cmap(256 * steps // 10))
    if np.abs(box[1]- box[0]) > max_side:
        box_L = (box - box[3]) * (np.exp(1j * angle) + 1) / 2 + box[0]
        box_R = (box - box[2]) * (1 - np.exp(1j * angle)) / 2 + box[1]
        make_tree(ax, box_L, angle, steps + 1)
        make_tree(ax, box_R, angle, steps + 1)

print('Making movie, please wait...')
fig, ax = plt.subplots(1,1, figsize = (6, 6))
fig.subplots_adjust(left=0, bottom=0, right=1, top=1, wspace=None, hspace=None)
fig.set_facecolor("tan")
ax.axis('off')
plt.close()

def animate(i):
    print(i)
    ax.cla()
    ax.axis('off')
    ax.set_xlim(-side/2, side/2)
    ax.set_ylim(-side/2, side/2)
    make_tree(ax, +1j + base_box, angles[i], 0)
    make_tree(ax, -1j - base_box, angles[i], 0)
    return fig

animation_fig = animation.FuncAnimation(fig, animate, frames = frames, interval = 50)
animation_fig.save("MIB0068_PithagorasTree.mp4", dpi = 200)
