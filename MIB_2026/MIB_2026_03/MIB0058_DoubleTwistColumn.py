import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.animation as animation

num_vertices = 8
amplitude    = 0.3
num_iters    = 6

num_points = num_vertices * 2**(num_iters + 2)
closed     = np.arange(num_points + 1) % num_points

theta = np.linspace(0, 2 * np.pi, num_points + 1)[:-1]
dt    = theta[1] - theta[0]
rot   = np.array([np.exp(1j * dt), np.exp( - 1j * dt)])

points = (1 + amplitude * np.cos(num_vertices * theta)) * np.exp(1j * theta)
points = points[:, None] @ np.ones((1,2))

##### Generate all the curves for the video
store_points = points[closed, [0]][None, :]
store_inters = points[closed, [0]][None, :]
store_colors = np.ones((1, 3))

for it in range(num_iters, -1, -1):
    
    for i in range(2**it):
        points      *= rot
        points[:, 0] = np.roll(points[:, 0], 2)
        intersect    = points[np.arange(num_points), np.argmin(np.abs(points), axis = 1)]
        
        store_inters = np.vstack([store_inters, intersect[closed]])
        store_points = np.vstack([store_points, points[closed, 0]])
        store_colors = np.vstack([store_colors, mpl.colors.hsv_to_rgb([(it / num_iters)**2, 1, 1 - i / 2**it])[None, :]])
            
    points = intersect[:, None] @ np.ones((1,2))

##### Plotting
fig, ax = plt.subplots(1,1, figsize = (6, 6))
fig.subplots_adjust(left=0, bottom=0, right=1, top=1, wspace=None, hspace=None)

ax.set_xlim(-1.4, 1.4)
ax.set_ylim(-1.4, 1.4)
fig.patch.set_facecolor('wheat')
ax.axis('off')
plt.close()

ln1, = ax.plot(store_points[0, :].real, +store_points[0, :].imag, 'indigo', zorder = store_colors.shape[0] + 1)
ln2, = ax.plot(store_points[0, :].real, -store_points[0, :].imag, 'indigo', zorder = store_colors.shape[0] + 1)
ln3, = ax.plot(store_points[0, :].real, +store_points[0, :].imag, 'w', zorder = store_colors.shape[0] + 1)

def update(ii): 
    i = min(store_colors.shape[0] - 1, ii)
    ax.fill(store_inters[i, :].real, store_inters[i, :].imag, color = 'indigo', zorder = i)
    ax.plot(store_inters[i, :].real, store_inters[i, :].imag, color = store_colors[i, :], zorder = i + 1)
    ln1.set_data(store_points[i, :].real, +store_points[i, :].imag)
    ln2.set_data(store_points[i, :].real, -store_points[i, :].imag)
    ln3.set_data(store_inters[i, :].real, store_inters[i, :].imag)
    
animation_fig = animation.FuncAnimation(fig, update, frames = store_colors.shape[0] + 10, interval = 100)
animation_fig.save("MIB0058_DoubleTwistColumn.mp4", dpi = 200)
