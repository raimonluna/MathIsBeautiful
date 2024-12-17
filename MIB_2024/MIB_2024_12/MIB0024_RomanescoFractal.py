import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from scipy.spatial.transform import Rotation as R

def iteration(p):
    
    M11 = R.from_rotvec( -np.pi / 50 * np.array([1, 0, 0])).as_matrix()
    M12 = R.from_rotvec( +np.pi / 13 * np.array([0, 0, 1])).as_matrix()
    b1  = np.array([[0], [0], [1/4]])
    M1  = 0.8 * M11 @ M12
    p1  = np.matmul(M1, p) + b1

    p2 = np.array([[],[],[]])
    for i in range(13):
        angle = 2 * np.pi * i / 13
        M21 = R.from_rotvec( angle * np.array([0, 0, 1])).as_matrix()
        M22 = R.from_rotvec( np.pi/3 * np.array([0, 1, 0])).as_matrix()
        b2  = np.array([[np.cos(angle)], [np.sin(angle)], [0]])
        M2  = 0.22 * M21 @ M22
        p2  = np.hstack([p2, M2 @ p[:, ::13] + b2])

    newp = np.hstack([p1, p2])[:, ::2]
    return newp

points = 100000
iters  = 15
frames = 100

np.random.seed(42)
p = (np.random.rand(3, points) - np.array([[0.5], [0.5], [0]])) * np.array([[2], [2], [1]])
for _ in range(iters):
    p = iteration(p)

###### Making the movie

print('Now making the movie. Please wait...')

fig  = plt.figure()
fig.subplots_adjust(left=0, bottom=0, right=1, top=1, wspace=None, hspace=None)
ax   = fig.add_subplot(projection='3d')

ax.scatter(*p, '.', color = 'green', s = .1)

ax.axis('off')
ax.set_xlim(-.8, .8)
ax.set_ylim(-.8, .8)
ax.set_zlim(-0.2, .8)
ax.set_facecolor("black")
fig.set_facecolor("black")

scene1 = np.hstack([30 * np.ones((3*frames, 1)), 45 + np.linspace(0, 360, 3*frames)[:, None] ])
scene2 = np.hstack([30 + np.linspace(0, 60, frames)[:, None], 45 * np.ones((frames, 1))  ])

all_scenes = np.vstack([ scene1, scene2, scene2[::-1] ] )

animation_fig = animation.FuncAnimation(fig, lambda i: ax.view_init(*all_scenes[i]), frames = len(all_scenes), interval = 50)
animation_fig.save("MIB0024_RomanescoFractal.mp4", dpi = 200)

