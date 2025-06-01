import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import random

shape = (40, 40)
random.seed(0)

distance   = np.zeros(shape, dtype = np.int64)
segments   = np.zeros(shape + (2,), dtype = np.int64)
horizontal = np.ones(shape, dtype = bool)
vertical   = np.ones(shape, dtype = bool)

##### Maze creation functions

def acceptable(c):
    i, j = c
    return  (0 <= i < shape[0]) and (0 <= j < shape[0]) and (distance[i,j] == 0)

def get_edges(p):
    candidates = [(p[0] + di, p[1] + dj) for di in (-1,0,1) for dj in (-1,0,1) if di*dj == 0]
    candidates = [c for c in candidates if acceptable(c)]
    return candidates

def remove_wall(p, e):
    if p[0] == e[0]: # Vertical connection, horizontal wall removal
        horizontal[p[0], min(p[1], e[1])] = False
    else:            # Horizontal connection, vertical wall removal
        vertical[min(p[0], e[0]), p[1]] = False

def add_vertex(p, e):
    distance[*e] = distance[*p] + 1
    segments[*e, :] = p
    remove_wall(p, e)

def draw_walls(ax):
    ax.plot([-0.5, shape[0] -0.5], [-0.5, -0.5], 'k')
    ax.plot([-0.5, -0.5], [-0.5, shape[1] -0.5], 'k')
    [ax.plot([i - 0.5, i + 0.5], [j + 0.5, j + 0.5], 'k') for i in range(shape[0]) for j in range(shape[1]) if horizontal[i,j]]
    [ax.plot([i + 0.5, i + 0.5], [j - 0.5, j + 0.5], 'k') for i in range(shape[0]) for j in range(shape[1]) if vertical[i,j]]

##### Create the maze: Prim's Algorithm

p = (0, 0)
distance[*p] = 1
edges = []
for e in get_edges(p):
    edges.append((p, e))

while len(edges) > 0:
    p, e = random.choice(edges)
    edges.remove((p, e))
    if distance[*e] == 0:
        add_vertex(p, e)
        for f in get_edges(e):
            edges.append((e, f))

##### Solve the maze

path = [np.asarray(shape) - 1]
while np.sum(path[-1]) > 0:
    path.append(segments[*path[-1]])
path = np.asarray(path)[::-1,:].T
    
##### Do the plotting

fig, ax = plt.subplots(1,1, figsize = (6, 6))
fig.subplots_adjust(left=0, bottom=0, right=1, top=1, wspace=None, hspace=None)
plt.close()

ax.axis('off')
ax.imshow(distance.T, cmap = 'cool', origin = 'lower')
draw_walls(ax)

ln, = ax.plot(*path[:, :1], 'r', lw = 3)

def update(i):
    ii = min(path.shape[1], i)
    ln.set_data(*path[:, :ii])

animation_fig = animation.FuncAnimation(fig, update, frames = path.shape[1] + 10, interval = 100)
animation_fig.save("MIB0038_PrimAlgorithm.mp4", dpi = 200)
