import numpy as np
import random
import matplotlib.pyplot as plt

shape = (200, 200)
random.seed(11)

distance   = np.zeros(shape, dtype = np.int64)

##### Maze creation functions

def acceptable(c):
    i, j = c
    return  (0 <= i < shape[0]) and (0 <= j < shape[0]) and (distance[i,j] == 0)

def get_edges(p):
    candidates = [(p[0] + di, p[1] + dj) for di in (-1,0,1) for dj in (-1,0,1) if di*dj == 0]
    candidates = [c for c in candidates if acceptable(c)]
    return candidates

##### Prim's Algorithm

p = (100, 100)
distance[*p] = 1
edges = []
for e in get_edges(p):
    edges.append((p, e))

while len(edges) > 0:
    p, e = random.choice(edges)
    edges.remove((p, e))
    if distance[*e] == 0:
        distance[*e] = distance[*p] + 1
        for f in get_edges(e):
            edges.append((e, f))
    
##### Do the plotting

fig, ax = plt.subplots(1,1, figsize = (10, 10))
ax.axis('off')
ax.imshow(distance.T, cmap = 'gnuplot2_r', interpolation = 'none', origin = 'lower', vmin = 20, vmax = 140)
plt.savefig('MIB0039_PrimDistance.png', dpi = fig.dpi, bbox_inches='tight', pad_inches = 0)
