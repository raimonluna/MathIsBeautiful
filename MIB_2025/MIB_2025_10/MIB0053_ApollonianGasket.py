import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from scipy.linalg import solve
from queue import Queue

class Circle:
    def __init__(self, center, radius):
        self.center = center
        self.radius = radius
        
    def plot(self, ax, color = 'b'):
        theta = np.linspace(0, 2*np.pi, 100)
        x, y   = self.center[0] + self.radius * np.cos(theta), self.center[1] + self.radius * np.sin(theta)
        x1, y1 = self.center[0] + 0.4 * self.radius * np.cos(theta), self.center[1] + 0.6 * self.radius + 0.2 * self.radius * np.sin(theta)
        ax.plot(x, y, color = 'k', lw = 0.75)
        ax.fill(x, y, color = color)
        ax.fill(x1, y1, color = 'w')

def find_tangent(circles):

    centers01 = (circles[0].center / circles[0].radius + circles[1].center / circles[1].radius) / (1 / circles[0].radius + 1 / circles[1].radius)
    unit      = (centers01 - circles[2].center) / np.linalg.norm((centers01 - circles[2].center))
    centers   = (circles[2].center + 1.001 * unit * circles[2].radius)[:, None]
    
    res = 100
    while res > 1e-10:
    
        dists = [np.sqrt( (centers[0,0] - circles[i].center[0])**2 + 
                          (centers[1,0] - circles[i].center[1])**2  ) 
                           for i in range(3)]
    
        if res == 100: signs = [np.sign(dists[i] - circles[i].radius) for i in range(3)]
        
        rhs = np.array([signs[0] * (dists[0] - circles[0].radius) -
                        signs[i] * (dists[i] - circles[i].radius) 
                        for i in range(1, 3)])[:, None]
    
        J = np.array([[signs[0] * (centers[j,0] - circles[0].center[j]) / dists[0] -
                       signs[i] * (centers[j,0] - circles[i].center[j]) / dists[i] 
                       for j in range(2)] for i in range(1, 3)])
        
        dcenters = solve(J, rhs)
        centers -= dcenters    
        res = np.sum(dcenters**2)
  
    return Circle(centers[:, 0], np.abs(dists[0] - circles[0].radius))

R3   =  2 * np.sqrt(3) - 3
phi3 = (2 * np.arange(3) - 1 / 2) * np.pi / 3
circles = np.array([Circle(np.array([0, 0]), 1)] + [Circle((1 - R3) * np.array([np.cos(p), np.sin(p)]), R3) for p in phi3])

#########################################################################3

fig, ax = plt.subplots(1, 1, figsize = (10, 10) )
fig.patch.set_facecolor('darkgreen')
ax.set_xlim(-1.05,1.05)
ax.set_ylim(-1.05,1.05)
ax.axis('off')

cmap = mpl.colormaps['pink']
circles[0].plot(ax, color = cmap(0))
[circles[i].plot(ax, color = cmap(230)) for i in range(1, 4)]

to_fill = Queue()
for indices in [[0,1,2], [0,2,3], [0,1,3], [1,2,3]]:
    to_fill.put(tuple(circles[indices]))

while not to_fill.empty():
    old = to_fill.get()
    new = find_tangent(old)
    new.plot(ax, color = cmap( int( 255 * (new.radius / R3)**0.3)))
    if new.radius > 0.005:
        to_fill.put((old[0], old[1], new))
        to_fill.put((old[1], old[2], new))
        to_fill.put((old[2], old[0], new))

plt.savefig('MIB0053_ApollonianGasket.png', dpi = 1000, bbox_inches='tight', pad_inches = 0)

