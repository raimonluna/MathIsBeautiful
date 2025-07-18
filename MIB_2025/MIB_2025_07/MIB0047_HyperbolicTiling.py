import numpy as np
from tqdm.auto import tqdm
import matplotlib.pyplot as plt
import matplotlib as mpl

iterations   = 20
tiling_type  = (5, 4)
color_scheme = 'YlGnBu'

##### Important definitions

class PoincareGeodesic:
    def __init__(self, P, Q):
        self.P, self.Q = P, Q
        if np.isclose(P[0] * Q[1], P[1] * Q[0]):
            self.type = 'straight'
        else:
            self.type   = 'arc'
            d = 2 * (P[0] * Q[1] - P[1] * Q[0])
            p = 1 + P[0]**2
            q = 1 + Q[0]**2 + Q[1]**2
            self.center = (( +p * Q[1] + P[1]**2 * Q[1] - P[1] * q) / d ,
                           ( -p * Q[0] - P[1]**2 * Q[0] + P[0] * q) / d )
            self.radius = np.sqrt((self.center[0] - P[0])**2 + (self.center[1] - P[1])**2)
            angleP      = np.arctan2(P[1] - self.center[1], P[0] - self.center[0]) 
            angleQ      = np.arctan2(Q[1] - self.center[1], Q[0] - self.center[0])

            if np.abs(angleP - angleQ) > np.pi:
                self.angles = ((angleP + 2 * np.pi) % (2 * np.pi),
                               (angleQ + 2 * np.pi) % (2 * np.pi))
            else:
                self.angles = (angleP, angleQ)
            
    def invert(self, R):
        if self.type == 'straight':
            u, v = self.P[0] - self.Q[0], self.Q[1] - self.P[1]
            return ((- R[0] * (v**2 - u**2) - 2 * u * v * R[1]) / (u**2 + v**2),
                    (+ R[1] * (v**2 - u**2) - 2 * u * v * R[0]) / (u**2 + v**2))
        elif self.type == 'arc':
            r2d2 = self.radius**2 / ((R[0] - self.center[0])**2 + (R[1] - self.center[1])**2)
            return (self.center[0] + r2d2 * (R[0] - self.center[0]),
                    self.center[1] + r2d2 * (R[1] - self.center[1]))
        
    def curve(self):
        if self.type == 'straight':
            x_data, y_data = [self.P[0], self.Q[0]], [self.P[1], self.Q[1]]
        elif self.type == 'arc':
            theta = np.linspace(*self.angles, 101)
            x_data, y_data = self.center[0] + self.radius * np.cos(theta), \
                             self.center[1] + self.radius * np.sin(theta)
        return x_data, y_data
        
    def plot(self, ax):
        ax.plot(*self.curve())

class PoincarePolygon:
    def __init__(self, vertices, color = 'gray', family = 0):
        self.vertices  = vertices
        self.middle    = np.mean(np.sum(np.array(vertices) * [1, 1j], axis = 1))
        self.color     = color
        self.family    = family
        self.num_faces = len(vertices)
        self.edges     = [PoincareGeodesic(vertices[i - 1], vertices[i]) for i in range(self.num_faces)]  

    def neighbors(self, color):
        return [PoincarePolygon([edge.invert(vertex) for vertex in self.vertices], color, family = (self.family + 1) % 2) for edge in self.edges]
    
    def plot(self, ax):
        bla = np.hstack([edge.curve() for edge in self.edges])
        ax.fill(*bla, color = self.color, edgecolor ='k')

def central_polygon(p, q):
    cotcot = 1 / (np.tan(np.pi / p) * np.tan(np.pi / q))
    radius = (cotcot - 1) / np.sqrt(cotcot**2 - 1)
    theta  = np.pi * np.linspace(1 / p, 2 - 1/p, p)
    return radius * np.array([np.cos(theta), np.sin(theta)]).T

def central_star(p, q, colors = ('r', 'b')):
    cotcot = 1 / (np.tan(np.pi / p) * np.tan(np.pi / q))
    radius = (cotcot - 1) / np.sqrt(cotcot**2 - 1)
    P      = radius * np.exp(1j * np.pi / p)
    x, y   = PoincareGeodesic((P.real, P.imag), (P.real, -P.imag)).curve()
    Pc, R  = P.conj(), x[len(x)//2] + 1j * y[len(y)//2]
    
    central = []
    for side in range(p):
        P, Pc, R = np.exp(2j * np.pi / p) * np.array([P, Pc, R])
        central.append(PoincarePolygon([(0, 0), (R.real, R.imag), (P.real,   P.imag)], colors[0], family = 0))
        central.append(PoincarePolygon([(0, 0), (R.real, R.imag), (Pc.real, Pc.imag)], colors[1], family = 1))
    return central
    
##### Generate Tiling

colormaps = mpl.colormaps[color_scheme], mpl.colormaps[color_scheme + '_r']
to_expand = central_star(*tiling_type, (colormaps[0](0.0), colormaps[1](0.0)))
storage   = to_expand.copy()
middles   = [poly.middle for poly in to_expand]

print(f'Generating polygons, please wait...')
for it in tqdm(range(1, iterations + 1)):    
    expansion = []
    for poly in to_expand:
        neighbors = poly.neighbors(color = colormaps[(poly.family + 1) % 2](it / 10))
        neighbors = [poly for poly in neighbors if np.min(np.abs(poly.middle - middles)) > 1e-5]
        middles   += [poly.middle for poly in neighbors]
        expansion += neighbors
    to_expand = expansion.copy()
    storage  += expansion

print(f'Generated {len(storage)} polygons!')

##### Plot the Tiling

fig, ax = plt.subplots(1, 1, figsize = (10, 10) )
fig.patch.set_facecolor('black')
ax.axis('off')
ax.set_xlim(-1.05,1.05)
ax.set_ylim(-1.05,1.05)

print(f'Plotting polygons, please wait...')
for poly in tqdm(storage):
    poly.plot(ax)

plt.savefig('MIB0047_HyperbolicTiling.png', dpi = 1000, bbox_inches='tight', pad_inches = 0)

    
