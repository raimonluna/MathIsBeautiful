import numpy as np
from scipy.linalg import solve
import matplotlib.pyplot as plt
import matplotlib.animation as animation

# Jansen's proportions 
a, b, c, d, e, f, g, h, i, j, k, l, m  = 38.0, 41.5, 39.3, 40.1, 55.8, 39.4, 36.7, 65.7, 49.0, 50.0, 61.9, 7.8, 15.0
link_list = [(0, 2, b), (0, 3, d), (0, 5, c), (1, 2, j), (1, 5, k), (2, 3, e), (3, 4, f), (4, 5, g), (4, 6, h), (5, 6, i)] 
x_vector  = np.array([[0, a + m, 20, -40, -20,  30,   0]]).T
y_vector  = np.array([[0, l + m, 40,  0, -40, -40, -100]]).T

angle_number = 150
speed        = 22.5 # In distance per radian
angles       = np.linspace(2 * np.pi, 0, angle_number + 1)[:-1]
x_saved      = np.zeros((angle_number, 7))
y_saved      = np.zeros((angle_number, 7))

for row, angle in enumerate(angles):
    x_vector[1] = a + m * np.cos(angle)
    y_vector[1] = l + m * np.sin(angle)
    
    bb = 1
    while np.sum(bb**2) > 1e-10:
        AA, bb = np.zeros((10, 10)), np.zeros((10,1))
        
        for equation, (origin, destiny, length) in enumerate(link_list):
            bb[equation, :] = (x_vector[destiny, 0] - x_vector[origin, 0])**2 + (y_vector[destiny, 0] - y_vector[origin, 0])**2 - length**2
    
            if destiny > 1:
                AA[equation, destiny - 2] = 2 * (x_vector[destiny, 0] - x_vector[origin, 0])
                AA[equation, destiny + 3] = 2 * (y_vector[destiny, 0] - y_vector[origin, 0])
            
            if origin > 1:
                AA[equation, origin - 2] = - 2 * (x_vector[destiny, 0] - x_vector[origin, 0])
                AA[equation, origin + 3] = - 2 * (y_vector[destiny, 0] - y_vector[origin, 0])
    
        correction    = solve(AA, bb)
        x_vector[2:] -= correction[:5]
        y_vector[2:] -= correction[5:]

    x_saved[row, :] = x_vector[:, 0]
    y_saved[row, :] = y_vector[:, 0]

print('Making movie, please wait...')
fig, ax = plt.subplots(1,1, figsize = (6, 6))
fig.subplots_adjust(left=0, bottom=0, right=1, top=1, wspace=None, hspace=None)
fig.set_facecolor("wheat")
ax.axis('off')
plt.close()

ax.set_xlim(a - 120, a + 120)
ax.set_ylim(l - 140, l + 100)
ax.grid()
ax.plot(        x_saved, y_saved, ':', color = 'brown')
ax.plot(2 * a - x_saved, y_saved, ':', color = 'brown')

row = 0
linesL, linesR = [], []

for equation, (origin, destiny, length) in enumerate(link_list):
    linesL.append(ax.plot(        x_saved[row, [origin, destiny]], y_saved[row, [origin, destiny]], 'k-', lw = 5)[0])
    linesR.append(ax.plot(2 * a - x_saved[row, [origin, destiny]], y_saved[row, [origin, destiny]], 'k-', lw = 5)[0])

pointsLk = ax.plot(        x_saved[row, :],                   y_saved[row, :], 'ko', ms = 15)[0]
pointsRk = ax.plot(2 * a - x_saved[angle_number//2 - row, :], y_saved[angle_number//2 - row, :], 'ko', ms = 15)[0]
pointsLw = ax.plot(        x_saved[row, :],                   y_saved[row, :], 'wo', ms = 5)[0]
pointsRw = ax.plot(2 * a - x_saved[angle_number//2 - row, :], y_saved[angle_number//2 - row, :], 'wo', ms = 5)[0]
floor    = ax.plot(np.linspace(a - 260, a + 120, 50), 50 * [-90], '^', color = 'brown', lw = 5)[0]

def update(row):
    row_mod = row % angle_number
    row_inv = angle_number//2 - row_mod
    
    for equation, (origin, destiny, length) in enumerate(link_list):
        linesL[equation].set_data(        x_saved[row_mod, [origin, destiny]], y_saved[row_mod, [origin, destiny]])
        linesR[equation].set_data(2 * a - x_saved[row_inv, [origin, destiny]], y_saved[row_inv, [origin, destiny]])
    
    pointsLk.set_data(        x_saved[row_mod, :], y_saved[row_mod, :])
    pointsRk.set_data(2 * a - x_saved[row_inv, :], y_saved[row_inv, :])
    pointsLw.set_data(        x_saved[row_mod, :], y_saved[row_mod, :])
    pointsRw.set_data(2 * a - x_saved[row_inv, :], y_saved[row_inv, :])
    floor.set_xdata(np.linspace(a - 260, a + 120, 50) + speed * angles[row_mod])
    
animation_fig = animation.FuncAnimation(fig, update, frames = 3 * angle_number, interval = 20)
animation_fig.save("MIB0067_Strandbeest.mp4", dpi = 200)
