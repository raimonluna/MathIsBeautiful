import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import matplotlib as mpl

number_of_disks  = 6

moves      = 2*[(1, 0, number_of_disks - 1 )]
occupation = np.array([number_of_disks, 0, 0])

def MakeMove(n, start, end):
    moves.append((n, end, occupation[end]))
    occupation[[start, end]] += -1, +1

def SolveTowers(n, start, end, pivot):
    if n == 1: return MakeMove(n, start, end)
    SolveTowers(n - 1, start, pivot, end)
    MakeMove(n, start, end)
    SolveTowers(n - 1, pivot, end, start)

SolveTowers(number_of_disks, 0, 2, 1) 
SolveTowers(number_of_disks, 2, 0, 1) 

##### Do the plotting

fig, ax = plt.subplots(figsize = (5, 5))
ax.set_xlim(-0.6, 2.7)
ax.set_ylim(-0.2, 0.6)
plt.axis('off')
plt.close()
fig.patch.set_facecolor('olivedrab')

cmap  = mpl.colormaps['pink']
disks = [plt.Rectangle((-0.5 + 0.075 * n, 0.05 * n), 1 - 0.15 * n, .05, color = cmap(40*n)) for n in range(number_of_disks)]

for i in range(3): ax.add_patch( plt.Rectangle((-0.05 + i, 0), .1, .35, color = "saddlebrown") )
ax.add_patch( plt.Rectangle((-0.6, -0.05), 3.2, .05, color = "saddlebrown"))
for disk in disks: ax.add_patch(disk)

def place_disk(i):
    n, peg, pos = moves[i]
    n = number_of_disks - n
    disks[n].set_xy((-0.5 + 0.075 * n + peg, 0.05 * pos))

animation_fig = animation.FuncAnimation(fig, place_disk, frames = len(moves), interval = 400)
animation_fig.save("MIB0030_TowerOfHanoi.mp4", dpi = 200)
