import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.animation as animation

total_frames     = 1000
frames_per_box   = 30
points_in_circle = 50

total_boxes = total_frames // frames_per_box
square      = np.array([0, 1, 1+1j, 1j, 0])
circle      = 1j + np.exp(1j * np.linspace(- np.pi/ 2, 0, points_in_circle))
phi         = (1 + np.sqrt(5)) / 2
fibonacci   = (1, 1) 

##### Plotting
fig, ax = plt.subplots(1,1, figsize = (6, 6))
fig.subplots_adjust(left=0, bottom=0, right=1, top=1, wspace=None, hspace=None)

fig.patch.set_facecolor('wheat')
ax.axis('off')
plt.close()

ax.fill(square.real, square.imag, facecolor = 'sandybrown', alpha = 0.8)
ax.plot(square.real, square.imag, 'saddlebrown', lw = 0.5)
ln, = ax.plot(circle.real, circle.imag, 'saddlebrown', lw = 1)
next_circle = circle.copy()

def update(i): 
    global square, circle, next_circle, fibonacci, fl

    if i % frames_per_box == 0:
        square      = 1j * (fibonacci[1] / fibonacci[0]) * (square - square[0]) + square[2]
        next_circle = 1j * (fibonacci[1] / fibonacci[0]) * (next_circle - next_circle[0]) + next_circle[-1]
        circle      = np.hstack([circle, next_circle])
        fibonacci   = fibonacci[1], fibonacci[0] + fibonacci[1]

        fl, = ax.fill(square.real, square.imag, alpha = 0.8, facecolor = 'sandybrown')
        ax.plot(square.real, square.imag, 'saddlebrown', lw = 0.5)

    completion    = (i % frames_per_box) / frames_per_box
    frames_behind = int((1 - completion) * points_in_circle)
    canvas_size   = phi**(i / frames_per_box + 1.8) + 1
    ax.set_xlim(-canvas_size, canvas_size)
    ax.set_ylim(-canvas_size, canvas_size)
    ln.set_data(circle[:-frames_behind].real, circle[:-frames_behind].imag)
    
    if fibonacci[1] > 2:
        mpl.pyplot.setp(fl, alpha = 0.8 * completion)

animation_fig = animation.FuncAnimation(fig, update, frames = total_frames, interval = 50)
animation_fig.save("MIB0059_GoldenRatio.mp4", dpi = 200)

