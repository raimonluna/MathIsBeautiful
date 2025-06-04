import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import matplotlib.animation as animation
from scipy.fft import fft, fftshift

n_modes  = 50
n_frames = 500

##### Drawing the umbrella

def semicirc(d, orientation = 'upper left', N = 21):
    out = (d * np.exp(1j * np.linspace(0, np.pi, N)) - d) / 2
    if 'right' in orientation:
        out -= 2*out.real
    if 'lower' in orientation:
        out = out.conj()
    return out

def add_curve(curve, newpart):
    return np.hstack([curve[:-1], curve[-1] +  newpart])

r_small = 9.5 / 8
curve = np.array([5])
curve = add_curve(curve, semicirc(10,      'upper left',  50))
curve = add_curve(curve, semicirc(r_small, 'lower right', 10))
curve = add_curve(curve, semicirc(r_small, 'upper right', 10))
curve = add_curve(curve, semicirc(r_small, 'lower right', 10))
curve = add_curve(curve, semicirc(r_small, 'upper right', 10))
curve = add_curve(curve, np.linspace(0, -5j, 20))
curve = add_curve(curve, semicirc(1.5,     'lower right', 10))
curve = add_curve(curve, semicirc(0.5,     'upper left',  5))
curve = add_curve(curve, semicirc(0.5,     'lower left',  5))
curve = add_curve(curve, np.linspace(0, +5j, 20))
curve = add_curve(curve, semicirc(r_small, 'upper right', 10))
curve = add_curve(curve, semicirc(r_small, 'lower right', 10))
curve = add_curve(curve, semicirc(r_small, 'upper right', 10))
curve = add_curve(curve, semicirc(r_small, 'lower right', 10))
curve = curve[:-1] # Make it periodic!

##### Fast Fourier Transform 
n_points   = len(curve)
coefs      =  fftshift(fft(curve)) / n_points
theta_full = np.linspace(0, 2 * np.pi, 1000)

##### Plotting 
fig, ax  = plt.subplots(1, 1, figsize = (6, 6))
fig.patch.set_facecolor('black')
fig.subplots_adjust(left=0, bottom=0, right=1, top=1, wspace=None, hspace=None)
plt.close()
ax.set_xlim(-6, 6)
ax.set_ylim(-6.25, 5.75)
ax.axis('off')

phasors  = [[ax.plot([0], [0], '.-', color = 'red', alpha = 0.5, lw = 1.5)[0] for k in range(1, n_modes)],
            [ax.plot([0], [0], '.-', color = 'red', alpha = 0.5, lw = 1.5)[0] for k in range(1, n_modes)]]
circles  = [[ax.plot([0], [0], color = 'yellow',    alpha = 0.5, lw = 1.0)[0] for k in range(1, n_modes)],
            [ax.plot([0], [0], color = 'yellow',    alpha = 0.5, lw = 1.0)[0] for k in range(1, n_modes)]]
umbrella = ax.plot([0], [0], 'blue', lw = 2)[0]

def update(i):
    
    t = i * 2 * np.pi / n_frames
    theta_part = np.linspace(0 + t * (i > n_frames), t + (4 * np.pi - t) * (i > n_frames), 1000)
    
    center = reconstruction = coefs[n_points//2]
    for k in range(1, n_modes):
        for s in (1, -1):  
            
            phase           = coefs[n_points//2 + s * k] * np.exp(s * 1j * k * t)
            circle          = center + np.abs(phase) * np.exp(1j * theta_full)
            reconstruction += coefs[n_points//2 + s * k] * np.exp( s * 1j * k * theta_part)

            phasors[(1 + s) // 2][k - 1].set_data([center.real, (center + phase).real], [center.imag, (center + phase).imag])
            circles[(1 + s) // 2][k - 1].set_data(circle.real, circle.imag)
            
            center += phase
    
    umbrella.set_data(reconstruction.real, reconstruction.imag)

animation_fig = animation.FuncAnimation(fig, update, frames = 2 * n_frames, interval = 25)
animation_fig.save("MIB0040_FourierSeries.mp4", dpi = 200)
