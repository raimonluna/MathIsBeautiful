import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from scipy.fft import fft, ifft, fftfreq

domain_length   = 32 * np.pi
point_number    = 1000
frames          = 1000
steps_per_frame = 10
dt              = 0.1
np.random.seed(42)

stored = np.zeros((point_number, point_number))
u = -1 + 2*np.random.rand(point_number)
x = np.linspace(0, domain_length, point_number+1)[:-1]

D = 1j * point_number * fftfreq(point_number) * (2 * np.pi / domain_length)
kernel  = np.abs(D) < point_number // 3
D[point_number//2] = 0

L  = - D**2 - D**4
eL = np.exp(dt * L)
eN = np.where(L == 0, dt, (eL - 1) / L)

def ETD_step(u):
    fft_lin = fft(u)
    fft_non = kernel * D * fft(-0.5 * u**2)
    return ifft(eL * fft_lin + eN * fft_non).real

for it in range(2*point_number):
    u = ETD_step(u)
    stored[it % point_number, :] = u

##### Do the plotting

fig, ax = plt.subplots(1,1, figsize = (6, 6))
fig.subplots_adjust(left=0, bottom=0, right=1, top=1, wspace=None, hspace=None)
ax.axis('off')
plt.close()

ish = ax.imshow(stored.T, origin = 'lower', cmap = 'twilight')

first = True
def update(i):
    global u, first
    if first:
        first = False
        return
    for it in range(steps_per_frame):
        u = ETD_step(u)
        stored[((i + 0) * steps_per_frame + it) % point_number, :] = u
    indices = (i + 1) * steps_per_frame + np.arange(point_number)
    to_plot = stored[indices % point_number, :]
    ish.set_data(to_plot.T)

animation_fig = animation.FuncAnimation(fig, update, frames = frames, interval = 50)
animation_fig.save("MIB0044_KuramotoSivashinsky.mp4", dpi = 200)
