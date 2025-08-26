import numpy as np
from tqdm.auto import tqdm
import matplotlib.pyplot as plt
import matplotlib as mpl
import matplotlib.animation as animation

shape = (400, 100)
iters = 4000
rho0  = 100
tau   = 0.55
frame_every = 8
np.random.seed(42)

x     = np.arange(shape[0])[:, None]
y     = np.arange(shape[1])[None, :]
curl  = np.zeros(shape + (iters//frame_every,)) 

D2Q9  = np.array([[di, dj, 4 / (9 * 4**(di**2 + dj**2))] for di in (-1,0,1) for dj in (-1,0,1)])
stone = (x - 50)**2 + (y - 50)**2 < 10**2

# Initial conditions
F = np.ones(shape + (9,)) + 0.01 * np.random.randn(*shape, 9)
F[:,:,7] += 2 * (1 + 0.2 * np.cos(2 * np.pi * x / shape[0] * 4))
F *= rho0 / np.sum(F, 2)[..., None]

print('Evolving fluid...')
for it in tqdm(range(iters)):
    
    # Advection
    for k, (di, dj, w) in enumerate(D2Q9):
        F[:,:,k] = np.roll(F[:,:,k], int(di), axis = 0)
        F[:,:,k] = np.roll(F[:,:,k], int(dj), axis = 1)
    
    # Save stone
    stoneF = F[stone,:]
    
    # Hydro variables
    rho = np.sum(F, 2)
    ux  = np.sum(F * D2Q9[:, 0], 2) / rho
    uy  = np.sum(F * D2Q9[:, 1], 2) / rho
    
    # Collisions
    Feq = rho[..., None] * D2Q9[:, 2] * (1 + 3 * (D2Q9[:, 0] * ux[..., None] + D2Q9[:, 1] * uy[..., None])    +  \
                                         (9/2) * (D2Q9[:, 0] * ux[..., None] + D2Q9[:, 1] * uy[..., None])**2 -  \
                                         (3/2) * (ux[..., None]**2 + uy[..., None]**2))
    F += - (1 / tau) * (F - Feq)
    
    # Apply boundaries 
    F[stone,:]   = stoneF[:, ::-1]

    #Compute vorticity
    if it % frame_every == 0:
        dvx_dy = np.roll(ux, -1, axis = 1) - np.roll(ux, 1, axis = 1)
        dvy_dx = np.roll(uy, -1, axis = 0) - np.roll(uy, 1, axis = 0)
        curl[..., it//frame_every] =  dvx_dy - dvy_dx

##### Do the plotting
print('Making the animation...')

cmap = mpl.colors.LinearSegmentedColormap.from_list("", [(0.0, 'blue'), (0.25, 'blue'), (0.5, 'black'), (0.75, 'red'), (1, 'red')])

fig, ax = plt.subplots(frameon = False, figsize = (10, 10))
fig.subplots_adjust(left=0, bottom=0, right=1, top=1, wspace=None, hspace=None)
ax.axis('off')
plt.close()

ish = ax.imshow(np.tile(curl[..., -1], (1, 4)).T, cmap=cmap, interpolation = 'bicubic', origin = 'lower', vmin = -.1, vmax = .1)

for k in range(4):
    stoneshape = 50 + 50j + k*100j + 10 * np.exp(1j * np.linspace(0, 2*np.pi, 100))
    ax.fill(stoneshape.real, stoneshape.imag, color = 'w', zorder = 2)

animation_fig = animation.FuncAnimation(fig, lambda i: ish.set_data(np.tile(curl[..., i], (1, 4)).T), frames = iters//frame_every, interval = 50)
animation_fig.save("MIB0050_VonKarman.mp4", dpi = 200)
