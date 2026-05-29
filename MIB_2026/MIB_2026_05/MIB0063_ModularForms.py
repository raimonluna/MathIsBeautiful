import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt

shape  = (1000, 1000)
xrange = (-1.05, 1.05)
yrange = (-1.05, 1.05)
weight = 6
num_terms = 50

zeta = [0, -1/12, 1/120, -1/252, 1/240, -1/132, 691/32760, -1/12, 3617/8160, -43867/14364, 174611/6600][weight//2]

rec    = np.linspace(*xrange, shape[0])[:, None]
imc    = np.linspace(*yrange, shape[1])[None, :]
c      = rec + 1j * imc
disk   = np.abs(c) < 1

# Move tau into the fundamental domain

q    = c[disk]
tau  = np.log(q) / (2 * np.pi * 1j)

tau_factor = np.ones_like(tau)
need_S = (tau == tau)

while need_S.any():
    
    #Apply T as needed
    tau -= np.round(tau.real) 

    #Apply S and keep track of the tau^weight factors
    need_S = np.abs(tau) < 1    
    tau[need_S] = - 1 / tau[need_S]
    tau_factor[need_S] *= tau[need_S]**weight

q = np.exp(2 * np.pi * 1j * tau)

# Sum over the Eisenstein series
EE = np.zeros_like(q)
for n in range(1, num_terms + 1):
    EE += n**(weight - 1) * q**n / (1 - q**n)
EE = tau_factor * (1 + 2 * EE / zeta)

# Do the plotting
canvas = np.ones(shape + (3,))
canvas[disk, 0] = (np.angle(EE) + np.pi) / (2 * np.pi)
canvas[disk, 2] = 0.3 + 0.7 * ((np.log10(np.abs(EE)) ) % 1)
canvas = mpl.colors.hsv_to_rgb(canvas)
canvas[~disk, :] = 0

plt.axis('off')
plt.imshow(canvas.transpose(1,0,2), origin = 'lower', extent = xrange + yrange)
plt.savefig('MIB0063_ModularForms.png', dpi = 1000, bbox_inches='tight', pad_inches = 0)
