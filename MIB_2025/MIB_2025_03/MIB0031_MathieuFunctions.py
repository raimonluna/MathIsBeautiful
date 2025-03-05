import cupy as cp
import matplotlib.pyplot as plt
import matplotlib as mpl

shape        = (2000, 2000)
xrange       = (-15, 15)
yrange       = (-5,  25)
total_iters  = 100

q  = cp.linspace(*xrange, shape[0])[:, None]
a  = cp.linspace(*yrange, shape[1])[None, :]

# Last index is (even, even', odd, odd')
mathieu = cp.ones(shape + (4,)) * cp.array([1,0,0,1])
x, dx   = 0.0,  cp.pi / total_iters 

def mathieu_equation(x, mathieu):
    return cp.dstack([mathieu[..., 1],  
                      mathieu[..., 0] * (2 * q * cp.cos(2 * x) - a),
                      mathieu[..., 3],  
                      mathieu[..., 2] * (2 * q * cp.cos(2 * x) - a)])
def RK4(x, mathieu):
    k1 = mathieu_equation(x,            mathieu)
    k2 = mathieu_equation(x + 0.5 * dx, mathieu + 0.5 * k1 * dx)
    k3 = mathieu_equation(x + 0.5 * dx, mathieu + 0.5 * k2 * dx)
    k4 = mathieu_equation(x + dx,       mathieu + k3 * dx)
    return x + dx, mathieu + dx * (k1 + 2*k2 + 2*k3 + k4) / 6

for it in range(total_iters): x, mathieu = RK4(x, mathieu)
full_eigenvalues = cp.asnumpy(cp.minimum(cp.abs(mathieu[..., 1]), cp.abs(mathieu[..., 2])))

plt.axis('off')
plt.imshow(full_eigenvalues.T, norm = 'log', origin = 'lower', extent = xrange + yrange, 
           cmap = 'PuOr_r', vmin = 5e-3, vmax = 1e2, interpolation = None)
plt.savefig('MIB0031_MathieuFunctions.png', dpi = 1000, bbox_inches='tight', pad_inches = 0)
