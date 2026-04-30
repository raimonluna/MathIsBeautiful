import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from scipy.linalg import solve, norm
from scipy.interpolate import interp1d

N = 64
trail_num  = 20
interp_num = 500

def FourierD(N):
    row = np.arange(N)[:, None]
    col = np.arange(N)[None, :]
    I  = np.eye(N)
    Ic = np.ones((N, N)) - I
    
    x = 2 * np.pi * row / N
    h = (x[1] - x[0])[0]
    
    if N % 2 == 0:
        center = - (N**2 + 2) / 12
        D1 = Ic * 0.5 * (-1)**(row + col) / np.tan(0.5 * ((row - col) * h) + I)
        D2 = Ic * 0.5 * (-1)**(row + col + 1) / np.sin(0.5 * ((row - col) * h) + I)**2 + center * I
    else:
        center = (1 - N**2) / 12
        D1 = Ic * 0.5 * (-1)**(row + col) / np.sin(0.5 * ((row - col) * h) + I)
        D2 = Ic * 0.5 * (-1)**(row + col + 1) / (np.sin(0.5 * ((row - col) * h)) * np.tan(0.5 * ((row - col) * h)) + I) + center * I
    return x, I, D1, D2

def Diag(f): 
    return np.diag(f.ravel())

def MatrixProblem(orbits):

    coords   = orbits.reshape(4, N, 1)
    x_coords = coords[0:2]
    y_coords = coords[2:4]

    J = np.zeros((4*N, 4*N))
    b = np.zeros((4*N, 1))
    
    for body1 in range(2): # What body are we moving?        
        body2 = (body1 + 1) % 2

        # Be careful with these definitions...
        dx12  = x_coords[body1] - x_coords[body2]
        dy12  = y_coords[body1] - y_coords[body2]
        dx13  = 2 * x_coords[body1] + x_coords[body2]
        dy13  = 2 * y_coords[body1] + y_coords[body2]
        dr12  = np.sqrt(dx12**2 + dy12**2)
        dr13  = np.sqrt(dx13**2 + dy13**2)
        
        J[(body1 + 0)*N:(body1 + 1)*N, (body1 + 0)*N:(body1 + 1)*N] = D2 - Diag( (2 * dx12**2 - dy12**2) / dr12**5 + 2 * (2 * dx13**2 - dy13**2) / dr13**5 )
        J[(body1 + 2)*N:(body1 + 3)*N, (body1 + 2)*N:(body1 + 3)*N] = D2 - Diag( (2 * dy12**2 - dx12**2) / dr12**5 + 2 * (2 * dy13**2 - dx13**2) / dr13**5 )

        J[(body1 + 0)*N:(body1 + 1)*N, (body1 + 2)*N:(body1 + 3)*N] = - 3 * Diag( dx12 * dy12 / dr12**5 + 2 * dx13 * dy13 / dr13**5 )
        J[(body1 + 2)*N:(body1 + 3)*N, (body1 + 0)*N:(body1 + 1)*N] = - 3 * Diag( dx12 * dy12 / dr12**5 + 2 * dx13 * dy13 / dr13**5 )

        J[(body1 + 0)*N:(body1 + 1)*N, (body2 + 0)*N:(body2 + 1)*N] = Diag( ( 2 * dx12**2 - dy12**2 ) / dr12**5 - ( 2 * dx13**2 - dy13**2 ) / dr13**5 )
        J[(body1 + 2)*N:(body1 + 3)*N, (body2 + 2)*N:(body2 + 3)*N] = Diag( ( 2 * dy12**2 - dx12**2 ) / dr12**5 - ( 2 * dy13**2 - dx13**2 ) / dr13**5 )

        J[(body1 + 0)*N:(body1 + 1)*N, (body2 + 2)*N:(body2 + 3)*N] = + 3 * Diag( dx12 * dy12 / dr12**5 - dx13 * dy13 / dr13**5 )
        J[(body1 + 2)*N:(body1 + 3)*N, (body2 + 0)*N:(body2 + 1)*N] = + 3 * Diag( dx12 * dy12 / dr12**5 - dx13 * dy13 / dr13**5 )

        b[(body1 + 0)*N:(body1 + 1)*N, :]                           = - dx12 / dr12**3 - dx13 / dr13**3 - D2 @ x_coords[body1]        
        b[(body1 + 2)*N:(body1 + 3)*N, :]                           = - dy12 / dr12**3 - dy13 / dr13**3 - D2 @ y_coords[body1]

    return J, b

def LevenbergMarquardt(equations, orbits, tol = 1e-10, max_iter = 100):

    residual, lam = 1, 1e-3
    
    for it in range(max_iter):
        J, b     = equations(orbits)
        if norm(b) < tol:
            break

        correction = solve(J.T @ J + lam * np.eye(J.shape[0]), J.T @ b)
        new_orbits = orbits + correction
        _, b_new   = MatrixProblem(new_orbits)

        if norm(b_new) < norm(b):
            orbits = new_orbits
            lam   *= 0.7
        else:
            lam   *= 2.0

    return orbits.ravel(), norm(b), it

###################################### Find some orbits #########################################

t, I, D1, D2 = FourierD(N)

s1, c1, s2, c2, s6, c6  = np.sin(t), np.cos(t), np.sin(2*t), np.cos(2*t), np.sin(6*t), np.cos(6*t)
seeds = np.array([[0.2 * (c1 + c2) , s1 + s2, 0.2 * (c1 - c2), s1 - s2],
                  [c1, 4 * s1, 4 * c1, s1],
                  [c1 + c6 , s1 + s6, c1 - c6, s1 - s6], 
                  [c1, s2, np.roll(c1, N//3), np.roll(s2, N//3)]])

orbits = np.zeros((4, 4*N))
for k in range(4):
    seed = np.vstack([seeds[k, 0] + seeds[k, 1], seeds[k, 2] + seeds[k, 3], 
                      seeds[k, 0] - seeds[k, 1], seeds[k, 2] - seeds[k, 3]])
    orbits[[k], :], residual, it = LevenbergMarquardt(MatrixProblem, seed)
    print(f'Found orbit number {k} in {it} iterations! Residual: {residual}')

#################################### Interpolate the orbits #######################################

coarse_coords = orbits.reshape(4, 4, N)
coarse_coords = np.dstack([coarse_coords, coarse_coords[..., [0]]]) 
long_t        = np.hstack([t.ravel(), [2 * np.pi]])
interpolated  = interp1d(long_t, coarse_coords, axis = 2, kind = 'cubic')
coords        = interpolated(np.linspace(0, 2 * np.pi, interp_num) )
x_coords      = np.hstack([coords[:, 0:2, :], - coords[:, [0], :] - coords[:, [1], :]])
y_coords      = np.hstack([coords[:, 2:4, :], - coords[:, [2], :] - coords[:, [3], :]])

######################################## Do the plotting ##########################################

fig, axes = plt.subplots(2,2, figsize = (6, 6));
fig.subplots_adjust(left=0.05, bottom=0.05, right=0.95, top=0.95, wspace=None, hspace=None)
fig.patch.set_facecolor('black')
plt.close()
plt.tight_layout()

planets = []
for frame, ax in enumerate(axes.ravel()):
    ax.axis('off')
    [ax.plot(x_coords[frame, body], y_coords[frame, body], lw = 1, alpha = 0.5,  color = ('r', 'g', 'b')[body]) for body in range(3)]
    planets.append([ax.scatter(x_coords[frame, body, -trail_num:], y_coords[frame, body, -trail_num:], s = 100,
                    cmap = ('Reds', 'Greens', 'Blues')[body], c = np.arange(trail_num)) for body in range(3)])

def update(i): 
    trail = (4 * i) % (interp_num) - np.arange(trail_num)
    for frame in range(4):
        for body in range(3):
            planets[frame][body].set_offsets(np.vstack([x_coords[frame, body, trail], y_coords[frame, body, trail]]).T)
    
print('Making the movie, please wait...')
animation_fig = animation.FuncAnimation(fig, update, frames = 2 * interp_num, interval = 50)
animation_fig.save("MIB0062_ThreeBodyProblem.mp4", dpi = 200)
