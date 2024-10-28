import numpy as np
import matplotlib.pyplot as plt

side = 7

def join_2s(ls):
    rep = ('[' + ']['.join([str(i) for i in ls]) + ']').replace('[2][2]', '[4]')
    return [int(i) for i in rep[1:-1].split('][')]

def factors(n):
    facts = [1]
    k = 2
    while n > 1:
        if n % k == 0:
            n /= k
            facts.append(k)
        else:
            k += 1
    return join_2s(facts)[::-1]
    
def factor_plot(ax, factors = [1], pos = 0 + 0j, size = 0 + 1j):
    factor = factors[0]
    circle1 = plt.Circle((pos.real, pos.imag), 1.2 * np.abs(size), color = 'forestgreen', alpha = 0.5)
    ax.add_patch(circle1)
    if factor == 1:
        circle1 = plt.Circle((pos.real, pos.imag), np.abs(size), color = 'indigo')
        circle2 = plt.Circle((pos.real - 0.4 * np.abs(size), pos.imag + 0.4 * np.abs(size)), 0.2 * np.abs(size), color = 'white')
        ax.add_patch(circle1)
        ax.add_patch(circle2)
        return
    for k in range(factor):
        angle   = np.exp(1j * ( 2 * np.pi * k / factor))
        correct = np.exp(1j * (factor == 4) * np.pi / 4 )
        dotpos  = pos + size * angle * correct
        factor_plot(ax, factors[1:], dotpos, 1.3 * angle * size / factor )
        

fig, ax = plt.subplots(figsize = (20,20))

for i in range(1, side + 1):
    for j in range(1, side + 1):
        num = side * (side - j) + i
        factor_plot(ax = ax, factors = factors(num), pos = 4 *(i + j*1j), size = 0 + 1j)

ax.set_xlim(1, 4.4 * side)
ax.set_ylim(1, 4.4 * side)
ax.set_facecolor("palegreen")
ax.xaxis.set_visible(False)
ax.yaxis.set_visible(False)
plt.gca().set_aspect('equal')
plt.savefig('MIB0010_FactorizationDiagram.png', dpi = fig.dpi, bbox_inches='tight', pad_inches = 0)
