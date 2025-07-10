import cupy as cp
from tqdm.auto import tqdm
import matplotlib.pyplot as plt
cp.random.seed(42)

side       = 500
batch_size = side**2 // 10
batch_num  = 100000
T_critical = 2 / cp.log(1 + cp.sqrt(2))
T          = T_critical
exp_table  = cp.exp(-cp.arange(-8, 9, 4)/T)
lattice    = lattice   = 1 - 2 * cp.random.randint(0, 2, (side, side))

def delta_energy(lattice, candidates_i, candidates_j):
    deltaH = cp.zeros_like(candidates_i)
    for di, dj in [(-1, 0), (0, -1), (0, 1), (1, 0)]:
        deltaH += lattice[(candidates_i + di) % side, (candidates_j + dj) % side]
    deltaH *= 2*lattice[candidates_i, candidates_j]
    return deltaH

for it in tqdm(range(batch_num)):
    candidates_i, candidates_j = cp.random.randint(0, side, (2, batch_size))
    deltaH   = delta_energy(lattice, candidates_i, candidates_j)
    accepted = cp.random.rand(batch_size) <  exp_table[(deltaH / 4 + 2).astype(cp.int32)]
    lattice[candidates_i[accepted], candidates_j[accepted]] *= -1

fig, ax = plt.subplots(1, 1, figsize = (8,8))
ax.axis('off')
ax.imshow(cp.asnumpy(lattice), cmap = 'copper_r', interpolation = 'none')
plt.savefig('MIB0046_IsingModel.png', dpi = 1000, bbox_inches='tight', pad_inches = 0)
