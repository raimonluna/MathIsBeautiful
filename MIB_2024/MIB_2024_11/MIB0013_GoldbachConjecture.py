import numpy as np
import matplotlib.pyplot as plt

size   = 100000
shape  = (400, 400)
xrange = (0, size + 1)
yrange = (0, 2200)

##### Sieve of Eratosthenes
is_prime = np.array(size*[True])
is_prime[[0,1]] = False
for i in range(2, size):
    if is_prime[i]:
        is_prime[2*i::i] = False
primes = np.arange(size)[is_prime]

##### Do the counting
a, b = np.meshgrid(primes, primes)
sums = (a + b)
sums = np.vstack([sums, np.diag(sums)])

sums = sums[sums <= size]
sums = sums[sums % 2 == 0]

unique, counts = np.unique(sums.flatten(), return_counts=True)
counts //= 2

##### Do the plotting
board = np.zeros(shape)
idx1  = np.int32( shape[0] * (unique - xrange[0]) / (xrange[1] - xrange[0]) )
idx2  = np.int32( shape[1] * (counts - yrange[0]) / (yrange[1] - yrange[0]) )
for i,j in zip(idx1, idx2):
    board[i, j] += 1

plt.axis('off')
plt.imshow(board.T, cmap = 'gnuplot', origin = 'lower', vmin = 0, vmax = 0.06 * np.max(board))
plt.savefig('MIB0013_GoldbachConjecture.png', dpi = 1000, bbox_inches='tight', pad_inches = 0)
