import numpy as np
import matplotlib.pyplot as plt

##### Useful functions
def common_edges(tile1, tile2, ind):
    counter = 0
    while np.isclose(tile1[(ind[0] + counter) % len(tile1)], 
                     tile2[(ind[1] - counter) % len(tile2)]):
        counter += 1
    return counter

def place_tile(tile1, tile2, ind):
    displacement = - tile2[ind[1]] + tile1[ind[0]]
    return tile2 + displacement, displacement

def merge_tiles(tile1, tile2):
    index  = 0
    while np.min(np.abs( tile1[index % len(tile1)] - tile2 )) < 0.001:
        index += 1
    merged = [tile1[index - 1]] 
    while np.min(np.abs( tile1[index % len(tile1)] - tile2 )) > 0.001:
        merged.append(tile1[index % len(tile1)])
        index += 1
    merged.append(tile1[index % len(tile1)])
    index = np.argmin(np.abs( tile1[index % len(tile1)] - tile2)) + 1
    while np.min(np.abs( tile2[index % len(tile2)] - tile1 )) > 0.001:
        merged.append(tile2[index % len(tile2)])
        index += 1
    return np.array(merged)

def apply_transform(old_hats, new_hats, rotation, displacement):
    for hat, color in new_hats:
        old_hats += [(rotation * hat + displacement, color)]

def plot_hats(ax, hats):
    for hat, color in hats:
        ax.fill(hat.real, hat.imag, color = color, edgecolor = 'k')
        
##### Define the fundamental tile
l_small = 1 / (2 *np.sqrt(3))
l_large = 0.5

tile   = (1 + 0j) * np.array([l_large] + 4*[l_small] + 2 * (2*[l_large] + 2*[l_small]))
tile  *= np.exp(1j * np.pi / 6)**np.array([1,4,6,6,8,5,7,10,0,9,11,2,0])
tile   = np.hstack([0, np.cumsum(tile)]) * np.exp(1j * 2 * np.pi / 6)
rtile  = tile.conj()[::-1] 

##### Define fundamental metatiles
H8 = np.roll(tile, -3)
inverted_tile, _ = place_tile(H8, rtile, (0, 2))
H7 = merge_tiles(H8, inverted_tile)
H7 = np.roll(H7, -9)

H8_hats_old = [(H8, '#E5D143')]
H8_green    = [(H8, '#A6E546')]
H7_hats_old = [(H8, '#A6E546'), (inverted_tile, '#2F9937')]
H8_hats_new = []
H7_hats_new = []

##### Substitution rules for supertiles
angles = [8, 10, 0, 2, 4, 0]
shifts = [6, 0, 2, 0, 23, 0]

##### Iterations
print('Generating tiling...')
for it in range(5):

    joint_clusters = H7
    H8_hats_new   += H7_hats_old
    H7_hats_new   += H7_hats_old

    for mt in range(len(angles)):
        
        rotation = np.exp(1j * angles[mt] * np.pi / 6)
        added    = H8 * rotation
        
        max_vertices = 0
        for i in range(len(added)):
            new_cluster, disp = place_tile(joint_clusters, added, (0, i))
            vertices          = common_edges(joint_clusters, new_cluster, (0, i))
            if vertices > max_vertices:
                best_cluster, max_vertices, best_disp = new_cluster, vertices, disp

        joint_clusters = merge_tiles(joint_clusters, best_cluster)
        joint_clusters = np.roll(joint_clusters, shifts[mt] * len(joint_clusters) // 42)

        if (len(H8_hats_old) == 1) and (mt in (0, 2)):
            apply_transform(H7_hats_new, H8_green, rotation, best_disp)
            apply_transform(H8_hats_new, H8_green, rotation, best_disp)
        else:
            if mt <= 4:
                apply_transform(H7_hats_new, H8_hats_old, rotation, best_disp)
                newH7 = joint_clusters
            apply_transform(H8_hats_new, H8_hats_old, rotation, best_disp)
            newH8 = joint_clusters

    newH7 = np.roll(newH7, -np.argmax(newH7.real + newH7.imag) - 2 * len(joint_clusters) // 42)
    newH8 = np.roll(newH8, -np.argmax(newH8.real + newH8.imag) - 2)
    
    H7, H8 = newH7.copy(), newH8.copy()
    H7_hats_old, H8_hats_old = H7_hats_new.copy(), H8_hats_new.copy()
    
##### Plotting
fig, ax  = plt.subplots(1, 1, figsize = (10, 10))
ax.set_xlim(-40, -10)
ax.set_ylim(-15, 15)
ax.axis('off')

print(f'Generated {len(H8_hats_new)} tiles! Now plotting...')
filtered_hats = [hat for hat in H8_hats_new if (np.abs(hat[0] + 25) < 22).any()]
plot_hats(ax, filtered_hats)

plt.savefig('MIB0043_EinsteinTile.png', dpi = 1000, bbox_inches='tight', pad_inches = 0)
