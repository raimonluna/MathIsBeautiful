import numpy as np
import matplotlib.pyplot as plt

turing_code = "1RB1LC_1RC1RB_1RD0LE_1LA1LD_1RZ0LA"
total_iter  = 35000

memory  = int(2*np.sqrt(total_iter))
tape    = np.zeros((memory), int)
storage = np.zeros((total_iter, memory), int)

pos     = len(tape) - 40
rule    = 0

for step in range(total_iter):
    bit       = tape[pos]
    command   = turing_code.split('_')[rule][3*bit:3 + 3*bit]
    
    rule = ord(command[2]) - ord('A')
    if rule < 0 or rule > 4: 
        print(f'The machine has halted at step {step + 1}!')
        break

    tape[pos] = int(command[0])
    pos      += np.sign(ord(command[1]) - 80)  
    storage[step, :] = tape *  np.arange(memory)

fig, ax = plt.subplots(1,1, figsize = (10,10))
ax.axis('off')
ax.imshow(storage, aspect = memory / total_iter, cmap = 'gist_heat', interpolation = 'none')
plt.savefig('MIB0032_BusyBeaver.png', dpi = fig.dpi, bbox_inches='tight', pad_inches = 0)
