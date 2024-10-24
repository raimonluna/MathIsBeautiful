import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from tqdm.auto import tqdm
import matplotlib.pyplot as plt
from matplotlib import animation

torch.manual_seed(42)

###### Parameters of the simulation

hidden_layers     = 4
neurons_per_layer = 256 
neuron_leak       = 0.1
learning_rate     = 1e-3
random_angles     = 20 
number_points     = 300
train_epochs      = 10000
device            = 'cpu' # torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print('Using device:', device)

###### Neural network definition

class Sofa(nn.Module):
    def __init__(self):
        super(Sofa, self).__init__()
        
        self.net = nn.Sequential()
        self.net.add_module('Input', nn.Linear(2, neurons_per_layer))
        self.net.add_module('Input activation', nn.LeakyReLU(neuron_leak))

        for l in range(hidden_layers):
            self.net.add_module(f'Hidden {l+1}', nn.Linear(neurons_per_layer, neurons_per_layer))
            self.net.add_module(f'Hidden {l+1} activation', nn.LeakyReLU(neuron_leak))
            
        self.net.add_module('Output', nn.Linear(neurons_per_layer, 2))

        for m in self.net.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                nn.init.constant_(m.bias, val = 0.0)
                
    def forward(self, phi):
        circle      = torch.cat((1.5*torch.cos(phi), 0.5*torch.sin(phi)), 1)
        input_coord = torch.cat((torch.cos(phi), torch.sin(phi)), 1)
        correction  = self.net(input_coord)
        return circle * (1 + correction)


###### Neural network training

arg  = torch.linspace(0, 2*np.pi, number_points).reshape(number_points,1).to(device)
sofa = Sofa().to(device)
relu = nn.ReLU()

optimizer = optim.Adam(sofa.parameters(), lr = learning_rate)
scheduler = optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.999)

for epoch in tqdm(range(train_epochs)):
    
    optimizer.zero_grad()

    x  = sofa(arg)
    dx = x - torch.roll(x, 1, dims=0)

    area      = (torch.sum(x[:, 0] * dx[:, 1]) - torch.sum(x[:, 1] * dx[:, 0])) / 2
    perimeter = torch.sum(torch.sqrt(torch.sum(dx**2, axis = 1)))
    length    = torch.max(x[:, 0]) - torch.min(x[:, 0])

    nonunif   = torch.std(torch.sqrt(torch.sum(dx**2, axis = 1)))
    xc        = (torch.max(x[:, 0]) + torch.min(x[:, 0])) / 2
    yc        = (torch.max(x[:, 1]) + torch.min(x[:, 1])) / 2
    
    centering = xc**2 + yc**2
    overlap   = 0

    for theta in torch.rand(random_angles) * np.pi / 2:
        u =   x[:, 0] * torch.cos(theta) + x[:, 1] * torch.sin(theta)
        v = - x[:, 0] * torch.sin(theta) + x[:, 1] * torch.cos(theta)
        overlap += torch.sum(relu(torch.max(u) - 1.0 - u) * relu(torch.max(v) - 1.0 - v)) / random_angles

    weight  = learning_rate / scheduler.get_last_lr()[0]
    loss    = 10 * nonunif + weight * (overlap + centering) - area

    loss.backward()
    optimizer.step()
    scheduler.step()
    
###### Making the movie

print('Now making the movie. Please wait...')
x      = sofa(arg).cpu().detach().numpy()
size   = length.item()

smallsq = np.array([[1, 1], [-1, 1], [-1, -1], [1, -1], [1, 1]])
largesq = 2 * smallsq

frame_n  = 100
params_1 = np.vstack([np.zeros(frame_n), (4 - size) * np.linspace(-1, 1, frame_n) / 2 , 1.5*np.ones(frame_n)]).T
params_2 = np.zeros((2 * frame_n, 3))
for i, theta in enumerate(np.linspace(0, np.pi/2, 2 * frame_n)):
    u =   x[:, 0] * np.cos(theta) + x[:, 1] * np.sin(theta)
    v = - x[:, 0] * np.sin(theta) + x[:, 1] * np.cos(theta)
    params_2[i] = [theta, - np.max(u) + 2, - np.max(v) + 2]
params = np.vstack([params_1, params_2])
params = np.vstack([np.hstack([params[:, [0]] + th, 
                               params[:, [1]] * np.cos(th) + params[:, [2]] * np.sin(th), 
                             - params[:, [1]] * np.sin(th) + params[:, [2]] * np.cos(th)]) for th in np.arange(4) * np.pi/2 ])

fig, ax = plt.subplots(frameon=False, figsize = (8, 8))
fig.subplots_adjust(left=0, bottom=0, right=1, top=1, wspace=None, hspace=None)
plt.close()

def animate(i):
    ax.cla()
    ax.set_facecolor("orange")
    ax.xaxis.set_visible(False)
    ax.yaxis.set_visible(False)
    ax.fill(*largesq.T, color = 'wheat')
    ax.plot(*largesq.T, color = 'black')
    ax.fill(*smallsq.T, color = 'orange')
    ax.plot(*smallsq.T, color = 'black')
    theta, du, dv = params[i]
    u =   x[:, 0] * np.cos(theta) + x[:, 1] * np.sin(theta) + du
    v = - x[:, 0] * np.sin(theta) + x[:, 1] * np.cos(theta) + dv
    sofa_plot = ax.fill(u,v, color = 'darkblue')
    return fig

animation_fig = animation.FuncAnimation(fig, animate, frames = len(params), interval = 10)
animation_fig.save("MIN0009_MovingSofa.mp4", dpi = 200)

