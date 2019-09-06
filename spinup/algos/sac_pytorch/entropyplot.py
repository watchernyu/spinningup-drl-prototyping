from torch.utils.data import TensorDataset, DataLoader
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn

hidden_size = 32
entropy_net = nn.Sequential(nn.Linear(2,hidden_size),
                            nn.ReLU(),
                            nn.Linear(hidden_size,hidden_size),
                            nn.ReLU(),
                            nn.Linear(hidden_size,1))


sanity_mu = np.linspace(-5, 5, num=100)
entropy_net = torch.load('entropy_net.pkl')


plot_x = []
plot_y = []
for mu in sanity_mu:
	pair = torch.from_numpy(np.array([mu, 0.3])).float()
	H = entropy_net(pair)
	plot_x.append(mu)
	plot_y.append(H)
plt.plot(plot_x, plot_y)
plt.xlabel('mu')
plt.ylabel('entropy')
plt.title('sigma=0.3')
plt.show()