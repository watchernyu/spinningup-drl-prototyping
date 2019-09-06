import numpy as np
import torch
import torch.nn as nn
import math
from torch.utils.data import TensorDataset, DataLoader
import matplotlib.pyplot as plt

def calculate_log(y, mu, sigma):
	log_list = []
	ep = 0.00001
	for yi in y:
		if yi > 1 - ep:
			yi = 1 - ep
		elif yi < -1 + ep:
			yi = -1 + ep
		inverse_tanh = 0.5 * (math.log(1+yi) - math.log(1-yi))
		#print ("inverse")
		#print (inverse_tanh)
		#print ("term1")
		#print (-0.5 * math.log(2 * math.pi * sigma * sigma))
		#print ("term2")
		#print ((1 - yi**2))
		#print ("term3")
		#print ((inverse_tanh - mu) * (inverse_tanh - mu))
		#print ("term4")
		#print (2 * sigma * sigma)
		#print ("log")
		log = -0.5 * math.log(2 * math.pi * sigma**2) - (1 - yi**2) - math.pow(inverse_tanh - mu, 2)/(2 * math.pow(sigma, 2))
		#print (log)
		log_list.append(log)
	return log_list

#calculate_log([0], 0, 0.01)

def sanity_check():
	#pair = (mu, sigma)
	mu = 0
	sigma = 0.01
	samples = np.random.normal(mu, sigma, 10000)
	y = np.tanh(samples)
	print (y)
	log_y = calculate_log(y, mu, sigma)
	H = -np.average(log_y)
	print (H)

sanity_check()

def sample_entropy():
	mu_vector = np.linspace(-3, 3, num=50)
	sigma_vector = np.linspace(0.01, 5, num=50)

	normal_pair = {'train':[], 'validation':[]}
	entropy = {'train':[], 'validation':[]}
	#normal_pair = []
	#entropy = []
	for mu in mu_vector:
		for sigma in sigma_vector:
			pair = (mu, sigma)
			samples = np.random.normal(mu, sigma, 5000)
			y = np.tanh(samples)
			log_y = calculate_log(y, mu, sigma)
			H = -np.average(log_y)
			#normal_pair.append([mu, sigma])
			#entropy.append(H)
			if np.random.rand() <= 0.1:
				normal_pair['validation'].append([mu, sigma])
				entropy['validation'].append(H)
			else:
				normal_pair['train'].append([mu, sigma])
				entropy['train'].append(H)
	return normal_pair, entropy

def train_entropy_net():
	max_epochs = 20

	mu_sig_dic, entropy = sample_entropy()

	train_pair = torch.from_numpy(np.array(mu_sig_dic['train']))
	train_target = torch.from_numpy(np.array(entropy['train']))
	
	#train_pair = torch.from_numpy(np.array(mu_sig_dic))
	#train_target = torch.from_numpy(np.array(entropy))
	
	training_set = TensorDataset(train_pair, train_target)
	training_loader = DataLoader(training_set, batch_size=1800, shuffle=True)

	vali_pair = torch.from_numpy(np.array(mu_sig_dic['validation']))
	vali_target = torch.from_numpy(np.array(entropy['validation']))
	validation_set = TensorDataset(vali_pair, vali_target)
	validation_loader = DataLoader(validation_set, batch_size=1800, shuffle=True)

	entropy_net = nn.Sequential(
		nn.Linear(2, 32), 
		nn.ReLU(),
		nn.Linear(32, 32),
		nn.ReLU(),
		nn.Linear(32, 1))

	mse_fn = torch.nn.MSELoss()
	optimizer = torch.optim.SGD(entropy_net.parameters(), lr=0.005)

	plot_x = []
	plot_y = []

	for epoch in range(max_epochs):
		#print (epoch)
		train_loss = 0
		for i_batch, sample in enumerate(training_loader):
			mu_sig = torch.from_numpy(np.array(sample[0])).float()
			target = torch.from_numpy(np.array(sample[1])).float().view(-1,1)

			e_pred = entropy_net(mu_sig)
			loss = mse_fn(e_pred, target)
			entropy_net.zero_grad()
			optimizer.step()

		train_loss += loss
		plot_x.append(epoch)
		plot_y.append(train_loss.item()/len(mu_sig_dic['train']))
		#plot_y.append(train_loss.item()/len(mu_sig_dic))


	print ("Training is complete.")
	plt.plot(plot_x, plot_y)
	plt.xlabel('num_epoch')
	plt.ylabel('training loss')
	plt.show()

	sanity_mu = [0, 0.2, 0.5, 1, 2, 3]
	sanity_sigma = np.linspace(0.01, 5, num=50)

	#plot_mu = []
	#plot_H = []
	for mu in sanity_mu:
		plot_x = []
		plot_y = []
		for sigma in sanity_sigma:
			pair = torch.from_numpy(np.array([mu, sigma])).float()
			H = entropy_net(pair)
			plot_x.append(sigma)
			plot_y.append(H)
		plt.plot(plot_x, plot_y)
		plt.xlabel('sigma')
		plt.ylabel('entropy')
		plt.title('mu='+str(mu))
		plt.show()
	# entropy_net.eval()
	# vali_loss = 0
	# for i_batch, sample in enumerate(validation_loader):
	# 	mu_sig = torch.from_numpy(np.array(sample[0])).float()
	# 	target = torch.from_numpy(np.array(sample[1])).float()
	# 	e_pred = entropy_net(mu_sig)
	# 	loss = mse_fn(e_pred, target)
	# 	vali_loss += loss
	# print (vali_loss.item()/len(mu_sig_dic['validation']))

	torch.save(entropy_net.state_dict(), 'entropy_net.pth')

#train_entropy_net()



