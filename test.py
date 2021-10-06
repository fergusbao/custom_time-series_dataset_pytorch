import torch
import utils

data = torch.randn(10, 4)*100
print(data)

print('-'*20)
print('mean')
mean = data.mean(dim=0)
print(mean)

print('std')
std = data.std(dim=0)
print(std)

mean, std = utils.calculate_mean_std(data)

print(mean, std)


data -= mean
data /= std

print(data)

data *= std
data += mean

print(data)

