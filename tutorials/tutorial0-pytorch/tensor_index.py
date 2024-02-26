import torch

batch_size = 10
features = 25
x = torch.rand((batch_size, features)) # (10, 25)


# Fancy indexing
x = torch.arange(10)
indices = [2, 5, 8]

# More advanced indexing
x = torch.arange(10)
print(x[(x < 2) | (x > 8)])
print(x[x.remainder(2) == 0]) # % 2 == 0
 
print(torch.tensor([0, 0, 1, 2, 2, 3, 4]).unique())
print(x.numel())