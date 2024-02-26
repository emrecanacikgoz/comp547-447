import torch

x = torch.arange(9)

x = x.view(3, 3)
x = x.reshape(3, 3)

# concat
x1 = torch.rand((2, 5))
x2 = torch.rand((2, 5))
print(x1.shape)
print(torch.cat((x1, x2), dim=0).shape)
print(torch.cat((x1, x2), dim=1).shape)

# flatten
z = x1.view(-1) # use in fc-layer if you have NbyN matrix

# permute
z = x.permute(0, 2, 1)

# squeeze/unsqueeze
x = torch.arange(10).unsqueeze(0).unsqueeze(1) # 10 => 1x10 => 1x1x10
