import torch
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


# tensor
x = torch.tensor([[1, 2, 3], [4, 5, 6], [7, 8, 9]], dtype=torch.float32, device=device, requires_grad=True)


# other common intialization methods
x1 = torch.empty(size=(3, 3))
x2 = torch.zeros((3, 3))
x3 = torch.rand((3, 3))
x4 = torch.randn((3, 3))
x5 = torch.ones((3, 3))
x6 = torch.eye(3, 3)

x7 = torch.arange(start=0, end=5, step=1)
x8 = torch.linspace(start=0.1, end=1, steps=10)

# tensor 
x = torch.arange(4)
print(x.bool())
print(x.short())
print(x.long())
print(x.half())
print(x.float()) # float32
print(x.double()) # float64

# numpy array to tensor
import numpy as np
np_array = np.zeros((5, 5))
tensor = torch.from_numpy(np_array)
np_array_back = tensor.numpy()
