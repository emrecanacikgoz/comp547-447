import torch

x = torch.tensor([1, 2, 3])
y = torch.tensor([9, 8, 7])

# addition
z1 = torch.add(x, y)
z2 = x + y

# subtraction
z3 = x - y

# division
z4 = torch.true_divide(x, y) # element wise division

# exponential
z5 = x.pow(2) # element wise exponentiation
z6 = x ** 2

# simple comparison
z7 = x > 0
z8 = x < 0

# matrix multiplication
x1 = torch.rand((2, 5))
x2 = torch.rand((5, 3))
x3 = torch.mm(x1, x2) # @

# matrix elemwise
z = x * y
print(f"x3: {x3}")
print(f"z: {z}")

# dot product
z = torch.dot(x, y)

# batch matrix multiplication
batch = 32
n = 10
m = 20
p = 30

tensor1 = torch.rand((batch, n, m)) # (batch, n, m)
tensor2 = torch.rand((batch, m, p)) # (batch, m, p)
z = torch.bmm(tensor1, tensor2) # (batch, n, p)

# example of broadcasting
x1 = torch.rand((5, 5))
x2 = torch.rand((1, 5))

z = x1 - x2

# other useful tensor operations
sum_x = torch.sum(x, dim=0)
values, indices = torch.max(x, dim=0)
values, indices = torch.min(x, dim=0)
abs_x = torch.abs(x)
z = torch.argmax(x, dim=0)
z = torch.argmin(x, dim=0)

z = torch.eq(x, y)

sorted_y, indices = torch.sort(y, dim=0, descending=False)

z = torch.clamp(x, min=0, max=10)