import torch

x = torch.rand(size=(2,2))
y = torch.rand(size=(2,3))

print(torch.equal(x, y))
print(torch.eq(x, y))