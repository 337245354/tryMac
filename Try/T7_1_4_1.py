import torch
from torch.autograd import Variable
import matplotlib.pyplot as plt


torch.nn.MSELoss(size_average=None, reduce=None, reduction='mean')
torch.nn.CrossEntropyLoss(weight=None, size_average=None, ignore_index=-100, reduce=None, reduction='mean')

loss = torch.nn.CrossEntropyLoss()
input = torch.randn(3, 5, requires_grad=True)
target = torch.empty(3, dtype=torch.long).random_(5)
output = loss(input, target)
output.backward()
