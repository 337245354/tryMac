from torch.autograd import Variable  # 导入Variable
from torch.autograd import Variable
import torch

x_tensor = torch.randn(10, 5)  # 从标准正态分布中返回多个样本值
# 将Tensor变成Variable
x = Variable(x_tensor, requires_grad=True)
# 默认Variable是不需要求梯度的，所以用这个方式申明需要对其进行求梯度的操作
print(x.data)
print(x.grad)
print(x.grad_fn)
