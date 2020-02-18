import torch
import numpy as np

if torch.cuda.is_available():
    x = 1
    y = 2
    x = x.cuda()
    y = y.cuda()
    print(x+y)

# np_data = np.arange(8).reshape((2,4)) #定义一个numpy的二维数组
# print(np_data)
# torch_data = torch.from_numpy(np_data)
# print(torch_data)
# np_data2 = torch_data.numpy() #转回numpy
# print(np_data2)

import torch
import numpy as np
np_data = np.array([[1,2],[3,5]])
torch_data = torch.from_numpy(np_data)
print(np_data)
print(np_data.dot(np_data))
print(torch_data.mm(torch_data))
