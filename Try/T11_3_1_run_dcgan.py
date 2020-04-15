# -*- coding: utf-8 -*-
from __future__ import print_function
# %matplotlib inline
import argparse
import os
import random
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim as optim
import torch.utils.data
import torchvision.datasets as dset
import torchvision.transforms as transforms
import torchvision.utils as vutils
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from IPython.display import HTML

# 设置随机种子
manualSeed = 999
# manualSeed = random.randint(1, 10000) #想获取新结果时使用
print("Random Seed: ", manualSeed)
random.seed(manualSeed)
torch.manual_seed(manualSeed)
dataroot = "data/celeba"  # 数据集的根目录
workers = 2  # 载入数据的线程数量
batch_size = 128  # 训练过程batch的大小
image_size = 64  # 训练图片的大小，所有图片均需要缩放到这个尺寸
nc = 3  # 通道数量，通常彩色图就是RGB三个值
nz = 100  # 产生网络输入向量的大小
ngf = 64  # 产生网络特征层的大小
ndf = 64  # 判别网络特征层的大小
num_epochs = 5  # 训练数据集迭代次数
lr = 0.0002  # 学习率
beta1 = 0.5  # Adam最优化方法中的超参beta1
ngpu = 1  # 可用的GPU数量（0为CPU模式）
# 创建数据集（包含各种初始化）
dataset = dset.ImageFolder(root=dataroot,
                           transform=transforms.Compose([
                               transforms.Resize(image_size),
                               transforms.CenterCrop(image_size),
                               transforms.ToTensor(),
                               transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                           ]))
# 创建数据载入器DataLoader
dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size,
                                         shuffle=True, num_workers=workers)
# 设置训练需要的处理器
device = torch.device("cuda:0" if (torch.cuda.is_available() and ngpu > 0) else "cpu")
####展示一些训练数据####
real_batch = next(iter(dataloader))
plt.figure(figsize=(8, 8))
plt.axis("off")
plt.title("Training Images")
plt.imshow(np.transpose(vutils.make_grid(real_batch[0].to(device)[:64], padding=2, normalize=True).cpu(), (1, 2, 0)))


