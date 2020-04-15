import torch
import torch.nn as nn


#构建产生网络的代码
class Generator(nn.Module):
    def __init__(self, ngpu):
        super(Generator, self).__init__()
        self.ngpu = ngpu
        self.main = nn.Sequential(
            #输入向量z，通过第一个反卷积
            #将100的向量z输入，输出channel设置为(ngf*8)，经过如下操作
            # class torch.nn.ConvTranspose2d(in_channels, out_channels, kernel_size, stride=1, padding=0, output_padding=0, groups=1, bias=True, dilation=1)
            #后得到(ngf*8) * 4 * 4，即长宽为4，channel为ngf*8的特征层
            nn.ConvTranspose2d( nz, ngf * 8, 4, 1, 0, bias=False),
                #这里的Conv-Transpose2d类似于deconv，前面第10章已介绍过其原理
            nn.BatchNorm2d(ngf * 8),
            nn.ReLU(True),
            #继续对特征层进行反卷积操作，得到长宽为8，channel为ngf*4的特征层 (ngf*4) * 8 * 8
            nn.ConvTranspose2d(ngf * 8, ngf * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf * 4),
            nn.ReLU(True),
            #继续对特征层进行反卷积操作，得到长宽为16，channel为ngf*2的特征层 (ngf*2) * 16 * 16
            nn.ConvTranspose2d( ngf * 4, ngf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf * 2),
            nn.ReLU(True),
            #继续对特征层进行反卷积操作，得到长宽为32，channel为ngf的特征层 (ngf) * 32 * 32
            nn.ConvTranspose2d( ngf * 2, ngf, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf),
            nn.ReLU(True),
            #继续对特征层进行反卷积操作，得到长宽为64，channel为nc的特征层 (nc) * 64 * 64
            nn.ConvTranspose2d( ngf, nc, 4, 2, 1, bias=False),
            nn.Tanh()
        )
    def forward(self, input):
        return self.main(input)
################################################################
####将产生网络实例化####
#创建生成器
netG = Generator(ngpu).to(device)
#处理多GPU情况
if (device.type == 'cuda') and (ngpu > 1):
    netG = nn.DataParallel(netG, list(range(ngpu)))
#应用weights_init函数对随机初始化进行重置，改为服从mean=0, stdev=0.2的正态分布的初始化
netG.apply(weights_init)