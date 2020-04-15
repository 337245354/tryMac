import torch.nn as nn

#判别网络的代码
class Discriminator(nn.Module):
    def __init__(self, ngpu):
        super(Discriminator, self).__init__()
        self.ngpu = ngpu
        self.main = nn.Sequential(
            #输入为一张宽高均为64，channel为nc的一张图片，得到宽高均为32，channel为ndf的一张图片 (ndf) * 32 * 32
            nn.Conv2d(nc, ndf, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            #经过第2次卷积操作后得到宽高均为16，channel为ndf*2的一张图片(ndf*2) * 16 * 16
            nn.Conv2d(ndf, ndf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 2), #使用大尺度的步长来代替采样（pooling），这样可以更好地学习降采样的方法
            nn.LeakyReLU(0.2, inplace=True),
            #经过第3次卷积操作后得到宽高均为8，channel为ndf*4的一张图片 (ndf*4) * 8 * 8
            nn.Conv2d(ndf * 2, ndf * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 4),
            nn.LeakyReLU(0.2, inplace=True),
            #经过第4次卷积操作后得到宽高均为4，channel为ndf*8的一张图片 (ndf*8) * 4 * 4
            nn.Conv2d(ndf * 4, ndf * 8, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 8),
            nn.LeakyReLU(0.2, inplace=True),
            #经过第5次卷积并过Sigmoid层，最终得到一个概率输出值
            nn.Conv2d(ndf * 8, 1, 4, 1, 0, bias=False),
            nn.Sigmoid() #最终通过Sigmoid激活函数输出该张图片是真实图片的概率
        )
    def forward(self, input):
        return self.main(input)
####将判别网络实例化####
#创建判别器
netD = Discriminator(ngpu).to(device)
#处理多GPU情况
if (device.type == 'cuda') and (ngpu > 1):
    netD = nn.DataParallel(netD, list(range(ngpu)))
#应用weights_init函数对随机初始化进行重置，改为服从mean=0, stdev=0.2的正态分布的初始化
netD.apply(weights_init)