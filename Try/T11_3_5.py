import torch.nn as nn

#初始化二元交叉熵损失函数
criterion = nn.BCELoss()
#创建一个batch大小的向量z，即产生网络的输入数据
fixed_noise = torch.randn(64, nz, 1, 1, device=device)
#定义训练过程的真图片/假图片的标签
real_label = 1
fake_label = 0
#为产生网络和判别网络设置Adam优化器
optimizerD = optim.Adam(netD.parameters(), lr=lr, betas=(beta1, 0.999))
optimizerG = optim.Adam(netG.parameters(), lr=lr, betas=(beta1, 0.999))
1