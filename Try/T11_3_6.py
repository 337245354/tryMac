#训练过程：主循环
img_list = []
G_losses = []
D_losses = []
iters = 0
print("Starting Training Loop...")
for epoch in range(num_epochs): #训练集迭代的次数
    for i, data in enumerate(dataloader, 0): #循环每个dataloader中的batch
        ############################
        # 1)更新判别网络：最大化log(D(x)) + log(1 - D(G(z)))
        ###########################
        ##用全部都是真图片的batch进行训练
        netD.zero_grad()
        #格式化batch
        real_cpu = data[0].to(device)
        b_size = real_cpu.size(0)
        label = torch.full((b_size,), real_label, device=device)
        #将带有正样本的batch输入到判别网络中进行前向计算，得到的结果将放到变量output中
        output = netD(real_cpu).view(-1)
        #计算Loss
        errD_real = criterion(output, label)
        #计算梯度
        errD_real.backward()
        D_x = output.mean().item()
        ##用全部都是假图片的batch进行训练
        #产生网络的输入向量
        noise = torch.randn(b_size, nz, 1, 1, device=device)
        #通过产生网络生成假的样本图片
        fake = netG(noise)
        label.fill_(fake_label)
        #将生成的全部假图片输入到判别网络中进行前向计算，将得到的结果放到变量output中
        output = netD(fake.detach()).view(-1)
        #在假图片batch中计算上述判别网络的Loss
        errD_fake = criterion(output, label)
        #计算该batch的梯度
        errD_fake.backward()
        D_G_z1 = output.mean().item()
        #将真图片与假图片的误差加和
        errD = errD_real + errD_fake
        #更新判别网络D
        optimizerD.step()
        ############################
        # 2)更新产生网络：最大化log(D(G(z)))
        ###########################
        netG.zero_grad()
        label.fill_(real_label) #产生网络的标签是真实的图片
        #由于刚刚更新了判别网络，这里让假数据再过一遍判别网络，用来计算产生网络的Loss并回传
        output = netD(fake).view(-1)
        errG = criterion(output, label)
        errG.backward()
        D_G_z2 = output.mean().item()
        #更新产生网络G
        optimizerG.step()
        #打印训练状态
        if i % 50 == 0:
            print('[%d/%d][%d/%d]\tLoss_D: %.4f\tLoss_G: %.4f\tD(x): %.4f\tD(G(z)): %.4f / %.4f'
                % (epoch, num_epochs, i, len(dataloader),
                    errD.item(), errG.item(), D_x, D_G_z1, D_G_z2))
        #保存Loss，用于后续画图
        G_losses.append(errG.item())
        D_losses.append(errD.item())
        #保留产生网络生成的图片，后续用来查看生成的图片效果
        if (iters % 500 == 0) or ((epoch == num_epochs-1) and (i == len(dataloader)-1)):
            with torch.no_grad():
                fake = netG(fixed_noise).detach().cpu()
            img_list.append(vutils.make_grid(fake, padding=2, normalize=True))
        iters += 1