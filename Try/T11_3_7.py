#####查看网络Loss的变化#####
plt.figure(figsize=(10,5))
plt.title("Generator and Discriminator Loss During Training")
plt.plot(G_losses,label="G") #画出产生网络Loss的变化
plt.plot(D_losses,label="D") #画出判别网络Loss的变化
plt.xlabel("iterations")
plt.ylabel("Loss")
plt.legend()
plt.show()


####对比真实图片和产生的假图片####
real_batch = next(iter(dataloader)) #从dataloader中取一个batch（64个）的图片
#画真实的图片
plt.figure(figsize=(15,15))
plt.subplot(1,2,1)
plt.axis("off")
plt.title("Real Images")
plt.imshow(np.transpose(vutils.make_grid(real_batch[0].to(device)[:64], padding=5, normalize=True).cpu(),(1,2,0)))
#画出产生网络最后一个迭代产生的图片
plt.subplot(1,2,2)
plt.axis("off")
plt.title("Fake Images")
plt.imshow(np.transpose(img_list[-1],(1,2,0)))
plt.show()