import torch
import torchvision
import torchvision.transforms as transforms
import math
import torch
import torch.nn as nn
import torch.optim as optim
import os

transform = transforms.Compose(
    [transforms.ToTensor(),  # 将PILImage转换为张量
     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]  # 将[0, 1]归一化到[-1, 1]
)
trainset = torchvision.datasets.CIFAR10(root='../ml/book/classifier_cifar10/data',
                                        # root表示的是Cifar10的数据存放目录，使用torchvision可直接下载Cifar10数据集，也可直接在https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz这里下载（链接来自Cifar10官网）
                                        train=True,
                                        download=False,
                                        transform=transform  # 按照上面定义的transform格式转换下载的数据
                                        )
trainloader = torch.utils.data.DataLoader(trainset,
                                          batch_size=4,  # 每个batch载入的图片数量，默认为1
                                          shuffle=True,
                                          num_workers=2  # 载入训练数据所需的子任务数
                                          )
testset = torchvision.datasets.CIFAR10(root='../ml/book/classifier_cifar10/data',
                                       train=False,
                                       download=False,
                                       transform=transform)
testloader = torch.utils.data.DataLoader(testset,
                                         batch_size=4,
                                         shuffle=False,
                                         num_workers=2)
cifar10_classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

cfg = {'VGG16': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M']}

class VGG(nn.Module):
    def __init__(self, net_name):
        super(VGG, self).__init__()
        # 构建网络的卷积层和池化层，最终输出命名features，原因是通常认为经过这些操作的输出为包含图像空间信息的特征层
        self.features = self._make_layers(cfg[net_name])
        # 构建卷积层之后的全连接层以及分类器
        self.classifier = nn.Sequential(
            nn.Dropout(),
            nn.Linear(512, 512),  # fc1
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(512, 512),  # fc2
            nn.ReLU(True),
            nn.Linear(512, 10),  # fc3，最终Cifar10的输出是10类
        )
        # 初始化权重
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
                m.bias.data.zero_()

    def forward(self, x):
        x = self.features(x)  # 前向传播的时候先经过卷积层和池化层
        x = x.view(x.size(0), -1)
        x = self.classifier(x)  # 再将features（得到网络输出的特征层）的结果拼接到分类器上
        return x

    def _make_layers(self, cfg):
        layers = []
        in_channels = 3
        for v in cfg:
            if v == 'M':
                layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
            else:
                # conv2d = nn.Conv2d(in_channels, v, kernel_size=3, padding=1)
                # layers += [conv2d, nn.ReLU(inplace=True)]
                layers += [nn.Conv2d(in_channels, v, kernel_size=3, padding=1),
                           nn.BatchNorm2d(v),
                           nn.ReLU(inplace=True)]
                in_channels = v
        return nn.Sequential(*layers)


net = VGG('VGG16')

# checkpoint = torch.load('./checkpoint/cifar10_epoch_2.ckpt')  # 载入现有模型 ,目前只训练到cifar10_epoch_2.ckpt，正确率60%
# net.load_state_dict(checkpoint['net'])
# 正式测试
correct = 0
total = 0
with torch.no_grad():
    for data in testloader:
        images, labels = data
        outputs = net(images)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()  # 当标记的label种类和预测的种类一致时认为正确，并计数
        print('Accuracy of the network on the 10000 test images: %d %%' % (100 * correct / total))


