from __future__ import print_function
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim  # 引入优化方法
# 下面两个引用用于载入/展示图片
from PIL import Image
import matplotlib.pyplot as plt
import torchvision.transforms as transforms  # 将PIL图片格式转换为向量（tensor）格式
import torchvision.models as models  # 训练/载入预训练模型
import copy  # 用于复制模型

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# 如果有cuda则使用GPU，否则使用CPU
# 图片预处理：原始PIL图片像素值的范围为0～255，在向量处理过程中需要先将这些值归一化到0～1。另外，图片需要缩放到相同的维度。
imsize = 512 if torch.cuda.is_available() else 128  # 如果没有GPU则使用小图
loader = transforms.Compose([
    transforms.Resize(imsize),  # 将图片进行缩放，需要缩放到相同的尺度再输入到神经网络
    transforms.ToTensor()])  # 将图片转为PyTorch可接受的向量（tensor）格式


def image_loader(image_name):
    image = Image.open(image_name)
    image = loader(image).unsqueeze(0)
    return image.to(device, torch.float)


style_img = image_loader("images/neural-style/style_sketch.jpg")  # 载入1张风格图片
content_img = image_loader("images/neural-style/content_person.jpg")  # 载入1张内容图片
assert style_img.size() == content_img.size(), \
    "we need to import style and content images of the same size"

unloader = transforms.ToPILImage()  # 将PyTorch中tensor格式的数据转成PIL格式的图片用于展示
plt.ion()


def imshow(tensor, title=None):  # 定义一个专门用于展示图片的函数
    image = tensor.cpu().clone()  # 为了不改变tensor的内容这里先备份一下
    image = image.squeeze(0)  # 去掉这里面没有用的batch这个维度
    image = unloader(image)
    plt.imshow(image)
    if title is not None:
        plt.title(title)
    plt.pause(0.001)  # pause a bit so that plots are updated


plt.figure()
imshow(style_img, title='Style Image')
plt.figure()
imshow(content_img, title='Content Image')


# 定义内容学习的损失函数
class ContentLoss(nn.Module):
    def __init__(self, target, ):
        super(ContentLoss, self).__init__()
        self.target = target.detach()

    def forward(self, input):
        self.loss = F.mse_loss(input, self.target)  # 输入内容图片和目标图片的均方差
        return input


# 定义风格学习的损失函数前需要先定义格雷姆矩阵的计算方法
def gram_matrix(input):
    a, b, c, d = input.size()  # a为batch中图片的个数（1）
    # b为feature map的个数
    # (c,d)feature map的维度(N=c*d)
    features = input.view(a * b, c * d)  # resise F_XL into \hat F_XL
    G = torch.mm(features, features.t())  # 计算得出格雷姆矩阵（内积）
    return G.div(a * b * c * d)  # 对格雷姆矩阵的数值进行归一化操作


# 定义风格学习的损失函数
class StyleLoss(nn.Module):
    def __init__(self, target_feature):
        super(StyleLoss, self).__init__()
        self.target = gram_matrix(target_feature).detach()

    def forward(self, input):
        G = gram_matrix(input)
        self.loss = F.mse_loss(G, self.target)  # 艺术图片的格雷姆矩阵与目标图片的格雷姆矩阵的均方差
        return input


# 载入用ImageNet预训练好的VGG19模型，并只使用features模块
# 注：PyTorch将VGG模型分为2个模块，features模块和classifier模块，其中features模块包含卷积和池化层，classifier模块包含全连接和分类层。
# 一些层在训练和预测（评估）时网络的行为（参数）是不同的，注意这里需要使用eval()
cnn = models.vgg19(pretrained=True).features.to(device).eval()
# VGG网络是用均值[0.485, 0.456, 0.406]，方差[0.229, 0.224, 0.225]对图片进行归一化之后再进行训练的，所以这里也需要对图片进行归一化操作
cnn_normalization_mean = torch.tensor([0.485, 0.456, 0.406]).to(device)
cnn_normalization_std = torch.tensor([0.229, 0.224, 0.225]).to(device)


class Normalization(nn.Module):
    def __init__(self, mean, std):
        super(Normalization, self).__init__()
        # 下面两个操作是将数据转换成[BatchSize x Channel x Hight x Weight]格式
        self.mean = torch.tensor(mean).view(-1, 1, 1)
        self.std = torch.tensor(std).view(-1, 1, 1)

    def forward(self, img):
        # normalize img
        return (img - self.mean) / self.std


# vgg19.features中包含(Conv2d, ReLU, MaxPool2d, Conv2d, ReLU…)等，为了实现图片风格转换，需要将内容损失层（content Loss）和风格损失层（style Loss）加到vgg19.features后面
content_layers_default = ['conv_4']
style_layers_default = ['conv_1', 'conv_2', 'conv_3', 'conv_4', 'conv_5']


def get_style_model_and_losses(cnn, normalization_mean, normalization_std,
                               style_img, content_img,
                               content_layers=content_layers_default,
                               style_layers=style_layers_default):
    cnn = copy.deepcopy(cnn)
    normalization = Normalization(normalization_mean, normalization_std).to(device)  # 归一化模块
    content_losses = []
    style_losses = []
    model = nn.Sequential(normalization)  # 可以设置一个新的nn.Sequential，顺序地激活
    i = 0  # 每看到一个卷积便加1
    for layer in cnn.children():  # 遍历当前CNN结构
        # 判断当前遍历的是CNN中的卷积层、ReLU层、池化层还是BatchNorm层
        if isinstance(layer, nn.Conv2d):
            i += 1
            name = 'conv_{}'.format(i)
        elif isinstance(layer, nn.ReLU):
            name = 'relu_{}'.format(i)
            layer = nn.ReLU(inplace=False)  # 由于实验过程中发现in-place在Conten-tLoss和StyleLoss上的表现不好，因此置为False
        elif isinstance(layer, nn.MaxPool2d):
            name = 'pool_{}'.format(i)
        elif isinstance(layer, nn.BatchNorm2d):
            name = 'bn_{}'.format(i)
        else:
            raise RuntimeError('Unrecognized layer: {}'.format(layer.__class__.__name__))
        model.add_module(name, layer)
        if name in content_layers:  # 向网络中加入content Loss
            target = model(content_img).detach()
            content_loss = ContentLoss(target)
            model.add_module("content_loss_{}".format(i), content_loss)
            content_losses.append(content_loss)
        if name in style_layers:  # 向网络中加入style Loss
            target_feature = model(style_img).detach()
            style_loss = StyleLoss(target_feature)
            model.add_module("style_loss_{}".format(i), style_loss)
            style_losses.append(style_loss)
    # now we trim off the layers after the last content and style losses
    for i in range(len(model) - 1, -1, -1):
        if isinstance(model[i], ContentLoss) or isinstance(model[i], StyleLoss):
            break
    model = model[:(i + 1)]
    return model, style_losses, content_losses


# 这里再确认一下输入的内容图片:
input_img = content_img.clone()
# 如果想测试待产生的白噪声图片则使用下面这行语句
# input_img = torch.randn(content_img.data.size(), device=device)
plt.figure()
imshow(input_img, title='Input Image')  # 画图


def get_input_optimizer(input_img):
    # 使用LBFGS方法进行梯度下降（不是常用的随机梯度下降，但不论是LBFGS还是随机梯度下降都是在空间中寻找最优解的优化方法）
    optimizer = optim.LBFGS([input_img.requires_grad_()])
    return optimizer


# 定义整个风格化的学习过程
def run_style_transfer(cnn, normalization_mean, normalization_std,
                       content_img, style_img, input_img, num_steps=300,
                       style_weight=1000000, content_weight=1):
    """Run the style transfer."""
    print('Building the style transfer model..')
    model, style_losses, content_losses = get_style_model_and_losses(cnn,
                                                                     normalization_mean, normalization_std, style_img,
                                                                     content_img)
    optimizer = get_input_optimizer(input_img)
    print('Optimizing..')
    run = [0]
    while run[0] <= num_steps:
        def closure():  # 用来评估并返回当前Loss的函数
            input_img.data.clamp_(0, 1)  # 将更新后的输入修正到0～1
            optimizer.zero_grad()
            model(input_img)
            style_score = 0
            content_score = 0
            for sl in style_losses:
                style_score += sl.loss
            for cl in content_losses:
                content_score += cl.loss
            style_score *= style_weight
            content_score *= content_weight
            loss = style_score + content_score
            loss.backward()
            run[0] += 1
            if run[0] % 50 == 0:
                print("run {}:".format(run))
                print('Style Loss : {:4f} Content Loss: {:4f}'.format(
                    style_score.item(), content_score.item()))
                print()
            return style_score + content_score

        optimizer.step(closure)
    input_img.data.clamp_(0, 1)  # 最后一次修正
    return input_img


# 最终运行算法的一行代码
output = run_style_transfer(cnn, cnn_normalization_mean, cnn_normalization_std,
                            content_img, style_img, input_img)

# 展示风格化后的图片的代码如下：
plt.figure()
imshow(output, title='Output Image')  # 画出风格化图片
plt.ioff()
plt.show()
