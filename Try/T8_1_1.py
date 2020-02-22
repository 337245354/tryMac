import torch.nn as nn
import torch.nn.functional as F

# classtorch.nn.Conv2d(in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1, bias=True)
# in_channels (int) ：输入图片的channel
# out_channels (int) ：输出图片（特征层）的channel
# kernel_size (int or tuple) ：kernel的大小
# stride (int or tuple, optional) ：卷积的步长，默认为1
# padding (int or tuple, optional) ：四周pad的大小，默认为0
# dilation (int or tuple, optional) ：kernel元素间的距离，默认为1（dilation翻译为扩张，有时候也称为“空洞”，有专门的文章研究dilation convolution）
# groups (int, optional) ：将原始输入channel划分成的组数，默认为1（初级读者暂时可以不必细究其用处）
# bias (bool, optional) ：如果是Ture，则输出的bias可学，默认为True

