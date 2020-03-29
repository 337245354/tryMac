import sys
from PIL import Image, ImageDraw
import torch
from torch.autograd import Variable
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from Try.T9_3_3_model import EzDetectConfig
from Try.T9_3_3_model import EzDetectNet
from Try.T9_3_5_bbox import decodeAllBox, doNMS
import numpy as np

ezConfig = EzDetectConfig()
ezConfig.batchSize = 1
mymodel = EzDetectNet(ezConfig, True)
modelFile = "./model/model_2_4.42.pth"
mymodel.load_state_dict(torch.load(modelFile))
print("finish load model")
normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
transformer = transforms.Compose([transforms.ToTensor(), normalize])
imgFile = "../ml/VOCdevkit/VOC2007/JPEGImages/000363.jpg"
img = Image.open(imgFile)
originImage = img
img = img.resize((ezConfig.targetWidth, ezConfig.targetHeight), Image.BILINEAR)
img = transformer(img)
img = img * 256
img = img.view(1, 3, ezConfig.targetHeight, ezConfig.targetWidth)
print("finish preprocess image")
if ezConfig.gpu == True:  # 使用gpu
    img = img.cuda()
    mymodel.cuda()
classOut, bboxOut = mymodel(Variable(img))
bboxOut = bboxOut.float()
bboxOut = decodeAllBox(ezConfig, bboxOut.data)
classScore = torch.nn.Softmax(dim=1)(classOut[0])
bestBox = doNMS(ezConfig, classScore.data.float(), bboxOut[0], 0.15)
draw = ImageDraw.Draw(originImage)
imgWidth, imgHeight = originImage.size
for b in bestBox:
    draw.rectangle([b[0] * imgWidth, b[1] * imgHeight,
                    b[2] * imgWidth, b[3] * imgHeight])
del draw
print("finish draw boxes")
originImage.save("./Results/1.jpg")
print("finish all!")
