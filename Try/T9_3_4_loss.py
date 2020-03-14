import math

import torch
import torch.nn as nn
from torch.autograd import Variable
from torch.autograd import Function
import torch.nn.functional as F
from Try.T9_3_5_bbox import bboxIOU, encodeBox

__all__ = ["EzDetectLoss"]


def buildbboxTarget(config, bboxOut, target):
    bboxMasks = torch.ByteTensor(bboxOut.size())
    bboxMasks.zero_()
    bboxTarget = torch.FloatTensor(bboxOut.size())
    batchSize = target.size()[0]
    for i in range(0, batchSize):
        num = int(target[i][0])
        for j in range(0, num):
            offset = j * 6
            cls = int(target[i][offset + 1])
            k = int(target[i][offset + 6])
            trueBox = [target[i][offset + 2],
                       target[i][offset + 3],
                       target[i][offset + 4],
                       target[i][offset + 5]]
            predBox = config.predBoxes[k]
            ebox = encodeBox(config, trueBox, predBox)
            bboxMasks[i, k, :] = 1
            bboxTarget[i, k, 0] = ebox[0]
            bboxTarget[i, k, 1] = ebox[1]
            bboxTarget[i, k, 2] = ebox[2]
            bboxTarget[i, k, 3] = ebox[3]
    if (config.gpu):
        bboxMasks = bboxMasks.cuda()
        bboxTarget = bboxTarget.cuda()
    return bboxMasks, bboxTarget


def buildConfTarget(config, confOut, target):
    batchSize = confOut.size()[0]
    boxNumber = confOut.size()[1]
    confTarget = torch.LongTensor(batchSize, boxNumber, config.classNumber)
    confMasks = torch.ByteTensor(confOut.size())
    confMasks.zero_()
    confScore = torch.nn.functional.log_softmax(Variable(confOut.view(-1, config.classNumber), requires_grad=False), dim=1)
    confScore = confScore.data.view(batchSize, boxNumber, config.classNumber)
    # positive samples
    pnum = 0
    for i in range(0, batchSize):
        num = int(target[i][0])
        for j in range(0, num):
            offset = j * 6
            k = int(target[i][offset + 6])
            cls = int(target[i][offset + 1])
            if cls > 0:
                confMasks[i, k, :] = 1
                confTarget[i, k, :] = cls
                confScore[i, k, :] = 0
                pnum = pnum + 1
            else:
                confScore[i, k, :] = 0
                '''
                cls = cls * -1
                confMasks[i, k, :] = 1
                confTarget[i, k, :] = cls
                confScore[i, k, :] = 0
                pnum = pnum + 1
                '''
    # negtive samples (background)
    confScore = confScore.view(-1, config.classNumber)
    confScore = confScore[:, 0].contiguous().view(-1)
    scoreValue, scoreIndex = torch.sort(confScore, 0, descending=False)
    for i in range(pnum * 3):
        b = scoreIndex[i] // boxNumber
        k = scoreIndex[i] % boxNumber
        if (confMasks[b, k, 0] > 0):
            break
        confMasks[b, k, :] = 1
        confTarget[b, k, :] = 0
    if (config.gpu):
        confMasks = confMasks.cuda()
        confTarget = confTarget.cuda()
    return confMasks, confTarget


# def encodeBox(config, box, predBox):
#     pcx = (predBox[0] + predBox[2]) / 2
#     pcy = (predBox[1] + predBox[3]) / 2
#     pw = (predBox[2] - predBox[0])
#     ph = (predBox[3] - predBox[1])
#     ecx = (box[0] + box[2]) / 2 - pcx
#     ecy = (box[1] + box[3]) / 2 - pcy
#     ecx = ecx / pw * 10
#     ecy = ecy / ph * 10
#     ew = (box[2] - box[0]) / pw
#     eh = (box[3] - box[1]) / ph
#     ew = math.log(ew) * 5
#     eh = math.log(eh) * 5
#     return [ecx, ecy, ew, eh]


# def decodeAllBox(config, allBox):
#     newBoxes = torch.FloatTensor(allBox.size())
#     batchSize = newBoxes.size()[0]
#     for k in range(len(config.predBoxes)):
#         predBox = config.predBoxes[k]
#         pcx = (predBox[0] + predBox[2]) / 2
#         pcy = (predBox[1] + predBox[3]) / 2
#         pw = (predBox[2] - predBox[0])
#         ph = (predBox[3] - predBox[1])
#         for i in range(batchSize):
#             box = allBox[i, k, :]
#             dcx = box[0] / 10 * pw + pcx
#             dcy = box[1] / 10 * ph + pcy
#             dw = math.exp(box[2] / 5) * pw
#             dh = math.exp(box[3] / 5) * ph
#             newBoxes[i, k, 0] = max(0, dcx - dw / 2)
#             newBoxes[i, k, 1] = max(0, dcy - dh / 2)
#             newBoxes[i, k, 2] = min(1, dcx + dw / 2)
#             newBoxes[i, k, 3] = min(1, dcy + dh / 2)
#     if config.gpu:
#         newBoxes = newBoxes.cuda()
#     return newBoxes


class EzDetectLoss(nn.Module):
    def __init__(self, config, pretrained=False):
        super(EzDetectLoss, self).__init__()
        self.config = config
        self.confLoss = nn.CrossEntropyLoss()
        self.bboxLoss = nn.SmoothL1Loss()

    def forward(self, confOut, bboxOut, target):
        batchSize = target.size()[0]
        # building loss of conf
        confMasks, confTarget = buildConfTarget(self.config, confOut.data, target)
        confSamples = confOut[confMasks.bool()].view(-1, self.config.classNumber)
        confTarget = confTarget[confMasks.bool()].view(-1, self.config.classNumber)
        confTarget = confTarget[:, 0].contiguous().view(-1)
        confTarget = Variable(confTarget, requires_grad=False)
        confLoss = self.confLoss(confSamples, confTarget)
        # building loss of bbox
        bboxMasks, bboxTarget = buildbboxTarget(self.config, bboxOut.data, target)
        bboxSamples = bboxOut[bboxMasks.bool()].view(-1, 4)
        bboxTarget = bboxTarget[bboxMasks.bool()].view(-1, 4)
        bboxTarget = Variable(bboxTarget)
        bboxLoss = self.bboxLoss(bboxSamples, bboxTarget)
        return confLoss, bboxLoss
