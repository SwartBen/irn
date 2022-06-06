import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from misc import torchutils
import torch.utils.model_zoo as model_zoo
from net import resnest269

class FixedBatchNorm(nn.BatchNorm2d):
    def forward(self, x):
        return F.batch_norm(x, self.running_mean, self.running_var, self.weight, self.bias, training=False,
                            eps=self.eps)

def group_norm(features):
    return nn.GroupNorm(4, features)

class Net(nn.Module):

    def __init__(self):
        super().__init__()

        self.resnest269 = resnest269.resnest50(pretrained=True)
        #self.resnest269 = resnest269.resnest269(pretrained=True, strides=(2, 2, 2, 1))

        self.norm_fn = FixedBatchNorm

        self.stage1 = nn.Sequential(self.resnest269.conv1,
                                    self.resnest269.bn1,
                                    self.resnest269.relu,
                                    self.resnest269.maxpool, self.resnest269.layer1)
        self.stage2 = nn.Sequential(self.resnest269.layer2)
        self.stage3 = nn.Sequential(self.resnest269.layer3)
        self.stage4 = nn.Sequential(self.resnest269.layer4)

        self.classifier = nn.Conv2d(2048, 20, 1, bias=False)

        self.backbone = nn.ModuleList([self.stage1, self.stage2, self.stage3, self.stage4])
        self.newly_added = nn.ModuleList([self.classifier])

    def forward(self, x):

        x = self.stage1(x)
        x = self.stage2(x).detach()

        x = self.stage3(x)
        x = self.stage4(x)

        x = torchutils.gap2d(x, keepdims=True)
        x = self.classifier(x)
        x = x.view(-1, 20)

        return x

    def train(self, mode=True):
        for p in self.resnest269.conv1.parameters():
            p.requires_grad = False
        for p in self.resnest269.bn1.parameters():
            p.requires_grad = False

    def trainable_parameters(self):
        return (list(self.backbone.parameters()), list(self.newly_added.parameters()))

class CAM(Net):
    def __init__(self):
        super(CAM, self).__init__()

    def forward(self, x):
        x = self.stage1(x)
        x = self.stage2(x)
        x = self.stage3(x)
        x = self.stage4(x)

        x = F.conv2d(x, self.classifier.weight)
        x = F.relu(x)
        x = x[0] + x[1].flip(-1)

        return x