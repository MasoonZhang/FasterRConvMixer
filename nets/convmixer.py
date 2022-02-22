import torch
import torch.nn as nn

import torch.nn.functional as F


class GELU(nn.Module):
    def forward(self, input):
        return F.gelu(input)

class Residual(nn.Module):
    def __init__(self, fn):
        super().__init__()
        self.fn = fn

    def forward(self, x):
        return self.fn(x) + x


def ConvMixer(dim, depth, kernel_size=8, patch_size=1, n_classes=10):
    return nn.Sequential(
        nn.Conv2d(3, dim, kernel_size=patch_size, stride=patch_size),
        GELU(),
        nn.BatchNorm2d(dim),
        *[nn.Sequential(
                Residual(nn.Sequential(
                    nn.Conv2d(dim, dim, kernel_size, groups=dim, padding=kernel_size//2),
                    GELU(),
                    nn.BatchNorm2d(dim)
                )),
                nn.Conv2d(dim, dim, kernel_size=1),
                GELU(),
                nn.BatchNorm2d(dim)
        ) for i in range(depth)],
        nn.AdaptiveAvgPool2d((1,1)),
        nn.Flatten(),
        nn.Linear(dim, n_classes)
    )


def convmixer(pretrained = False):
    model = ConvMixer(1024, 20, kernel_size=9, patch_size=14, n_classes=1000)
    if pretrained:
            model.load_state_dict(torch.load('./model_data/voc_weights_convmixer.pth'))
    # ----------------------------------------------------------------------------#
    #   获取特征提取部分，从conv1到model.layer3，最终获得一个38,38,1024的特征层
    # ----------------------------------------------------------------------------#
    #features = list([model.conv1, model.bn1, model.gelu, model.maxpool, model.layer1, model.layer2, model.layer3])
    # ----------------------------------------------------------------------------#
    #   获取分类部分，从model.layer4到model.avgpool
    # ----------------------------------------------------------------------------#
    # classifier = list([model.layer4, model.avgpool])
    #
    features = model[:20]
    classifier = nn.Sequential(*list(model.children())[20:-2])
    return features, classifier