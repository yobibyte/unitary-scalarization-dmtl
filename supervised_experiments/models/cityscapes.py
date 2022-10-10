import torch, sys
import torch.nn as nn
import torch.nn.functional as F
from supervised_experiments.models.resnet_dilated import ResnetDilated

class BaseModel(nn.Module):
    def __init__(self, task_num, weighting=None, random_distribution=None):
        super(BaseModel, self).__init__()

        self.task_num = task_num
        self.weighting = weighting
        self.random_distribution = random_distribution

        self.rep_detach = False

        self.loss_weight_init = None

        if self.rep_detach:
            self.rep = [0] * self.task_num
            self.rep_i = [0] * self.task_num
        if isinstance(self.loss_weight_init, float):
            self.loss_scale = nn.Parameter(torch.FloatTensor([self.loss_weight_init] * self.task_num))

        if self.weighting == 'RLW' and self.random_distribution == 'random_normal':
            self.random_normal_mean, self.random_normal_std = torch.rand(self.task_num), torch.rand(self.task_num)
        else:
            self.random_normal_mean, self.random_normal_std = None, None

    def forward(self):
        pass

class DeepLabHead(nn.Sequential):
    def __init__(self, in_channels, num_classes, log_softmax_out=True, img_size=[128,256], no_dropout=False):
        super(DeepLabHead, self).__init__(
            ASPP(in_channels, [12, 24, 36], no_dropout=no_dropout),
            nn.Conv2d(256, 256, 3, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.Conv2d(256, num_classes, 1)
        )
        self.log_softmax_out = log_softmax_out
        self.img_size = img_size

    def forward(self, x, mask):
        rep = super(DeepLabHead, self).forward(x)
        rep = F.interpolate(rep, self.img_size, mode='bilinear', align_corners=True)
        if self.log_softmax_out:
            return F.log_softmax(rep, dim=1), None
        else:
            return rep, None


class ASPPConv(nn.Sequential):
    def __init__(self, in_channels, out_channels, dilation):
        modules = [
            nn.Conv2d(in_channels, out_channels, 3, padding=dilation, dilation=dilation, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU()
        ]
        super(ASPPConv, self).__init__(*modules)


class ASPPPooling(nn.Sequential):
    def __init__(self, in_channels, out_channels):
        super(ASPPPooling, self).__init__(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channels, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU())

    def forward(self, x):
        size = x.shape[-2:]
        x = super(ASPPPooling, self).forward(x)
        return F.interpolate(x, size=size, mode='bilinear', align_corners=False)


class ASPP(nn.Module):
    def __init__(self, in_channels, atrous_rates, no_dropout=False):
        super(ASPP, self).__init__()
        out_channels = 256
        modules = []
        modules.append(nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU()))

        rate1, rate2, rate3 = tuple(atrous_rates)
        modules.append(ASPPConv(in_channels, out_channels, rate1))
        modules.append(ASPPConv(in_channels, out_channels, rate2))
        modules.append(ASPPConv(in_channels, out_channels, rate3))
        modules.append(ASPPPooling(in_channels, out_channels))

        self.convs = nn.ModuleList(modules)

        layers = [
            nn.Conv2d(5 * out_channels, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU()
        ]
        if not no_dropout:
            layers.append(nn.Dropout(0.5))
        self.project = nn.Sequential(*layers)

    def forward(self, x):
        res = []
        for conv in self.convs:
            res.append(conv(x))
        res = torch.cat(res, dim=1)
        return self.project(res)

