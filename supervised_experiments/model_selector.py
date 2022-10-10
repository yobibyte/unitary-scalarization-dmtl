# Code adapted from https://github.com/isl-org/MultiObjectiveOptimization/blob/master/multi_task/

from supervised_experiments.models.multi_lenet import MultiLeNetO, MultiLeNetR
from supervised_experiments.models.multi_faces_resnet import ResNet, FaceAttributeDecoder, BasicBlock
from supervised_experiments.models.resnet_dilated import ResnetDilated
from supervised_experiments.models.cityscapes import DeepLabHead
from supervised_experiments.models import resnet_cityscapes
import torch.nn as nn


def get_model(dataset, tasks, device, parallel=False, add_dropout=False, no_dropout=False):
    data = dataset
    if 'mnist' in data:
        model = {}
        model['rep'] = MultiLeNetR(no_dropout=no_dropout)
        if parallel:
            model['rep'] = nn.DataParallel(model['rep'])
        model['rep'].to(device)
        if 'L' in tasks:
            model['L'] = MultiLeNetO(no_dropout=no_dropout)
            if parallel:
                model['L'] = nn.DataParallel(model['L'])
            model['L'].to(device)
        if 'R' in tasks:
            model['R'] = MultiLeNetO(no_dropout=no_dropout)
            if parallel:
                model['R'] = nn.DataParallel(model['R'])
            model['R'].to(device)
        return model

    if 'cityscapes' in data:
        model = {}
        model['rep'] = ResnetDilated(resnet_cityscapes.__dict__['resnet50'](pretrained=True), dropout=add_dropout)
        model['rep'].to(device)
        if "segmentation" in tasks:
            model["segmentation"] = DeepLabHead(2048, 7, log_softmax_out=True, no_dropout=no_dropout)
            model["segmentation"].to(device)
        if "depth" in tasks:
            model["depth"] = DeepLabHead(2048, 1, log_softmax_out=False, no_dropout=no_dropout)
            model["depth"].to(device)
        return model

    if 'celeba' in data:
        model = {}
        model['rep'] = ResNet(BasicBlock, [2, 2, 2, 2], dropout=add_dropout)
        if parallel:
            model['rep'] = nn.DataParallel(model['rep'])
        model['rep'].to(device)
        for t in tasks:
            model[t] = FaceAttributeDecoder()
            if parallel:
                model[t] = nn.DataParallel(model[t])
            model[t].to(device)
        return model

