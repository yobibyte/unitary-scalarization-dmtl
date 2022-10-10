# Code adapted from https://github.com/isl-org/MultiObjectiveOptimization/blob/master/multi_task/

import torch
import torch.nn.functional as F 

def nll(pred, gt, average=True):
    if average:
        return F.nll_loss(pred, gt)
    else:
        return F.nll_loss(pred, gt, reduction='none')


def cross_entropy2d(log_p, target, weight=None, average=True):
    if average:
        loss = F.nll_loss(log_p, target.long(), ignore_index=-1, weight=weight)
    else:
        loss = F.nll_loss(log_p, target.long(), ignore_index=-1, weight=weight, reduction='none')
        loss = loss.mean(dim=[1, 2]) # to get a scalar per pic
        
    return loss


def l1_loss_depth(input, target, average=True):
    mask = (torch.sum(target, dim=1) != 0).unsqueeze(1)

    if average:
        lss = F.l1_loss(input[mask], target[mask])
    else:
        # Get the losses per batch entry.
        lss = torch.zeros((mask.shape[0],), device=mask.device)
        for idx in range(mask.shape[0]):
            lss[idx] = F.l1_loss(input[idx][mask[idx]], target[idx][mask[idx]])
    return lss


def l1_loss_instance(input, target, average=True):
    mask = target!=250
    if mask.data.sum() < 1:
        # no instance pixel
        return None 

    if average:
        lss = F.l1_loss(input[mask], target[mask])
    else:
        lss = F.l1_loss(input[mask], target[mask], reduction='none')
    return lss


def get_loss(dataset, tasks):
    if 'mnist' in dataset:
        loss_fn = {}
        for t in tasks:
            loss_fn[t] = nll 
        return loss_fn

    if 'cityscapes' in dataset:
        loss_fn = {}
        if 'segmentation' in tasks:
            loss_fn['segmentation'] = cross_entropy2d
        if 'I' in tasks:
            raise NotImplementedError("This was not implemented in rlw code.")
            loss_fn['I'] = l1_loss_instance
        if 'depth' in tasks:
            loss_fn['depth'] = l1_loss_depth
        return loss_fn

    if 'celeba' in dataset:
        loss_fn = {}
        for t in tasks:
            loss_fn[t] = nll
        return loss_fn
