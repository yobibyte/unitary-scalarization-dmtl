# Adapted from: https://github.com/meetshah1995/pytorch-semseg/blob/master/ptsemseg/metrics.py

import numpy as np
import torch
import functools


class RunningMetric(object):
    def __init__(self, metric_type, n_classes =None):
        self._metric_type = metric_type
        if metric_type == 'ACC':
            self.accuracy = 0.0
            self.num_updates = 0.0
        if metric_type == 'L1':
            self.l1_abs = 0.0
            self.l1_rel = 0.0
            self.num_updates = 0.0
        if metric_type == 'IOU':
            if n_classes is None:
                print('ERROR: n_classes is needed for IOU')
            self.num_updates = 0
            self._n_classes = n_classes
            self.confusion_matrix = torch.zeros((n_classes, n_classes))

    def reset(self):
        if self._metric_type == 'ACC':
            self.accuracy = 0.0
            self.num_updates = 0.0
        if self._metric_type == 'L1':
            self.l1_abs = 0.0
            self.l1_rel = 0.0
            self.num_updates = 0.0
        if self._metric_type == 'IOU':
            self.num_updates = 0
            self.confusion_matrix = torch.zeros((self._n_classes, self._n_classes))

    def update(self, pred, gt):
        if self._metric_type == 'ACC':
            predictions = pred.data.max(1, keepdim=True)[1]
            self.accuracy += (predictions.eq(gt.data.view_as(predictions)).sum())
            self.num_updates += predictions.shape[0]
    
        if self._metric_type == 'L1':
            # Adapted from https://github.com/lorenmt/mtan/blob/master/im2im_pred/utils.py
            binary_mask = (torch.sum(gt, dim=1) != 0).unsqueeze(1).to(pred.device)
            x_pred_true = pred.masked_select(binary_mask)
            x_output_true = gt.masked_select(binary_mask)
            abs_err = torch.abs(x_pred_true - x_output_true)
            rel_err = torch.abs(x_pred_true - x_output_true) / x_output_true
            self.l1_abs += abs_err.sum()
            self.l1_rel += rel_err.sum()
            self.num_updates += torch.sum(binary_mask)

        if self._metric_type == 'IOU':
            target = gt.flatten().long()
            pred = pred.argmax(1).flatten()
            n = self._n_classes
            k = (target >= 0) & (target < n)
            inds = n * target[k] + pred[k]
            self.confusion_matrix += torch.bincount(inds, minlength=n ** 2).reshape(n, n).cpu()
        
    def get_result(self):
        # Note: the 'init' values are not valid metrics, and are only used to initialize model_saver
        if self._metric_type == 'ACC':
            return {'acc': 'init' if self.num_updates == 0 else self.accuracy/self.num_updates}
        if self._metric_type == 'L1':
            return {'l1_abs': 'init' if self.num_updates == 0 else self.l1_abs/self.num_updates,
                    'l1_rel': 'init' if self.num_updates == 0 else self.l1_rel/self.num_updates}
        if self._metric_type == 'IOU':
            acc = torch.diag(self.confusion_matrix).sum() / self.confusion_matrix.sum()
            iou = torch.diag(self.confusion_matrix) / \
                  (self.confusion_matrix.sum(1) + self.confusion_matrix.sum(0) - torch.diag(self.confusion_matrix))
            return {'acc': acc, 'mIOU': iou.mean()}


def get_metrics(dataset, tasks):
    # Returns a dictionary of metrics, and a function whose output aggregates all metrics
    # model_saver is a dict of functions of the metric dict: we save a model each time a larger value of this function
    # is attained
    met = {}
    aggregators = {}
    model_saver = {}
    if 'mnist' in dataset:
        for t in tasks:
            met[t] = RunningMetric(metric_type = 'ACC')
        aggregators["avg"] = lambda met_dict: np.array([cmet['acc'].item() for cmet in met_dict.values()]).mean()
        aggregators["min"] = lambda met_dict: np.min(np.array([cmet['acc'].item() for cmet in met_dict.values()]))
        model_saver["best"] = aggregators["avg"]
    if 'cityscapes' in dataset:
        if 'segmentation' in tasks:
            met['segmentation'] = RunningMetric(metric_type='IOU', n_classes=7)
        if 'I' in tasks:
            raise NotImplementedError("Instance segmentation is not implemented in rlw database.")
            met['I'] = RunningMetric(metric_type='L1')
        if 'depth' in tasks:
            met['depth'] = RunningMetric(metric_type='L1')

        # For cityscapes, we save a separate model for the best result on each metric.
        def metric_function(task, task_metric, met_dict):
            if task == "segmentation":
                return met_dict[task][task_metric]
            else:
                # assert task == 'depth'
                return -met_dict[task][task_metric]
        for t in tasks:
            for task_metric in met[t].get_result():
                model_saver[t + "_" + task_metric] = functools.partial(metric_function, t, task_metric)

    if 'celeba' in dataset:
        for t in tasks:
            met[t] = RunningMetric(metric_type = 'ACC')
        aggregators["avg"] = lambda met_dict: np.array([cmet['acc'].item() for cmet in met_dict.values()]).mean()
        aggregators["min"] = lambda met_dict: np.min(np.array([cmet['acc'].item() for cmet in met_dict.values()]))
        model_saver["best"] = aggregators["avg"]
    return met, aggregators, model_saver
