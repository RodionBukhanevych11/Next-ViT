# Copyright (c) 2015-present, Facebook, Inc.
# All rights reserved.
"""
Train and eval functions used in main.py
"""
import math
import sys
from typing import Iterable, Optional

import torch
import numpy as np
from timm.data import Mixup
from timm.utils import accuracy, ModelEma
from copy import deepcopy
from losses import DistillationLoss
import utils
from tqdm import tqdm

def train_one_epoch(model: torch.nn.Module, criterion: DistillationLoss,
                    data_loader: Iterable, optimizer: torch.optim.Optimizer,
                    device: torch.device, epoch: int, loss_scaler, max_norm: float = 0,
                    model_ema: Optional[ModelEma] = None, mixup_fn: Optional[Mixup] = None,
                    set_training_mode=True):
    model.train(set_training_mode)
    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', utils.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    for samples, targets in tqdm(data_loader):
        samples = samples.to(device, non_blocking=True)
        print(samples.shape)
        targets = targets.to(device, non_blocking=True)
        with torch.cuda.amp.autocast():
            outputs = model(samples)
            loss = criterion(samples, outputs, targets)
        loss_value = loss.item()
        if not math.isfinite(loss_value):
            print("Loss is {}, stopping training".format(loss_value))
            sys.exit(1)

        optimizer.zero_grad()

        # this attribute is added by timm on one optimizer (adahessian)
        is_second_order = hasattr(optimizer, 'is_second_order') and optimizer.is_second_order
        loss_scaler(loss, optimizer, clip_grad=max_norm,
                    parameters=model.parameters(), create_graph=is_second_order)

        torch.cuda.synchronize()
        if model_ema is not None:
            model_ema.update(model)

        metric_logger.update(train_loss=loss_value)
        metric_logger.update(lr=optimizer.param_groups[0]["lr"])
    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}


@torch.no_grad()
def evaluate(data_loader, model, device, config):
    criterion = torch.nn.BCEWithLogitsLoss()
    metrics_dict = {'tp':0,'tn':0,'fp':0,'fn':0}
    nc = config['nc']
    groups = [] + list(config['labels'].values())
    metrics_per_class = []
    f1_per_class = []
    for _ in range(nc):
        metrics_per_class.append(deepcopy(metrics_dict))
    # switch to evaluation mode
    model.eval()
    all_predicts = []
    all_targets = []
    val_loss = []
    conf_th = []
    for images, target in tqdm(data_loader):
        images = images.to(device, non_blocking=True)
        target = target.to(device, non_blocking=True)
        # compute output
        with torch.cuda.amp.autocast():
            output = model(images)
            loss = criterion(output, target)
        output = torch.nn.Sigmoid()(output)
        loss_value = loss.item()
        val_loss.append(loss_value)
        output = output.cpu().detach().numpy()
        targets = target.cpu().detach().numpy()
        for prediction in output:
            all_predicts.append(prediction)
        for t in targets:
            all_targets.append(t)
        
    for i, predict_onehot in enumerate(all_predicts):
        predict = []
        predict.append(0*3+np.argmax(predict_onehot[0:3]))
        predict.append(1*3+np.argmax(predict_onehot[3:6]))
        predict.append(2*3+np.argmax(predict_onehot[6:9]))
        predict.append(3*3+np.argmax(predict_onehot[9:]))
        target = decode_labels(all_targets[i])
        for label in predict:
            if label in target:
                metrics_per_class[label]['tp']+=1
            else:
                metrics_per_class[label]['fp']+=1
        for label in target:
            if label not in predict:
                metrics_per_class[label]['fn']+=1
                
    for i in range(nc):
        pr_05 = metrics_per_class[i]['tp'] / (metrics_per_class[i]['tp'] + metrics_per_class[i]['fp'] + 1e-9)
        recall_05 = metrics_per_class[i]['tp'] / (metrics_per_class[i]['tp'] + metrics_per_class[i]['fn'] + 1e-9)
        f1_per_class.append(round(2 * pr_05 * recall_05/(pr_05 + recall_05 + 1e-9),3))

    del metrics_per_class, metrics_dict
        
    return f1_per_class, np.mean(val_loss)


def decode_labels(labels):
    labels_cat = []
    for i,label in enumerate(labels):
        if label==1:
            labels_cat.append(i)
    return labels_cat


    