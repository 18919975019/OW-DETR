import math
import os
import sys
from typing import Iterable

import torch
import util.misc as utils
from Datasets.open_world_eval import OWEvaluator
from Datasets.data_prefetcher import data_prefetcher
from util.plot_utils import plot_prediction
import matplotlib.pyplot as plt
from copy import deepcopy


def train_one_epoch(model: torch.nn.Module, criterion: torch.nn.Module,
                    data_loader: Iterable, optimizer: torch.optim.Optimizer,
                    device: torch.device, epoch: int, nc_epoch: int, max_norm: float = 0):
    # train模式, 反向传播梯度更新参数
    model.train()
    criterion.train()

    # 初始化参数字典
    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', utils.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    metric_logger.add_meter('class_error', utils.SmoothedValue(window_size=1, fmt='{value:.2f}'))
    metric_logger.add_meter('grad_norm', utils.SmoothedValue(window_size=1, fmt='{value:.2f}'))

    header = 'Epoch: [{}]'.format(epoch)
    print_freq = 10

    """
    This part is to accelerate the copy from cpu to gpu.
    When the forward propagation of current batch is carried out in default stram, 
    the next batch is prefetched into another stream parallelly. 
    """
    prefetcher = data_prefetcher(data_loader, device, prefetch=True)
    samples, targets = prefetcher.next()

    # 每个epoch循环
    for _ in metric_logger.log_every(range(len(data_loader)), print_freq, header):
        outputs = model(samples)

        # 三部分损失
        loss_dict = criterion(samples, outputs, targets, epoch)
        # 三部分损失的权重
        weight_dict = deepcopy(criterion.weight_dict)
        # condition for starting nc loss computation after certain epoch so that the F_cls branch has the time
        # to learn the within classes separation.
        # 在前n=nc_epoch个epoch, 不训练novelty classification branch, 将novelty classification的权重设置为0
        if epoch < nc_epoch:
            for k, v in weight_dict.items():
                if 'NC' in k:
                    weight_dict[k] = 0

        losses = sum(loss_dict[k] * weight_dict[k] for k in loss_dict.keys() if k in weight_dict)
        # 计算所有gpu上的平均loss,在logging中输出
        loss_dict_reduced = utils.reduce_dict(loss_dict)
        # 存储每一种loss
        loss_dict_reduced_unscaled = {f'{k}_unscaled': v for k, v in loss_dict_reduced.items()}
        # 存储每一种加权的loss
        loss_dict_reduced_scaled = {k: v * weight_dict[k] for k, v in loss_dict_reduced.items() if k in weight_dict}
        # 加权平均的total loss
        losses_reduced_scaled = sum(loss_dict_reduced_scaled.values())
        loss_value = losses_reduced_scaled.item()

        # 如果loss爆炸, 停止训练退出程序
        if not math.isfinite(loss_value):
            print("Loss is {}, stopping training".format(loss_value))
            print(loss_dict_reduced)
            sys.exit(1)

        # 优化器梯度置0, 反向传播
        optimizer.zero_grad()
        losses.backward()
        # 梯度剪裁
        if max_norm > 0:
            grad_total_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm)
        else:
            grad_total_norm = utils.get_total_grad_norm(model.parameters(), max_norm)
        optimizer.step()

        # 更新参数字典
        metric_logger.update(loss=loss_value, **loss_dict_reduced_scaled, **loss_dict_reduced_unscaled)
        metric_logger.update(class_error=loss_dict_reduced['class_error'])
        metric_logger.update(lr=optimizer.param_groups[0]["lr"])
        metric_logger.update(grad_norm=grad_total_norm)

        # 将下一个batch读进另一个stream,从cpu拷贝到gpu
        samples, targets = prefetcher.next()

    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}


## ORIGINAL FUNCTION
@torch.no_grad()
def evaluate(model, criterion, postprocessors, data_loader, base_ds, device, output_dir, args):
    # 评估模式, 梯度不反向传播
    model.eval()
    criterion.eval()

    metric_logger = utils.MetricLogger(delimiter="  ")

    header = 'Test:'
    # iou type is 'bbox'
    iou_types = tuple(k for k in postprocessors.keys())
    coco_evaluator = OWEvaluator(base_ds, iou_types)

    for samples, targets in metric_logger.log_every(data_loader, 10, header):
        samples = samples.to(device)
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
        outputs = model(samples)

        orig_target_sizes = torch.stack([t["orig_size"] for t in targets], dim=0)
        results = postprocessors['bbox'](outputs, orig_target_sizes)

        res = {target['image_id'].item(): result for target, result in zip(targets, results)}
        if coco_evaluator is not None:
            coco_evaluator.update(res)

    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    if coco_evaluator is not None:
        coco_evaluator.synchronize_between_processes()

    # accumulate predictions from all images
    if coco_evaluator is not None:
        coco_evaluator.accumulate()
        coco_evaluator.summarize()

    stats = {k: meter.global_avg for k, meter in metric_logger.meters.items()}
    if coco_evaluator is not None:
        if 'bbox' in postprocessors.keys():
            stats['coco_eval_bbox'] = coco_evaluator.coco_eval['bbox'].stats.tolist()

    return stats, coco_evaluator


@torch.no_grad()
# 预测目标可视化方法
def viz(model, criterion, postprocessors, data_loader, base_ds, device, output_dir):
    import numpy as np
    os.makedirs(output_dir, exist_ok=True)
    model.eval()
    criterion.eval()

    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter('class_error', utils.SmoothedValue(window_size=1, fmt='{value:.2f}'))

    for samples, targets in data_loader:
        samples = samples.to(device)
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
        top_k = len(targets[0]['boxes'])

        outputs = model(samples)

        indices = outputs['pred_logits'][0].softmax(-1)[..., 1].sort(descending=True)[1][:top_k]
        predictied_boxes = torch.stack([outputs['pred_boxes'][0][i] for i in indices]).unsqueeze(0)
        logits = torch.stack([outputs['pred_logits'][0][i] for i in indices]).unsqueeze(0)
        fig, ax = plt.subplots(1, 3, figsize=(10, 3), dpi=200)

        img = samples.tensors[0].cpu().permute(1, 2, 0).numpy()
        img = img * np.array([0.229, 0.224, 0.225]) + np.array([0.485, 0.456, 0.406])
        img = (img * 255)
        img = img.astype('uint8')
        h, w = img.shape[:-1]

        # Pred results
        plot_prediction(samples.tensors[0:1], predictied_boxes, logits, ax[1], plot_prob=False)
        ax[1].set_title('Prediction (Ours)')

        # Ground-Truth Results
        plot_prediction(samples.tensors[0:1], targets[0]['boxes'].unsqueeze(0),
                        torch.zeros(1, targets[0]['boxes'].shape[0], 4).to(logits), ax[2], plot_prob=False)
        ax[2].set_title('GT')

        for i in range(3):
            ax[i].set_aspect('equal')
            ax[i].set_axis_off()

        plt.savefig(os.path.join(output_dir, f'img_{int(targets[0]["image_id"][0])}.jpg'))