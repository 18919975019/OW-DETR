import argparse
import datetime
import json
import random
import time
from pathlib import Path
import os
import numpy as np
import torch
from torch.utils.data import DataLoader
import util.misc as utils
from Datasets import build_dataset, get_coco_api_from_dataset
from Datasets.coco import make_coco_transforms
from Datasets.torchvision_datasets.open_world import OWDetection
from Engine import evaluate, train_one_epoch, viz
from Models import build_model


def get_args_parser():
    parser = argparse.ArgumentParser('Deformable DETR Detector', add_help=False)
    """
    模型超参数
    :param batch_size
    :param epochs
    :param weight_decay loss function的正则化系数
    :param lr 模型的学习率
    :param lr_backbone 骨干网络的学习率
    :param lr_linear_proj_mult 线性映射层的学习率
    :param lr_drop 学习率衰减的step(40 epochs)
    :param lr_drop_epochs
    :param clip_max_norm 梯度剪裁的最大了L2范数，防止梯度爆炸
    :param sgd 优化器，默认为不使用sgd
    :param with_box_refine 是否在Deformable DETR中使用 box_refine
    :param dilation 在backbone的最后一个卷积层是否扩张
    :param position_embedding 位置编码的类型：sin函数/可学习参数
    :param position_embedding_scale 位置编码的周期
    :param num_feature_levels 特征图的尺度数
    """
    parser.add_argument('--batch_size', default=2, type=int)
    parser.add_argument('--epochs', default=51, type=int)
    parser.add_argument('--weight_decay', default=1e-4, type=float)
    parser.add_argument('--lr', default=2e-4, type=float)
    parser.add_argument('--lr_backbone', default=2e-5, type=float)
    parser.add_argument('--lr_linear_proj_mult', default=0.1, type=float)
    parser.add_argument('--lr_backbone_names', default=["backbone.0"], type=str, nargs='+')
    parser.add_argument('--lr_linear_proj_names', default=['reference_points', 'sampling_offsets'], type=str, nargs='+')
    parser.add_argument('--lr_drop', default=40, type=int)
    parser.add_argument('--lr_drop_epochs', default=None, type=int, nargs='+')
    parser.add_argument('--clip_max_norm', default=0.1, type=float,help='gradient clipping max norm')
    parser.add_argument('--sgd', action='store_true')
    parser.add_argument('--with_box_refine', default=False, action='store_true')
    parser.add_argument('--dilation', action='store_true',
                        help="If true, we replace stride with dilation in the last convolutional block (DC5)")
    parser.add_argument('--position_embedding', default='sine', type=str, choices=('sine', 'learned'),
                        help="Type of positional embedding to use on top of the image features")
    parser.add_argument('--position_embedding_scale', default=2 * np.pi, type=float, help="position / size * scale")
    parser.add_argument('--num_feature_levels', default=4, type=int, help='number of feature levels')
    """
    Transformer超参数
    :param enc_layers 编码器层数
    :param dec_layers 解码器层数
    :param dim_feedforward fc层输出维度
    :param hidden_dim 序列中每个元素的编码长度
    :param dropout dropout比率
    :param nheads 注意力头数
    :param num_queries 解码器的query数量
    :param enc_n_points 编码器参考点的采样数量
    :param dec_n_points 解码器参考点的采样数量
    :param no_aux_loss 禁用解码器每一层的辅助loss,默认为True禁用,设置False来使用
    """
    parser.add_argument('--enc_layers', default=6, type=int, help="Number of encoding layers in the transformer")
    parser.add_argument('--dec_layers', default=6, type=int, help="Number of decoding layers in the transformer")
    parser.add_argument('--dim_feedforward', default=1024, type=int,
                        help="Intermediate size of the feedforward layers in the transformer blocks")
    parser.add_argument('--hidden_dim', default=256, type=int,
                        help="Size of the embeddings (dimension of the transformer)")
    parser.add_argument('--dropout', default=0.1, type=float, help="Dropout applied in the transformer")
    parser.add_argument('--nheads', default=8, type=int,
                        help="Number of attention heads inside the transformer's attentions")
    parser.add_argument('--num_queries', default=300, type=int, help="Number of query slots")
    parser.add_argument('--dec_n_points', default=4, type=int)
    parser.add_argument('--enc_n_points', default=4, type=int)
    parser.add_argument('--no_aux_loss', dest='aux_loss', action='store_false',
                        help="Disables auxiliary decoding losses (loss at each layer)")
    """
    Matcher损失权重
    :param set_cost_class 类别损失的系数
    :param set_cost_bbox box位置损失的系数
    :param set_cost_giou giou损失的系数
    """
    parser.add_argument('--set_cost_class', default=2, type=float, help="Class coefficient in the matching cost")
    parser.add_argument('--set_cost_bbox', default=5, type=float, help="L1 box coefficient in the matching cost")
    parser.add_argument('--set_cost_giou', default=2, type=float, help="giou box coefficient in the matching cost")
    """
    Criterion损失权重
    :param cls_loss_coef 
    :param bbox_loss_coef
    :param giou_loss_coef
    :param focal_alpha focal loss的负样本衰减系数
    """
    parser.add_argument('--cls_loss_coef', default=2, type=float)
    parser.add_argument('--bbox_loss_coef', default=5, type=float)
    parser.add_argument('--giou_loss_coef', default=2, type=float)
    parser.add_argument('--mask_loss_coef', default=1, type=float)
    parser.add_argument('--dice_loss_coef', default=1, type=float)
    parser.add_argument('--focal_alpha', default=0.25, type=float)
    """
    Dataset超参数
    :param remove_difficult
    :param output_dir
    :param device
    :param seed
    :param resume
    :param start_epoch
    :param eval 默认为False,设置True来评估
    :param viz 默认为False, 设置True来可视化bonding box
    :param eval_every
    :param num_workers
    :param cache_mode 是否在内存中缓存图片
    """
    parser.add_argument('--remove_difficult', action='store_true')
    parser.add_argument('--output_dir', default='', help='path where to save, empty for no saving')
    parser.add_argument('--device', default='cuda', help='device to use for training / testing')
    parser.add_argument('--seed', default=42, type=int)
    parser.add_argument('--resume', default='', help='resume from checkpoint')
    parser.add_argument('--start_epoch', default=0, type=int, metavar='N', help='start epoch')
    parser.add_argument('--eval', action='store_true')
    parser.add_argument('--viz', action='store_true')
    parser.add_argument('--eval_every', default=1, type=int)
    parser.add_argument('--num_workers', default=2, type=int)
    parser.add_argument('--cache_mode', default=False, action='store_true', help='whether to cache images on memory')
    """
    OW-DETR超参数
    :param dataset 数据集文件名
    :param data_root 数据集根目录
    :param backbone 骨干网络的名称，用来从torchvision中调用骨干网络
    :param PREV_INTRODUCED_CLS: 前一训练阶段的可识别类别数量
    :param CUR_INTRODUCED_CLS: 当前训练阶段的可识别类别数量
    :param top_unk: 平均激活强度最高的k个proposal
    :param unmatched_boxes: 是否从未匹配的proposal里选出未知物体
    :param featdim 计算平均激活强度的特征图channel数量
    :param pretrain 预训练模型的名称
    :param train_set 训练日志数据
    :param test_set 测试日志数据
    :param NC_branch 是否启用object classification branch,默认False
    :param nc_loss_coef novelty classification branch损失系数
    :param nc_epoch :开始object classification的epoch数
    :param invalid_cls_logits: 当前训练阶段不可识别的类别标记
    :param num_classes 分类类别数量
    :param bbox_thresh 锚框的置信度，大于该置信度回归为锚框
    """
    parser.add_argument('--backbone', default='resnet50', type=str, help="Name of the convolutional backbone to use")
    parser.add_argument('--dataset', default='owod')
    parser.add_argument('--data_root', default='../data/OWDETR', type=str)
    parser.add_argument('--PREV_INTRODUCED_CLS', default=0, type=int)
    parser.add_argument('--CUR_INTRODUCED_CLS', default=20, type=int)
    parser.add_argument('--bbox_thresh', default=0.3, type=float)
    parser.add_argument('--top_unk', default=5, type=int)
    parser.add_argument('--unmatched_boxes', default=False, action='store_true')
    parser.add_argument('--featdim', default=1024, type=int)
    parser.add_argument('--pretrain', default='', help='initialized from the pre-training model')
    parser.add_argument('--train_set', default='', help='training txt files')
    parser.add_argument('--test_set', default='', help='testing txt files')
    parser.add_argument('--NC_branch', default=False, action='store_true')
    parser.add_argument('--nc_loss_coef', default=2, type=float)
    parser.add_argument('--invalid_cls_logits', default=False, action='store_true', help='owod setting')
    parser.add_argument('--nc_epoch', default=0, type=int)
    parser.add_argument('--num_classes', default=81, type=int)

    return parser


def main(args):

    # 设置分布式
    utils.init_distributed_mode(args)
    print("git:\n  {}\n".format(utils.get_sha()))
    # 设置gpu
    device = torch.device(args.device)
    # 设置random seed以复现
    seed = args.seed + utils.get_rank()
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    # 初始化model, 损失函数计算模块, 后处理模块
    model, criterion, postprocessors = build_model(args)
    # 计算模型参数数量
    model_without_ddp = model
    print(model_without_ddp)
    n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print('number of params:', n_parameters)

# ---------------------------------------------------Data Loading-------------------------------------------------------

    # 获取预处理的数据集
    dataset_train, dataset_val = get_datasets(args)
    # 从dataset中加载采样集合
    sampler_train = torch.utils.data.RandomSampler(dataset_train)
    sampler_val = torch.utils.data.SequentialSampler(dataset_val)
    # 将train set的采样集合转为批处理样本
    batch_sampler_train = torch.utils.data.BatchSampler(sampler_train, args.batch_size, drop_last=True)
    data_loader_train = DataLoader(dataset_train, batch_sampler=batch_sampler_train,
                                   collate_fn=utils.collate_fn, num_workers=args.num_workers,
                                   pin_memory=True)
    data_loader_val = DataLoader(dataset_val, args.batch_size, sampler=sampler_val,
                                 drop_last=False, collate_fn=utils.collate_fn, num_workers=args.num_workers,
                                 pin_memory=True)
    # lr_backbone_names = ["backbone.0", "backbone.neck", "input_proj", "transformer.encoder"]
    # 匹配函数, 如果匹配到key words中的任何一个element, 返回true
    def match_name_keywords(n, name_keywords):
        out = False
        for b in name_keywords:
            if b in n:
                out = True
                break
        return out
    # 存储模型中不同部分["backbone.0", "backbone.neck", "input_proj", "transformer.encoder"]的学习率
    param_dicts = [
        {
            "params":
                [p for n, p in model_without_ddp.named_parameters()
                if not match_name_keywords(n, args.lr_backbone_names) and not match_name_keywords(n,args.lr_linear_proj_names) and p.requires_grad],
            "lr": args.lr,
        },
        {
            "params":
                [p for n, p in model_without_ddp.named_parameters()
                if match_name_keywords(n, args.lr_backbone_names) and p.requires_grad],
            "lr": args.lr_backbone,
        },
        {
            "params":
                [p for n, p in model_without_ddp.named_parameters()
                if match_name_keywords(n, args.lr_linear_proj_names) and p.requires_grad],
            "lr": args.lr * args.lr_linear_proj_mult,
        }
    ]
    # 设置优化器
    optimizer = torch.optim.AdamW(param_dicts, lr=args.lr, weight_decay=args.weight_decay)
    # 设置学习率衰减优化器
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, args.lr_drop)


    if args.dataset == "coco":
        base_ds = get_coco_api_from_dataset(dataset_val)
    else:
        base_ds = dataset_val
    # 保存模型路径
    output_dir = Path(args.output_dir)


    if args.pretrain:
        print('Initialized from the pre-training model')
        checkpoint = torch.load(args.pretrain, map_location='cpu')
        state_dict = checkpoint['model']
        msg = model_without_ddp.load_state_dict(state_dict, strict=False)
        print(msg)
        args.start_epoch = checkpoint['epoch'] + 1

    if args.resume:
        if args.resume.startswith('https'):
            checkpoint = torch.hub.load_state_dict_from_url(
                args.resume, map_location='cpu', check_hash=True)
        else:
            checkpoint = torch.load(args.resume, map_location='cpu')
        missing_keys, unexpected_keys = model_without_ddp.load_state_dict(checkpoint['model'], strict=False)
        unexpected_keys = [k for k in unexpected_keys if not (k.endswith('total_params') or k.endswith('total_ops'))]
        if len(missing_keys) > 0:
            print('Missing Keys: {}'.format(missing_keys))
        if len(unexpected_keys) > 0:
            print('Unexpected Keys: {}'.format(unexpected_keys))
        if not args.eval and 'optimizer' in checkpoint and 'lr_scheduler' in checkpoint and 'epoch' in checkpoint:
            import copy
            p_groups = copy.deepcopy(optimizer.param_groups)
            optimizer.load_state_dict(checkpoint['optimizer'])
            for pg, pg_old in zip(optimizer.param_groups, p_groups):
                pg['lr'] = pg_old['lr']
                pg['initial_lr'] = pg_old['initial_lr']
            print(optimizer.param_groups)
            lr_scheduler.load_state_dict(checkpoint['lr_scheduler'])
            # todo: this is a hack for doing experiment that resume from checkpoint and also modify lr scheduler (e.g., decrease lr in advance).
            args.override_resumed_lr_drop = True
            if args.override_resumed_lr_drop:
                print(
                    'Warning: (hack) args.override_resumed_lr_drop is set to True, so args.lr_drop would override lr_drop in resumed lr_scheduler.')
                lr_scheduler.step_size = args.lr_drop
                lr_scheduler.base_lrs = list(map(lambda group: group['initial_lr'], optimizer.param_groups))
            lr_scheduler.step(lr_scheduler.last_epoch)
            args.start_epoch = checkpoint['epoch'] + 1
        # check the resumed model
        if not args.eval and not args.viz and args.dataset in ['coco', 'voc']:
            test_stats, coco_evaluator = evaluate(
                model, criterion, postprocessors, data_loader_val, base_ds, device, args.output_dir, args
            )
        if args.eval:
            test_stats, coco_evaluator = evaluate(model, criterion, postprocessors, data_loader_val, base_ds, device,
                                                  args.output_dir, args)
            if args.output_dir:
                utils.save_on_master(coco_evaluator.coco_eval["bbox"].eval, output_dir / "eval.pth")
            return

    # 检测目标可视化
    if args.viz:
        viz(model, criterion, postprocessors, data_loader_val, base_ds, device, args.output_dir)
        return

# -------------------------------------------------Train and Evaluate Process-------------------------------------------
    print("Start Training")
    start_time = time.time()
    for epoch in range(args.start_epoch, args.epochs):
        train_stats = train_one_epoch(
            model, criterion, data_loader_train, optimizer, device, epoch, args.nc_epoch, args.clip_max_norm)
        lr_scheduler.step()

        # 分断点存储模型
        if args.output_dir:
            checkpoint_paths = [output_dir / 'checkpoint.pth']
            # extra checkpoint before LR drop and every 5 epochs
            # 在每次学习率衰减或每训练5个epoch时，存储断点模型
            if (epoch + 1) % args.lr_drop == 0 or (epoch + 1) % 5 == 0:
                checkpoint_paths.append(output_dir / f'checkpoint{epoch:04}.pth')
            for checkpoint_path in checkpoint_paths:
                utils.save_on_master({
                    'model': model_without_ddp.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'lr_scheduler': lr_scheduler.state_dict(),
                    'epoch': epoch,
                    'args': args,
                }, checkpoint_path)

        # 每训练5个epcoh, evaluate模型
        if args.dataset in ['owod'] and epoch % args.eval_every == 0 and epoch > 0:
            test_stats, coco_evaluator = evaluate(
                model, criterion, postprocessors, data_loader_val, base_ds, device, args.output_dir, args
            )
        else:
            test_stats = {}

        # 训练日志
        log_stats = {**{f'train_{k}': v for k, v in train_stats.items()},
                     **{f'test_{k}': v for k, v in test_stats.items()},
                     'epoch': epoch,
                     'n_parameters': n_parameters}

        # 存储训练日志和测试日志
        if args.output_dir and utils.is_main_process():
            with (output_dir / "log.txt").open("a") as f:
                f.write(json.dumps(log_stats) + "\n")

            if args.dataset in ['owod'] and epoch % args.eval_every == 0 and epoch > 0:
                # for evaluation logs
                if coco_evaluator is not None:
                    (output_dir / 'eval').mkdir(exist_ok=True)
                    if "bbox" in coco_evaluator.coco_eval:
                        filenames = ['latest.pth']
                        if epoch % 50 == 0:
                            filenames.append(f'{epoch:03}.pth')
                        for name in filenames:
                            torch.save(coco_evaluator.coco_eval["bbox"].eval,
                                       output_dir / "eval" / name)
    # 计算训练时间
    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print('Training time {}'.format(total_time_str))

def get_datasets(args):
    print(args.dataset)
    # owod_path: '../data/OWDETR/VOC2007'
    if args.dataset == 'owod':
        dataset_train = OWDetection(args, args.owod_path, ["2007"], image_sets=[args.train_set],
                                    transforms=make_coco_transforms(args.train_set))
        dataset_val = OWDetection(args, args.owod_path, ["2007"], image_sets=[args.test_set],
                                  transforms=make_coco_transforms(args.test_set))
    else:
        raise ValueError("Wrong dataset name")

    print(args.dataset)
    print(args.train_set)
    print(args.test_set)
    print(dataset_train)
    print(dataset_val)

    return dataset_train, dataset_val

def set_dataset_path(args):
    # '../data/OWDETR/VOC2007'
    args.owod_path = os.path.join(args.data_root, 'VOC2007')


# -------------------------------------------------Main Process---------------------------------------------------------
if __name__ == '__main__':
    parser = argparse.ArgumentParser('Deformable DETR training and evaluation script', parents=[get_args_parser()])
    args = parser.parse_args()
    set_dataset_path(args)
    if args.output_dir:
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    main(args)