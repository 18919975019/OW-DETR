import torch
import torch.nn.functional as F
from torch import nn
import math

from util import box_ops
from util.misc import (NestedTensor, nested_tensor_from_tensor_list,
                       accuracy, get_world_size, interpolate,
                       is_dist_avail_and_initialized, inverse_sigmoid)

from .ResNet_50 import build_backbone
from .DeformableTransformer import build_deforamble_transformer
from .HungarianMatcher import build_matcher
from .segmentation import (DETRsegm, PostProcessPanoptic, PostProcessSegm,
                           dice_loss, sigmoid_focal_loss)

import copy


def _get_clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for i in range(N)])

class MLP(nn.Module):
    """ Very simple multi-layer perceptron (also called FFN)"""

    def __init__(self, input_dim, hidden_dim, output_dim, num_layers):
        super().__init__()
        self.num_layers = num_layers
        h = [hidden_dim] * (num_layers - 1)
        self.layers = nn.ModuleList(nn.Linear(n, k) for n, k in zip([input_dim] + h, h + [output_dim]))

    def forward(self, x):
        for i, layer in enumerate(self.layers):
            x = F.relu(layer(x)) if i < self.num_layers - 1 else layer(x)
        return x


class DeformableDETR(nn.Module):
    """ This is the Deformable DETR module that performs object detection """
    def    __init__(self, backbone, transformer, num_classes, num_queries, num_feature_levels,
                 aux_loss=True, with_box_refine=False):
        """ Initializes the model.
        Parameters:
            backbone: torch module of the backbone to be used. See backbone.py
            transformer: torch module of the transformer architecture. See transformer.py
            num_classes: number of object classes
            num_queries: number of object queries, ie detection slot. This is the maximal number of objects
                         DETR can detect in a single image. For COCO, we recommend 100 queries.
            aux_loss: True if auxiliary decoding losses (loss at each decoder layer) are to be used.
            with_box_refine: iterative bounding box refinement
        """
        super().__init__()

        d_model = transformer.d_model
        self.num_queries = num_queries
        self.n_levels = num_feature_levels

        # self.query_embed网络：把每个query编码成2 * d_model的向量，前半段为常规embedding，后半段为pos_embedding
        self.query_embed = nn.Embedding(num_queries, d_model * 2)
        self.transformer = transformer
        self.backbone = backbone
        self.class_predictor = nn.Linear(d_model, num_classes)
        # 多层感知机，用transformer输出的注意力向量预测框的坐标
        # __init__(self, input_dim, hidden_dim, output_dim, num_layers): output.size(x,y,w,h)
        self.bbox_predictor = MLP(d_model, d_model, 4, 3)


        if num_feature_levels > 1:
            n_resolutions = backbone.n_resolutions
            channel_convertor = []
            # 对于通过不同步长卷积输出的不同尺度的特征图（C3,C4,C5）,通过1*1卷积核将不同大小的in_channels转换为统一的hidden_dim(即d_model)
            for _ in range(n_resolutions):
                in_channels = backbone.n_channels[_]
                channel_convertor.append(nn.Sequential(
                    nn.Conv2d(in_channels, d_model, kernel_size=1),
                    nn.GroupNorm(32, d_model),
                ))
            # 对于C5的输出,通过步长为2的3*3卷积核将它的in_channels转换为统一的hidden_dim(即d_model)，得到第四个尺度的特征图
            for _ in range(self.n_levels - n_resolutions):
                channel_convertor.append(nn.Sequential(
                    nn.Conv2d(in_channels, d_model, kernel_size=3, stride=2, padding=1),
                    nn.GroupNorm(32, d_model),
                ))
                in_channels = d_model
            # self.channel_convertor网络：将backbone输出的不同尺度不同channel的特征图全部转换到d_model大小
            self.channel_convertor = nn.ModuleList(channel_convertor)
        else:
            self.input_proj = nn.ModuleList([
                nn.Sequential(
                    nn.Conv2d(backbone.n_channels[0], d_model, kernel_size=1),
                    nn.GroupNorm(32, d_model),
                )])

        self.aux_loss = aux_loss
        self.with_box_refine = with_box_refine

        # 先验概率
        prior_prob = 0.01
        bias_value = -math.log((1 - prior_prob) / prior_prob)
        # 初始化self.class_embed网络的偏置
        self.class_predictor.bias.data = torch.ones(num_classes) * bias_value
        # 用0初始化self.bbox_embed（MLP）最后一个线性层的权重和偏置
        nn.init.constant_(self.bbox_predictor.layers[-1].weight.data, 0)
        nn.init.constant_(self.bbox_predictor.layers[-1].bias.data, 0)
        for proj in self.channel_convertor:
            nn.init.xavier_uniform_(proj[0].weight, gain=1)
            nn.init.constant_(proj[0].bias, 0)

        # if two-stage, the last class_embed and bbox_embed is for region proposal generation
        # 如果是两阶段模型，输出decoder层数+1个预测，否则每层输出一个预测
        n_pred = transformer.decoder.n_layers
        if with_box_refine:
            # 为每个decoder_layer设置一个self.bbox_embed,利用每个decoder_layer的输出向量来预测一次候选框
            self.class_predictor = _get_clones(self.class_predictor, n_pred)
            self.bbox_predictor = _get_clones(self.bbox_predictor, n_pred)
            nn.init.constant_(self.bbox_predictor[0].layers[-1].bias.data[2:], -2.0)
            # hack implementation for iterative bounding box refinement
            self.transformer.decoder.bbox_embed = self.bbox_predictor
        else:
            nn.init.constant_(self.bbox_predictor.layers[-1].bias.data[2:], -2.0)
            # decoder每一层的输出通过一个self.class_embed网络，预测一次类别+框坐标
            # 每一层class_embed和bbox_embed为浅拷贝, 共享参数, 随着训练同时变化参数，即对每一层使用同样的网络预测
            self.class_predictor = nn.ModuleList([self.class_predictor for _ in range(n_pred)])
            self.bbox_predictor = nn.ModuleList([self.bbox_predictor for _ in range(n_pred)])
            self.transformer.decoder.bbox_embed = None

    def forward(self, samples: NestedTensor):
        """ The forward expects a NestedTensor, which consists of:
               - samples.tensor: batched images, of shape [batch_size x 3 x H x W]
               - samples.mask: a binary mask of shape [batch_size x H x W], containing 1 on padded pixels

            It returns a dict with the following elements:
               - "pred_logits": the classification logits (including no-object) for all queries.
                                Shape= [batch_size x num_queries x (num_classes + 1)]
               - "pred_boxes": The normalized boxes coordinates for all queries, represented as
                               (center_x, center_y, height, width). These values are normalized in [0, 1],
                               relative to the size of each individual image (disregarding possible padding).
                               See PostProcess for information on how to retrieve the unnormalized bounding box.
               - "aux_outputs": Optional, only returned when auxilary losses are activated. It is a list of
                                dictionnaries containing the two above keys for each decoder layer.
        """
        if not isinstance(samples, NestedTensor):
            samples = nested_tensor_from_tensor_list(samples)
        # 由backbone得到src、mask、pos
        feature_maps, pos = self.backbone(samples)

        srcs = []
        masks = []
        # 将backbone输出的不同尺度不同channel的特征图————>深度均为d_model的几个不同尺度的特征图及其掩码矩阵
        for l, feature_map in enumerate(feature_maps):
            src, mask = feature_map.decompose()
            srcs.append(self.channel_convertor[l](src))
            masks.append(mask)
            assert mask is not None

        # 如果DDETR的特征图尺度数多于backbone输出的特征图尺度数，多出来的新尺度特征图由srcs的最后一个尺度的特征图卷积得到
        # 掩码矩阵由输入图片（samples）的mask插值得到
        if self.n_levels > len(srcs):
            _len_srcs = len(srcs)
            for l in range(_len_srcs, self.n_levels):
                if l == _len_srcs:
                    src = self.channel_convertor[l](feature_maps[-1].tensors)
                else:
                    src = self.channel_convertor[l](srcs[-1])
                m = samples.mask
                mask = F.interpolate(m[None].float(), size=src.shape[-2:]).to(torch.bool)[0]
                pos_l = self.backbone[1](NestedTensor(src, mask)).to(src.dtype)
                srcs.append(src)
                masks.append(mask)
                pos.append(pos_l)

        query_embeds = self.query_embed.weight
        hs, init_reference, inter_references, enc_outputs_class, enc_outputs_coordinate = self.transformer(srcs, masks, pos, query_embeds)

        outputs_classes = []
        outputs_coordinates = []
        for layer in range(hs.shape[0]):
            if layer == 0:
                reference = init_reference
            else:
                reference = inter_references[layer - 1]
            reference = inverse_sigmoid(reference)
            outputs_class = self.class_predictor[layer](hs[layer])
            tmp = self.bbox_predictor[layer](hs[layer])
            if reference.shape[-1] == 4:
                tmp += reference
            else:
                assert reference.shape[-1] == 2
                tmp[..., :2] += reference
            outputs_coordinate = tmp.sigmoid()
            outputs_classes.append(outputs_class)
            outputs_coordinates.append(outputs_coordinate)
        outputs_class = torch.stack(outputs_classes)
        outputs_coordinate = torch.stack(outputs_coordinate)

        out = {'pred_logits': outputs_class[-1], 'pred_boxes': outputs_coordinate[-1]}
        if self.aux_loss:
            out['aux_outputs'] = self._set_aux_loss(outputs_class, outputs_coordinate)

        return out

    @torch.jit.unused
    def _set_aux_loss(self, outputs_class, outputs_coord):
        # this is a workaround to make torchscript happy, as torchscript
        # doesn't support dictionary with non-homogeneous values, such
        # as a dict having both a Tensor and a list.
        return [{'pred_logits': a, 'pred_boxes': b}
                for a, b in zip(outputs_class[:-1], outputs_coord[:-1])]


class SetCriterion(nn.Module):
    """ This class computes the loss for DETR.
    The process happens in two steps:
        1) we compute hungarian assignment between ground truth boxes and the outputs of the model
        2) we supervise each pair of matched ground-truth / prediction (supervise class and box)
    """
    def __init__(self, num_classes, matcher, weight_dict, losses, focal_alpha=0.25):
        """ Create the criterion.
        Parameters:
            num_classes: number of object categories, omitting the special no-object category
            matcher: module able to compute a matching between targets and proposals
            weight_dict: dict containing as key the names of the losses and as values their relative weight.
            losses: list of all the losses to be applied. See get_loss for list of available losses.
            focal_alpha: alpha in Focal Loss
        """
        super().__init__()
        self.num_classes = num_classes
        self.matcher = matcher
        self.weight_dict = weight_dict
        self.losses = losses
        self.focal_alpha = focal_alpha

    def loss_labels(self, outputs, targets, indices, num_boxes, log=True):
        """Classification loss (NLL)
        targets dicts must contain the key "labels" containing a tensor of dim [nb_target_boxes]
        """
        assert 'pred_logits' in outputs
        src_logits = outputs['pred_logits']

        idx = self._get_src_permutation_idx(indices)
        target_classes_o = torch.cat([t["labels"][J] for t, (_, J) in zip(targets, indices)])
        target_classes = torch.full(src_logits.shape[:2], self.num_classes,
                                    dtype=torch.int64, device=src_logits.device)
        target_classes[idx] = target_classes_o

        target_classes_onehot = torch.zeros([src_logits.shape[0], src_logits.shape[1], src_logits.shape[2] + 1],
                                            dtype=src_logits.dtype, layout=src_logits.layout, device=src_logits.device)
        target_classes_onehot.scatter_(2, target_classes.unsqueeze(-1), 1)

        target_classes_onehot = target_classes_onehot[:,:,:-1]
        loss_ce = sigmoid_focal_loss(src_logits, target_classes_onehot, num_boxes, alpha=self.focal_alpha, gamma=2) * src_logits.shape[1]
        losses = {'loss_ce': loss_ce}

        if log:
            # TODO this should probably be a separate loss, not hacked in this one here
            losses['class_error'] = 100 - accuracy(src_logits[idx], target_classes_o)[0]
        return losses

    @torch.no_grad()
    def loss_cardinality(self, outputs, targets, indices, num_boxes):
        """ Compute the cardinality error, ie the absolute error in the number of predicted non-empty boxes
        This is not really a loss, it is intended for logging purposes only. It doesn't propagate gradients
        """
        pred_logits = outputs['pred_logits']
        device = pred_logits.device
        tgt_lengths = torch.as_tensor([len(v["labels"]) for v in targets], device=device)
        # Count the number of predictions that are NOT "no-object" (which is the last class)
        card_pred = (pred_logits.argmax(-1) != pred_logits.shape[-1] - 1).sum(1)
        card_err = F.l1_loss(card_pred.float(), tgt_lengths.float())
        losses = {'cardinality_error': card_err}
        return losses

    def loss_boxes(self, outputs, targets, indices, num_boxes):
        """Compute the losses related to the bounding boxes, the L1 regression loss and the GIoU loss
           targets dicts must contain the key "boxes" containing a tensor of dim [nb_target_boxes, 4]
           The target boxes are expected in format (center_x, center_y, h, w), normalized by the image size.
        """
        assert 'pred_boxes' in outputs
        idx = self._get_src_permutation_idx(indices)
        src_boxes = outputs['pred_boxes'][idx]
        target_boxes = torch.cat([t['boxes'][i] for t, (_, i) in zip(targets, indices)], dim=0)

        loss_bbox = F.l1_loss(src_boxes, target_boxes, reduction='none')

        losses = {}
        losses['loss_bbox'] = loss_bbox.sum() / num_boxes

        loss_giou = 1 - torch.diag(box_ops.generalized_box_iou(
            box_ops.box_cxcywh_to_xyxy(src_boxes),
            box_ops.box_cxcywh_to_xyxy(target_boxes)))
        losses['loss_giou'] = loss_giou.sum() / num_boxes
        return losses

    def loss_masks(self, outputs, targets, indices, num_boxes):
        """Compute the losses related to the masks: the focal loss and the dice loss.
           targets dicts must contain the key "masks" containing a tensor of dim [nb_target_boxes, h, w]
        """
        assert "pred_masks" in outputs

        src_idx = self._get_src_permutation_idx(indices)
        tgt_idx = self._get_tgt_permutation_idx(indices)

        src_masks = outputs["pred_masks"]

        # TODO use valid to mask invalid areas due to padding in loss
        target_masks, valid = nested_tensor_from_tensor_list([t["masks"] for t in targets]).decompose()
        target_masks = target_masks.to(src_masks)

        src_masks = src_masks[src_idx]
        # upsample predictions to the target size
        src_masks = interpolate(src_masks[:, None], size=target_masks.shape[-2:],
                                mode="bilinear", align_corners=False)
        src_masks = src_masks[:, 0].flatten(1)

        target_masks = target_masks[tgt_idx].flatten(1)

        losses = {
            "loss_mask": sigmoid_focal_loss(src_masks, target_masks, num_boxes),
            "loss_dice": dice_loss(src_masks, target_masks, num_boxes),
        }
        return losses

    def _get_src_permutation_idx(self, indices):
        # permute predictions following indices
        batch_idx = torch.cat([torch.full_like(src, i) for i, (src, _) in enumerate(indices)])
        src_idx = torch.cat([src for (src, _) in indices])
        return batch_idx, src_idx

    def _get_tgt_permutation_idx(self, indices):
        # permute targets following indices
        batch_idx = torch.cat([torch.full_like(tgt, i) for i, (_, tgt) in enumerate(indices)])
        tgt_idx = torch.cat([tgt for (_, tgt) in indices])
        return batch_idx, tgt_idx

    def get_loss(self, loss, outputs, targets, indices, num_boxes, **kwargs):
        loss_map = {
            'labels': self.loss_labels,
            'cardinality': self.loss_cardinality,
            'boxes': self.loss_boxes,
            'masks': self.loss_masks
        }
        assert loss in loss_map, f'do you really want to compute {loss} loss?'
        return loss_map[loss](outputs, targets, indices, num_boxes, **kwargs)

    def forward(self, outputs, targets):
        """ This performs the loss computation.
        Parameters:
             outputs: dict of tensors, see the output specification of the model for the format
             targets: list of dicts, such that len(targets) == batch_size.
                      The expected keys in each dict depends on the losses applied, see each loss' doc
        """
        outputs_without_aux = {k: v for k, v in outputs.items() if k != 'aux_outputs' and k != 'enc_outputs'}

        # Retrieve the matching between the outputs of the last layer and the targets
        indices = self.matcher(outputs_without_aux, targets)

        # Compute the average number of target boxes accross all nodes, for normalization purposes
        num_boxes = sum(len(t["labels"]) for t in targets)
        num_boxes = torch.as_tensor([num_boxes], dtype=torch.float, device=next(iter(outputs.values())).device)
        if is_dist_avail_and_initialized():
            torch.distributed.all_reduce(num_boxes)
        num_boxes = torch.clamp(num_boxes / get_world_size(), min=1).item()

        # Compute all the requested losses
        losses = {}
        for loss in self.losses:
            kwargs = {}
            losses.update(self.get_loss(loss, outputs, targets, indices, num_boxes, **kwargs))

        # In case of auxiliary losses, we repeat this process with the output of each intermediate layer.
        if 'aux_outputs' in outputs:
            for i, aux_outputs in enumerate(outputs['aux_outputs']):
                indices = self.matcher(aux_outputs, targets)
                for loss in self.losses:
                    if loss == 'masks':
                        # Intermediate masks losses are too costly to compute, we ignore them.
                        continue
                    kwargs = {}
                    if loss == 'labels':
                        # Logging is enabled only for the last layer
                        kwargs['log'] = False
                    l_dict = self.get_loss(loss, aux_outputs, targets, indices, num_boxes, **kwargs)
                    l_dict = {k + f'_{i}': v for k, v in l_dict.items()}
                    losses.update(l_dict)

        if 'enc_outputs' in outputs:
            enc_outputs = outputs['enc_outputs']
            bin_targets = copy.deepcopy(targets)
            for bt in bin_targets:
                bt['labels'] = torch.zeros_like(bt['labels'])
            indices = self.matcher(enc_outputs, bin_targets)
            for loss in self.losses:
                if loss == 'masks':
                    # Intermediate masks losses are too costly to compute, we ignore them.
                    continue
                kwargs = {}
                if loss == 'labels':
                    # Logging is enabled only for the last layer
                    kwargs['log'] = False
                l_dict = self.get_loss(loss, enc_outputs, bin_targets, indices, num_boxes, **kwargs)
                l_dict = {k + f'_enc': v for k, v in l_dict.items()}
                losses.update(l_dict)

        return losses


class PostProcess(nn.Module):
    """ This module converts the model's output into the format expected by the coco api"""

    @torch.no_grad()
    def forward(self, outputs, target_sizes):
        """ Perform the computation
        Parameters:
            outputs: raw outputs of the model
            target_sizes: tensor of dimension [batch_size x 2] containing the size of each images of the batch
                          For evaluation, this must be the original image size (before any data augmentation)
                          For visualization, this should be the image size after data augment, but before padding
        """
        out_logits, out_bbox = outputs['pred_logits'], outputs['pred_boxes']

        assert len(out_logits) == len(target_sizes)
        assert target_sizes.shape[1] == 2

        prob = out_logits.sigmoid()
        topk_values, topk_indexes = torch.topk(prob.view(out_logits.shape[0], -1), 100, dim=1)
        scores = topk_values
        topk_boxes = topk_indexes // out_logits.shape[2]
        labels = topk_indexes % out_logits.shape[2]
        boxes = box_ops.box_cxcywh_to_xyxy(out_bbox)
        boxes = torch.gather(boxes, 1, topk_boxes.unsqueeze(-1).repeat(1,1,4))

        # and from relative [0, 1] to absolute [0, height] coordinates
        img_h, img_w = target_sizes.unbind(1)
        scale_fct = torch.stack([img_w, img_h, img_w, img_h], dim=1)
        boxes = boxes * scale_fct[:, None, :]

        results = [{'scores': s, 'labels': l, 'boxes': b} for s, l, b in zip(scores, labels, boxes)]

        return results




def build_OWDETR(args):

    # 为模型指定分类的类别数量
    # if args.dataset_file == 'coco'
    num_classes = 91
    # 选择cpu/gpu
    device = torch.device(args.device)
    """
      初始化model
    """
    # 创建backbone网络
    backbone = build_backbone(args)
    # 创建transformer网络
    transformer = build_deforamble_transformer(args)
    # 创建DeformableDETR和matcher
    model = DeformableDETR(
        backbone,
        transformer,
        num_classes=num_classes,
        num_queries=args.num_queries,
        num_feature_levels=args.num_feature_levels,
        aux_loss=args.aux_loss,
        with_box_refine=args.with_box_refine,
    )
    model.to(device)

    """
      初始化criterion
    """
    # 初始化匹配器
    matcher = build_matcher(args)
    # 创建字典记录三部分损失：class loss, bonding box loss, giou loss
    weight_dict = {'loss_cls': args.cls_loss_coef,
                   'loss_bbox': args.bbox_loss_coef,
                   'loss_giou': args.giou_loss_coef}
    # TODO this is a hack
    # 创建字典记录decoder每一层输出的损失名称及其权重
    """
        weight_dict = {'loss_cls': args.cls_loss_coef,
                       'loss_bbox': args.bbox_loss_coef,
                       'loss_giou': args.giou_loss_coef,
                       'loss_cls_enc': args.cls_loss_coef,
                       'loss_bbox_enc': args.bbox_loss_coef,
                       'loss_giou_enc': args.giou_loss_coef,
                       'loss_cls_1':args.cls_loss_coef,
                       'loss_bbox_1':args.bbox_loss_coef,
                       'loss_giou_1':args.giou_loss_coef,
                       'loss_cls_2':args.cls_loss_coef,
                       'loss_bbox_2':args.bbox_loss_coef,
                       'loss_giou_2':args.giou_loss_coef
                       }
        """
    if not args.no_aux_loss:
        aux_weight_dict = {}
        for i in range(args.dec_layers - 1):
            aux_weight_dict.update({k + f'_{i}': v for k, v in weight_dict.items()})
        aux_weight_dict.update({k + f'_enc': v for k, v in weight_dict.items()})
        weight_dict.update(aux_weight_dict)


    losses = ['labels', 'boxes', 'cardinality']
    criterion = SetCriterion(num_classes, matcher, weight_dict, losses, focal_alpha=args.focal_alpha)
    criterion.to(device)

    """
    初始化postprocessors
    后处理模块为一个bbox字典{bbox:[{'scores': s, 'labels': l, 'boxes': b}]
    """
    # 将锚框从相对坐标映射到绝对坐标
    postprocessors = {'bbox': PostProcess()}

    return model, criterion, postprocessors
