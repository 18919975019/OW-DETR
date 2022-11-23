"""
Deformable DETR model and criterion classes.
"""
import torch
import torch.nn.functional as F
from torch import nn
import math
import pickle
from util import box_ops
from util.misc import (NestedTensor, nested_tensor_from_tensor_list,
                       accuracy, get_world_size, interpolate,
                       is_dist_avail_and_initialized, inverse_sigmoid)
from .ResNet_50 import build_backbone
from .HungarianMatcher import build_matcher
from .segmentation import (DETRsegm, PostProcessPanoptic, PostProcessSegm,
                           dice_loss, sigmoid_focal_loss)
from .DeformableTransformer import build_deforamble_transformer
import copy
import heapq
import operator
import os
from copy import deepcopy


def _get_clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for i in range(N)])


class DeformableDETR(nn.Module):
    """
    This is the Deformable DETR module that performs object detection
    """
    def __init__(self, backbone, transformer, n_classes, n_queries, n_feature_levels,
                 aux_loss=True, with_box_refine=False, unmatched_boxes=False, novelty_cls=False, d_feature=1024):
        """
        Initializes the model.
        :param backbone: torch module of the backbone to be used. See backbone.py
        :param transformer: torch module of the transformer architecture. See transformer.py
        :param n_classes: number of object classes
        :param n_queries: number of object queries, ie detection slot. This is the maximal number of objects
                            DETR can detect in a single image. For COCO, we recommend 100 queries.
        :param n_feature_levels: number of feature levels
        :param aux_loss: True if auxiliary decoding losses (loss at each decoder layer) are to be used.
        :param with_box_refine: iterative bounding box refinement
        """
        super().__init__()
        d_model = transformer.d_model

        self.n_queries = n_queries
        self.n_feature_levels = n_feature_levels
        self.d_feature = d_feature

        self.transformer = transformer
        self.backbone = backbone

        # class预测头,用transformer输出的注意力向量预测query的类别
        self.class_predictor = nn.Linear(d_model, n_classes)
        # bbox预测头,用transformer输出的注意力向量预测query的bbox坐标
        self.bbox_predictor = MLP(d_model, d_model, 4, 3)
        self.unmatched_boxes = unmatched_boxes
        self.novelty_cls = novelty_cls
        # novelty classification预测头, separate the foreground objects (known and unknown) from the background
        if self.novelty_cls:
            self.nc_class_predictor = nn.Linear(d_feature, 1)
        # self.query_embed网络：把每个query编码成2 * d_model的向量，前半段为常规embedding，后半段为pos_embedding
        self.query_embed = nn.Embedding(n_queries, d_model * 2)

        if n_feature_levels > 1:
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

        # 初始化参数
        prior_prob = 0.01
        bias_value = -math.log((1 - prior_prob) / prior_prob)
        self.class_predictor.bias.data = torch.ones(n_classes) * bias_value
        if self.novelty_cls:
            self.nc_class_predictor.bias.data = torch.ones(1) * bias_value
        nn.init.constant_(self.bbox_predictor.layers[-1].weight.data, 0)
        nn.init.constant_(self.bbox_predictor.layers[-1].bias.data, 0)
        for proj in self.input_proj:
            nn.init.xavier_uniform_(proj[0].weight, gain=1)
            nn.init.constant_(proj[0].bias, 0)


        n_pred = transformer.decoder.num_layers
        if with_box_refine:
            # 深拷贝,为每个decoder每一层设置一个bbox_predictor和class_predictor,利用decoder每一层的输出向量来预测一次类别+框坐标
            self.class_predictor = _get_clones(self.class_predictor, n_pred)
            self.bbox_predictor = _get_clones(self.bbox_predictor, n_pred)
            nn.init.constant_(self.bbox_predictor[0].layers[-1].bias.data[2:], -2.0)
            self.transformer.decoder.bbox_embed = self.bbox_predictor
        else:
            # 浅拷贝,多层bbox_predictor和class_predictor共享参数以实现接口一致性, 随着训练同时变化参数，相当一个网络
            nn.init.constant_(self.bbox_predictor.layers[-1].bias.data[2:], -2.0)
            self.class_predictor = nn.ModuleList([self.class_predictor for _ in range(n_pred)])
            if self.novelty_cls:
                self.nc_class_predictor = nn.ModuleList([self.nc_class_embed for _ in range(n_pred)])
            self.bbox_predictor = nn.ModuleList([self.bbox_predictor for _ in range(n_pred)])
            self.transformer.decoder.bbox_embed = None

    def forward(self, samples: NestedTensor):
        """
        The forward expects a NestedTensor, which consists of:
        :param samples.tensor: batched images, of shape [bs,3,H,W]
        :param samples.mask:  binary mask of shape [bs,H,W], containing 1 on padded pixels

        :return out: a dict which consists of:
        "pred_logits": classification logits including no-object for all queries, of shape [bs,n_queries,(n_classes+1)]
        "pred_boxes":  relative bboxes coordinates for all queries, of shape [bs,n_queries,(x, y, h, w)]
                       These values are normalized in [0, 1],relative to the size of each individual image .
                       See PostProcess for information on how to retrieve the absolute bbox coordinates.
        "aux_outputs": a list of dicts containing the {"pred_logits","pred_boxes"} for each decoder layer.
        """
        if not isinstance(samples, NestedTensor):
            samples = nested_tensor_from_tensor_list(samples)

        """In this part, backbone take the samples as inout and output the features and pos embeds."""
        # images输入backbone产生不同level的特征图和位置编码
        features, pos = self.backbone(samples)

        """In this part, prepare features and pos embeds for the transformer."""
        srcs = []
        masks = []
        if self.d_feature == 512:
            dim_index = 0
        elif self.d_feature == 1024:
            dim_index = 1
        else:
            dim_index = 2

        # 提取每一尺度的特征图, 将它们转换到相同channel dimension
        for level, feature_map in enumerate(features):
            src, mask = feature_map.decompose()
            # extracting the resnet features which are used for selecting unmatched queries
            if self.unmatched_boxes:
                if level == dim_index:
                    resnet_1024_feature = src.clone()  # [2,1024,61,67]
            else:
                resnet_1024_feature = None
            # 将每个level的特征图的channel维度映射到同一维度d_model
            srcs.append(self.input_proj[level](src))
            masks.append(mask)
            assert mask is not None

        # 如果backbone产生的特征图数少于DDETR预设的尺度数，利用backbone最小分辨率的特征图(最后一层的输出)进行等尺度映射, 补足特征图数量
        if self.n_feature_levels > len(srcs):
            _len_srcs = len(srcs)
            for level in range(_len_srcs, self.n_feature_levels):
                if level == _len_srcs:
                    src = self.input_proj[level](features[-1].tensors)
                else:
                    src = self.input_proj[level](srcs[-1])
                m = samples.mask
                mask = F.interpolate(m[None].float(), size=src.shape[-2:]).to(torch.bool)[0]
                pos_level = self.backbone[1](NestedTensor(src, mask)).to(src.dtype)
                srcs.append(src)
                masks.append(mask)
                pos.append(pos_level)

        query_embeds = self.query_embed.weight

        """In this part, the transformer takes prepared features, pos embeds and queries as input and output the class
           and bbox coordinates of the queries."""
        hs, init_reference, inter_references, enc_outputs_class, enc_outputs_coord_unact = \
            self.transformer(srcs, masks, pos, query_embeds)
        outputs_classes = []
        outputs_coords = []
        outputs_classes_nc = []

        for layer in range(hs.shape[0]):
            # 提取当前层的参考点
            if layer == 0:
                reference = init_reference
            else:
                reference = inter_references[layer - 1]
            reference = inverse_sigmoid(reference)

            # 用decoder当前层输出向量预测类别
            outputs_class = self.class_predictor[layer](hs[layer])

            # 用decoder当前层输出向量预测前景概率(objectiveness)
            if self.novelty_cls:
                outputs_class_nc = self.nc_class_predictor[layer](hs[layer])

            # 用decoder当前层输出向量预测bbox坐标
            tmp = self.bbox_predictor[layer](hs[layer])
            if reference.shape[-1] == 4:
                tmp += reference
            else:
                assert reference.shape[-1] == 2
                tmp[..., :2] += reference
            outputs_coord = tmp.sigmoid()

            # 存储当前层的预测（类别、bbox坐标，objectiveness）
            outputs_classes.append(outputs_class)
            if self.novelty_cls:
                outputs_classes_nc.append(outputs_class_nc)
            outputs_coords.append(outputs_coord)

        outputs_class = torch.stack(outputs_classes)
        outputs_coord = torch.stack(outputs_coords)
        if self.novelty_cls:
            output_class_nc = torch.stack(outputs_classes_nc)

        out = {'pred_logits': outputs_class[-1], 'pred_boxes': outputs_coord[-1], 'resnet_1024_feat': resnet_1024_feature}

        if self.novelty_cls:
            out = {'pred_logits': outputs_class[-1], 'pred_nc_logits': output_class_nc[-1],
                   'pred_boxes': outputs_coord[-1], 'resnet_1024_feat': resnet_1024_feature}

        if self.aux_loss:
            out['aux_outputs'] = self._set_aux_loss(outputs_class, outputs_coord, output_class_nc=None)
            if self.novelty_cls:
                out['aux_outputs'] = self._set_aux_loss(outputs_class, outputs_coord, output_class_nc=output_class_nc)
        if self.two_stage:
            enc_outputs_coord = enc_outputs_coord_unact.sigmoid()
            out['enc_outputs'] = {'pred_logits': enc_outputs_class, 'pred_boxes': enc_outputs_coord}

        return out

    @torch.jit.unused
    def _set_aux_loss(self, outputs_class, outputs_coord, output_class_nc=None):
        # this is a workaround to make torchscript happy, as torchscript
        # doesn't support dictionary with non-homogeneous values, such
        # as a dict having both a Tensor and a list.
        # import pdb;pdb.set_trace()
        if output_class_nc is not None:
            xx = [{'pred_logits': a, 'pred_nc_logits': c, 'pred_boxes': b}
                  for a, c, b in zip(outputs_class[:-1], output_class_nc[:-1], outputs_coord[:-1])]
        else:
            xx = [{'pred_logits': a, 'pred_boxes': b}
                  for a, b in zip(outputs_class[:-1], outputs_coord[:-1])]
        return xx


class SetCriterion(nn.Module):
    """ This class computes the loss for DETR.
    The process happens in two steps:
        1) we compute hungarian assignment between ground truth boxes and the outputs of the model
        2) we supervise each pair of matched ground-truth / prediction (supervise class and box)
    """

    def __init__(self, args, num_classes, matcher, weight_dict, losses, invalid_cls_logits, focal_alpha=0.25):
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
        self.nc_epoch = args.nc_epoch
        self.output_dir = args.output_dir
        self.invalid_cls_logits = invalid_cls_logits
        self.unmatched_boxes = args.unmatched_boxes
        self.top_unk = args.top_unk
        self.bbox_thresh = args.bbox_thresh
        self.num_seen_classes = args.PREV_INTRODUCED_CLS + args.CUR_INTRODUCED_CLS

    def loss_NC_labels(self, outputs, targets, indices, num_boxes, current_epoch, owod_targets, owod_indices, log=True):
        """Novelty classification loss
        target labels will contain class as 1
        owod_indices -> indices combining matched indices + psuedo labeled indices
        owod_targets -> targets combining GT targets + psuedo labeled unknown targets
        target_classes_o -> contains all 1's
        """
        assert 'pred_nc_logits' in outputs
        src_logits = outputs['pred_nc_logits']

        idx = self._get_src_permutation_idx(owod_indices)
        target_classes_o = torch.cat(
            [torch.full_like(t["labels"][J], 0) for t, (_, J) in zip(owod_targets, owod_indices)])
        target_classes = torch.full(src_logits.shape[:2], 1, dtype=torch.int64, device=src_logits.device)
        target_classes[idx] = target_classes_o

        target_classes_onehot = torch.zeros([src_logits.shape[0], src_logits.shape[1], src_logits.shape[2] + 1],
                                            dtype=src_logits.dtype, layout=src_logits.layout, device=src_logits.device)
        target_classes_onehot.scatter_(2, target_classes.unsqueeze(-1), 1)

        target_classes_onehot = target_classes_onehot[:, :, :-1]
        loss_ce = sigmoid_focal_loss(src_logits, target_classes_onehot, num_boxes, alpha=self.focal_alpha, gamma=2) * \
                  src_logits.shape[1]

        losses = {'loss_NC': loss_ce}
        return losses

    def loss_labels(self, outputs, targets, indices, num_boxes, current_epoch, owod_targets, owod_indices, log=True):
        """Classification loss (NLL)
        targets dicts must contain the key "labels" containing a tensor of dim [nb_target_boxes]
        """
        assert 'pred_logits' in outputs
        ## comment lines from 317-320 when running for oracle settings
        temp_src_logits = outputs['pred_logits'].clone()
        temp_src_logits[:, :, self.invalid_cls_logits] = -10e10
        src_logits = temp_src_logits

        if self.unmatched_boxes:
            idx = self._get_src_permutation_idx(owod_indices)
            target_classes_o = torch.cat([t["labels"][J] for t, (_, J) in zip(owod_targets, owod_indices)])
        else:
            idx = self._get_src_permutation_idx(indices)
            target_classes_o = torch.cat([t["labels"][J] for t, (_, J) in zip(targets, indices)])

        target_classes = torch.full(src_logits.shape[:2], self.num_classes, dtype=torch.int64, device=src_logits.device)
        target_classes[idx] = target_classes_o
        target_classes_onehot = torch.zeros([src_logits.shape[0], src_logits.shape[1], src_logits.shape[2] + 1],
                                            dtype=src_logits.dtype, layout=src_logits.layout, device=src_logits.device)
        target_classes_onehot.scatter_(2, target_classes.unsqueeze(-1), 1)
        target_classes_onehot = target_classes_onehot[:, :, :-1]
        loss_ce = sigmoid_focal_loss(src_logits, target_classes_onehot, num_boxes, alpha=self.focal_alpha, gamma=2) * \
                  src_logits.shape[1]

        losses = {'loss_ce': loss_ce}

        if log:
            # TODO this should probably be a separate loss, not hacked in this one here
            losses['class_error'] = 100 - accuracy(src_logits[idx], target_classes_o)[0]
        return losses

    @torch.no_grad()
    def loss_cardinality(self, outputs, targets, indices, num_boxes, current_epoch, owod_targets, owod_indices):
        """ Compute the cardinality error, ie the absolute error in the number of predicted non-empty boxes
        This is not really a loss, it is intended for logging purposes only. It doesn't propagate gradients
        """
        temp_pred_logits = outputs['pred_logits'].clone()
        temp_pred_logits[:, :, self.invalid_cls_logits] = -10e10
        pred_logits = temp_pred_logits

        device = pred_logits.device
        if self.unmatched_boxes:
            tgt_lengths = torch.as_tensor([len(v["labels"]) for v in owod_targets], device=device)
        else:
            tgt_lengths = torch.as_tensor([len(v["labels"]) for v in targets], device=device)
        # Count the number of predictions that are NOT "no-object" (which is the last class)
        card_pred = (pred_logits.argmax(-1) != pred_logits.shape[-1] - 1).sum(1)
        card_err = F.l1_loss(card_pred.float(), tgt_lengths.float())
        losses = {'cardinality_error': card_err}
        return losses

    def loss_boxes(self, outputs, targets, indices, num_boxes, current_epoch, owod_targets, owod_indices):
        # def loss_boxes(self, outputs, targets, indices, num_boxes, current_epoch, owod_targets, owod_indices, ca_owod_targets, ca_owod_indices):
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

    def loss_masks(self, outputs, targets, indices, num_boxes, current_epoch, owod_targets, owod_indices):
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

    def save_dict(self, di_, filename_):
        with open(filename_, 'wb') as f:
            pickle.dump(di_, f)

    def load_dict(self, filename_):
        with open(filename_, 'rb') as f:
            ret_dict = pickle.load(f)
        return ret_dict

    def _get_src_permutation_idx(self, indices):
        # permute predictions following indices
        batch_idx = torch.cat([torch.full_like(src, i) for i, (src, _) in enumerate(indices)])
        src_idx = torch.cat([src for (src, _) in indices])
        return batch_idx, src_idx

    def _get_src_single_permutation_idx(self, indices, index):
        ## Only need the src query index selection from this function for attention feature selection
        batch_idx = [torch.full_like(src, i) for i, src in enumerate(indices)][0]
        src_idx = indices[0]
        return batch_idx, src_idx

    def _get_tgt_permutation_idx(self, indices):
        # permute targets following indices
        batch_idx = torch.cat([torch.full_like(tgt, i) for i, (_, tgt) in enumerate(indices)])
        tgt_idx = torch.cat([tgt for (_, tgt) in indices])
        return batch_idx, tgt_idx

    def get_loss(self, loss, outputs, targets, indices, num_boxes, epoch, owod_targets, owod_indices, **kwargs):
        loss_map = {
            'labels': self.loss_labels,
            'NC_labels': self.loss_NC_labels,
            'cardinality': self.loss_cardinality,
            'boxes': self.loss_boxes,
            'masks': self.loss_masks
        }
        assert loss in loss_map, f'do you really want to compute {loss} loss?'
        return loss_map[loss](outputs, targets, indices, num_boxes, epoch, owod_targets, owod_indices, **kwargs)

    def forward(self, samples, outputs, targets, epoch):
        """ This performs the loss computation.
        Parameters:
             outputs: dict of tensors, see the output specification of the model for the format
             targets: list of dicts, such that len(targets) == batch_size.
                      The expected keys in each dict depends on the losses applied, see each loss' doc
        """
        if self.nc_epoch > 0:
            loss_epoch = 9
        else:
            loss_epoch = 0

        outputs_without_aux = {k: v for k, v in outputs.items() if k != 'aux_outputs' and k != 'enc_outputs'}
        indices = self.matcher(outputs_without_aux, targets)

        owod_targets = deepcopy(targets)
        owod_indices = deepcopy(indices)

        owod_outputs = outputs_without_aux.copy()
        owod_device = owod_outputs["pred_boxes"].device

        if self.unmatched_boxes and epoch >= loss_epoch:
            ## get pseudo unmatched boxes from this section
            res_feat = torch.mean(outputs['resnet_1024_feat'], 1)
            queries = torch.arange(outputs['pred_logits'].shape[1])
            for i in range(len(indices)):
                combined = torch.cat(
                    (queries, self._get_src_single_permutation_idx(indices[i], i)[-1]))  ## need to fix the indexing
                uniques, counts = combined.unique(return_counts=True)
                unmatched_indices = uniques[counts == 1]
                boxes = outputs_without_aux['pred_boxes'][i]  # [unmatched_indices,:]
                img = samples.tensors[i].cpu().permute(1, 2, 0).numpy()
                h, w = img.shape[:-1]
                img_w = torch.tensor(w, device=owod_device)
                img_h = torch.tensor(h, device=owod_device)
                unmatched_boxes = box_ops.box_cxcywh_to_xyxy(boxes)
                unmatched_boxes = unmatched_boxes * torch.tensor([img_w, img_h, img_w, img_h], dtype=torch.float32).to(
                    owod_device)
                means_bb = torch.zeros(queries.shape[0]).to(unmatched_boxes)
                bb = unmatched_boxes
                for j, _ in enumerate(means_bb):
                    if j in unmatched_indices:
                        upsaple = nn.Upsample(size=(img_h, img_w), mode='bilinear')
                        img_feat = upsaple(res_feat[i].unsqueeze(0).unsqueeze(0))
                        img_feat = img_feat.squeeze(0).squeeze(0)
                        xmin = bb[j, :][0].long()
                        ymin = bb[j, :][1].long()
                        xmax = bb[j, :][2].long()
                        ymax = bb[j, :][3].long()
                        means_bb[j] = torch.mean(img_feat[ymin:ymax, xmin:xmax])
                        if torch.isnan(means_bb[j]):
                            means_bb[j] = -10e10
                    else:
                        means_bb[j] = -10e10

                _, topk_inds = torch.topk(means_bb, self.top_unk)
                topk_inds = torch.as_tensor(topk_inds)

                topk_inds = topk_inds.cpu()

                unk_label = torch.as_tensor([self.num_classes - 1], device=owod_device)
                owod_targets[i]['labels'] = torch.cat(
                    (owod_targets[i]['labels'], unk_label.repeat_interleave(self.top_unk)))
                owod_indices[i] = (torch.cat((owod_indices[i][0], topk_inds)), torch.cat(
                    (owod_indices[i][1], (owod_targets[i]['labels'] == unk_label).nonzero(as_tuple=True)[0].cpu())))

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
            losses.update(
                self.get_loss(loss, outputs, targets, indices, num_boxes, epoch, owod_targets, owod_indices, **kwargs))

        # In case of auxiliary losses, we repeat this process with the output of each intermediate layer.
        if 'aux_outputs' in outputs:
            for i, aux_outputs in enumerate(outputs['aux_outputs']):
                indices = self.matcher(aux_outputs, targets)

                owod_targets = deepcopy(targets)
                owod_indices = deepcopy(indices)

                aux_owod_outputs = aux_outputs.copy()
                owod_device = aux_owod_outputs["pred_boxes"].device

                if self.unmatched_boxes and epoch >= loss_epoch:
                    ## get pseudo unmatched boxes from this section
                    res_feat = torch.mean(outputs['resnet_1024_feat'], 1)  # 2 X 67 X 50
                    queries = torch.arange(aux_owod_outputs['pred_logits'].shape[1])
                    for i in range(len(indices)):
                        combined = torch.cat((queries, self._get_src_single_permutation_idx(indices[i], i)[
                            -1]))  ## need to fix the indexing
                        uniques, counts = combined.unique(return_counts=True)
                        unmatched_indices = uniques[counts == 1]
                        boxes = aux_owod_outputs['pred_boxes'][i]  # [unmatched_indices,:]
                        img = samples.tensors[i].cpu().permute(1, 2, 0).numpy()
                        h, w = img.shape[:-1]
                        img_w = torch.tensor(w, device=owod_device)
                        img_h = torch.tensor(h, device=owod_device)
                        unmatched_boxes = box_ops.box_cxcywh_to_xyxy(boxes)
                        unmatched_boxes = unmatched_boxes * torch.tensor([img_w, img_h, img_w, img_h],
                                                                         dtype=torch.float32).to(owod_device)
                        means_bb = torch.zeros(queries.shape[0]).to(
                            unmatched_boxes)  # torch.zeros(unmatched_boxes.shape[0])
                        bb = unmatched_boxes
                        ## [INFO]: iterating over the full list of boxes and then selecting the unmatched ones
                        for j, _ in enumerate(means_bb):
                            if j in unmatched_indices:
                                upsaple = nn.Upsample(size=(img_h, img_w), mode='bilinear')
                                img_feat = upsaple(res_feat[i].unsqueeze(0).unsqueeze(0))
                                img_feat = img_feat.squeeze(0).squeeze(0)
                                xmin = bb[j, :][0].long()
                                ymin = bb[j, :][1].long()
                                xmax = bb[j, :][2].long()
                                ymax = bb[j, :][3].long()
                                means_bb[j] = torch.mean(img_feat[ymin:ymax, xmin:xmax])
                                if torch.isnan(means_bb[j]):
                                    means_bb[j] = -10e10
                            else:
                                means_bb[j] = -10e10

                        _, topk_inds = torch.topk(means_bb, self.top_unk)
                        topk_inds = torch.as_tensor(topk_inds)

                        topk_inds = topk_inds.cpu()
                        unk_label = torch.as_tensor([self.num_classes - 1], device=owod_device)
                        owod_targets[i]['labels'] = torch.cat(
                            (owod_targets[i]['labels'], unk_label.repeat_interleave(self.top_unk)))
                        owod_indices[i] = (torch.cat((owod_indices[i][0], topk_inds)), torch.cat((owod_indices[i][1], (
                                    owod_targets[i]['labels'] == unk_label).nonzero(as_tuple=True)[0].cpu())))

                for loss in self.losses:
                    if loss == 'masks':
                        # Intermediate masks losses are too costly to compute, we ignore them.
                        continue
                    kwargs = {}
                    if loss == 'labels':
                        # Logging is enabled only for the last layer
                        kwargs['log'] = False
                    l_dict = self.get_loss(loss, aux_outputs, targets, indices, num_boxes, epoch, owod_targets,
                                           owod_indices, **kwargs)
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
        boxes = torch.gather(boxes, 1, topk_boxes.unsqueeze(-1).repeat(1, 1, 4))

        # and from relative [0, 1] to absolute [0, height] coordinates
        img_h, img_w = target_sizes.unbind(1)
        scale_fct = torch.stack([img_w, img_h, img_w, img_h], dim=1)
        boxes = boxes * scale_fct[:, None, :]

        results = [{'scores': s, 'labels': l, 'boxes': b} for s, l, b in zip(scores, labels, boxes)]

        return results


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


def build(args):
    num_classes = args.num_classes
    print(num_classes)
    if args.dataset == "coco_panoptic":
        num_classes = 250
    device = torch.device(args.device)

    backbone = build_backbone(args)

    transformer = build_deforamble_transformer(args)

    prev_intro_cls = args.PREV_INTRODUCED_CLS
    curr_intro_cls = args.CUR_INTRODUCED_CLS
    seen_classes = prev_intro_cls + curr_intro_cls
    invalid_cls_logits = list(
        range(seen_classes, num_classes - 1))  # unknown class indx will not be included in the invalid class range
    print("Invalid class rangw: " + str(invalid_cls_logits))

    model = DeformableDETR(
        backbone,
        transformer,
        num_classes=num_classes,
        num_queries=args.num_queries,
        num_feature_levels=args.num_feature_levels,
        aux_loss=args.aux_loss,
        with_box_refine=args.with_box_refine,
        two_stage=args.two_stage,
        unmatched_boxes=args.unmatched_boxes,
        novelty_cls=args.NC_branch,
        featdim=args.featdim,
    )
    if args.masks:
        model = DETRsegm(model, freeze_detr=(args.frozen_weights is not None))
    matcher = build_matcher(args)

    weight_dict = {'loss_ce': args.cls_loss_coef, 'loss_bbox': args.bbox_loss_coef}
    if args.NC_branch:
        weight_dict = {'loss_ce': args.cls_loss_coef, 'loss_NC': args.nc_loss_coef, 'loss_bbox': args.bbox_loss_coef}

    weight_dict['loss_giou'] = args.giou_loss_coef
    if args.masks:
        weight_dict["loss_mask"] = args.mask_loss_coef
        weight_dict["loss_dice"] = args.dice_loss_coef
    # TODO this is a hack
    if args.aux_loss:
        aux_weight_dict = {}
        for i in range(args.dec_layers - 1):
            aux_weight_dict.update({k + f'_{i}': v for k, v in weight_dict.items()})
        aux_weight_dict.update({k + f'_enc': v for k, v in weight_dict.items()})
        weight_dict.update(aux_weight_dict)

    losses = ['labels', 'boxes', 'cardinality']
    if args.NC_branch:
        losses = ['labels', 'NC_labels', 'boxes', 'cardinality']
    if args.masks:
        losses += ["masks"]
    criterion = SetCriterion(args, num_classes, matcher, weight_dict, losses, invalid_cls_logits,
                             focal_alpha=args.focal_alpha)
    criterion.to(device)
    postprocessors = {'bbox': PostProcess()}
    if args.masks:
        postprocessors['segm'] = PostProcessSegm()
        if args.dataset == "coco_panoptic":
            is_thing_map = {i: i <= 90 for i in range(201)}
            postprocessors["panoptic"] = PostProcessPanoptic(is_thing_map, threshold=0.85)

    return model, criterion, postprocessors