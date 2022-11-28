import torch
import torch.nn as nn
from util import box_ops

def update_pseudo_labels(samples, outputs, targets, indices, device, top_unk, num_classes):
    # feature heatmap，表示每个像素在所有通道上的平均激活强度，of size(bs,h,w)
    res_feat = torch.mean(outputs['resnet_1024_feat'], 1)
    # 根据一张img的query数量来生成从0到99的100个query
    all_indices = torch.arange(outputs['pred_logits'].shape[1])

    # i:img index in a batch
    for i in range(len(indices)):
        # 求所有query和matched query的差集，即unmatched query的索引
        matched_indices = indices[i][0]
        combined = torch.cat((all_indices, matched_indices))
        uniques, counts = combined.unique(return_counts=True)
        unmatched_indices = uniques[counts == 1]

        # boxes: the coordinates of each bbox in the ith img, of size(n_queries,4)
        boxes = outputs['pred_boxes'][i]
        img = samples.tensors[i].cpu().permute(1, 2, 0).numpy()
        h, w = img.shape[:-1]
        img_w = torch.tensor(w, device=device)
        img_h = torch.tensor(h, device=device)
        # 将bbox的坐标(x,y,h,w)转换为左下角(x,y)和右上角坐标(x,y)
        unmatched_boxes = box_ops.box_cxcywh_to_xyxy(boxes)
        # 将归一化的坐标乘img的宽和高,映射到实际坐标
        unmatched_boxes = unmatched_boxes * torch.tensor([img_w, img_h, img_w, img_h], dtype=torch.float32).to(
            device)
        # 生成n_queries长度的零向量
        box_mean_activation = torch.zeros(all_indices.shape[0]).to(unmatched_boxes)

        for indice, _ in enumerate(box_mean_activation):
            if indice in unmatched_indices:
                feature_map = res_feat[i].unsqueeze(0).unsqueeze(0)
                # 上采样网络, 对feature heatmap线性插值扩大到img的分辨率, 输出分辨率(img_h, img_w)
                upsample = nn.Upsample(size=(img_h, img_w), mode='bilinear')
                # 扩张为size:(bs,c,h,w)->img_feat:(1,1,h,w),进行线性插值
                img_features = upsample(feature_map)
                # img_feat:(img_h,img_w)
                img_features = img_features.squeeze(0).squeeze(0)
                # bbox左下角和右上角的坐标
                x_min = unmatched_boxes[indice, :][0].long()
                y_min = unmatched_boxes[indice, :][1].long()
                x_max = unmatched_boxes[indice, :][2].long()
                y_max = unmatched_boxes[indice, :][3].long()
                # 对于匹配到的bbox，计算每张图片feature heatmap中bbox左下角和右上角之间的平均激活强度
                box_mean_activation[indice] = torch.mean(img_features[y_min:y_max, x_min:x_max])
                if torch.isnan(box_mean_activation[indice]):
                    box_mean_activation[indice] = -10e10
            else:
                # 对于未匹配到的bbox,平均激活强度记为极大负值
                box_mean_activation[indice] = -10e10

        # 选取平均激活强度最大的前k个未匹配bbox,topl_inds存储0-99之间的bbox索引
        _, topk_inds = torch.topk(box_mean_activation, top_unk)
        topk_inds = torch.as_tensor(topk_inds)
        topk_inds = topk_inds.cpu()

        # 将unknown obj bbox的label设为最后一类
        unk_label = torch.as_tensor([num_classes - 1], device=device)
        # 增加top_unk个pseudo labels
        targets[i]['labels'] = torch.cat(
            (targets[i]['labels'], unk_label.repeat_interleave(top_unk)))
        # 扩展这张img的匹配索引,index_i和index_j分别增加top_unk个索引
        indices[i] = (torch.cat((indices[i][0], topk_inds)), torch.cat(
            (indices[i][1], (targets[i]['labels'] == unk_label).nonzero(as_tuple=True)[0].cpu())))

        return targets, indices

"""
            # feature heatmap，表示每个像素在所有通道上的平均激活强度，of size(bs,h,w)
            res_feat = torch.mean(outputs['resnet_1024_feat'], 1)
            # 根据一张img的query数量来生成从0到99的100个query
            all_indices = torch.arange(outputs['pred_logits'].shape[1])

            # i:img index in a batch, 循环处理每张图片
            for i in range(len(indices)):
                # 求所有query和matched query的差集，即unmatched query的索引
                matched_indices = indices[i][0]
                combined = torch.cat((all_indices, matched_indices))
                uniques, counts = combined.unique(return_counts=True)
                unmatched_indices = uniques[counts == 1]

                # boxes: the coordinates of each bbox in the ith img, of size(n_queries,4)
                boxes = outputs_without_aux['pred_boxes'][i]
                img = samples.tensors[i].cpu().permute(1, 2, 0).numpy()
                h, w = img.shape[:-1]
                img_w = torch.tensor(w, device=owod_device)
                img_h = torch.tensor(h, device=owod_device)
                # 将bbox的坐标(x,y,h,w)转换为左下角(x,y)和右上角坐标(x,y)
                unmatched_boxes = box_ops.box_cxcywh_to_xyxy(boxes)
                unmatched_boxes = unmatched_boxes * torch.tensor([img_w, img_h, img_w, img_h], dtype=torch.float32).to(
                    owod_device)
                # 生成n_queries长度的零向量
                means_bb = torch.zeros(all_indices.shape[0]).to(unmatched_boxes)

                bb = unmatched_boxes
                for j, _ in enumerate(means_bb):
                    if j in unmatched_indices:
                        # 上采样网络, 对feature heatmap线性插值扩大到img的分辨率, 输出分辨率(img_h, img_w)
                        upsample = nn.Upsample(size=(img_h, img_w), mode='bilinear')
                        # 扩张为size:(bs,c,h,w)->img_feat:(1,1,h,w),进行线性插值
                        img_feat = upsample(res_feat[i].unsqueeze(0).unsqueeze(0))
                        # img_feat:(img_h,img_w)
                        img_feat = img_feat.squeeze(0).squeeze(0)
                        # bbox左下角和右上角的坐标
                        xmin = bb[j, :][0].long()
                        ymin = bb[j, :][1].long()
                        xmax = bb[j, :][2].long()
                        ymax = bb[j, :][3].long()
                        # 对于匹配到的bbox，计算每张图片feature heatmap中bbox左下角和右上角之间的平均激活强度
                        means_bb[j] = torch.mean(img_feat[ymin:ymax, xmin:xmax])
                        if torch.isnan(means_bb[j]):
                            means_bb[j] = -10e10
                    else:
                        # 对于未匹配到的bbox,平均激活强度记为极大负值
                        means_bb[j] = -10e10

                # 选取平均激活强度最大的前k个未匹配bbox,topl_inds存储0-99之间的bbox索引
                _, topk_inds = torch.topk(means_bb, self.top_unk)
                topk_inds = torch.as_tensor(topk_inds)
                topk_inds = topk_inds.cpu()

                # 将unknown obj bbox的label设为最后一类
                unk_label = torch.as_tensor([self.num_classes - 1], device=owod_device)
                # 增加top_unk个pseudo labels
                owod_targets[i]['labels'] = torch.cat(
                    (owod_targets[i]['labels'], unk_label.repeat_interleave(self.top_unk)))
                # 扩展这张img的匹配索引,index_i和index_j分别增加top_unk个索引
                owod_indices[i] = (torch.cat((owod_indices[i][0], topk_inds)), torch.cat(
                    (owod_indices[i][1], (owod_targets[i]['labels'] == unk_label).nonzero(as_tuple=True)[0].cpu())))
"""