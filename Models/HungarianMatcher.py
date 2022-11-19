import torch
from scipy.optimize import linear_sum_assignment
from torch import nn

from util.box_ops import box_cxcywh_to_xyxy, generalized_box_iou


class HungarianMatcher(nn.Module):
    """
    This class computes an assignment between the targets and the predictions of the network
    For efficiency reasons, the targets don't include the no_object. Because of this, in general,
    there are more predictions than targets. In this case, we do a 1-to-1 matching of the best predictions,
    while the others are un-matched (and thus treated as non-objects).
    """

    def __init__(self,
                 cost_class: float = 1,
                 cost_bbox: float = 1,
                 cost_giou: float = 1):
        """
        :param cost_class: weight of the classification error
        :param cost_bbox: weight of the L1 error of the bounding box coordinates in the matching cost
        :param cost_giou: weight of the giou loss of the bounding box in the matching cost
        """
        super().__init__()
        self.cost_class = cost_class
        self.cost_bbox = cost_bbox
        self.cost_giou = cost_giou
        assert cost_class != 0 or cost_bbox != 0 or cost_giou != 0, "all costs cant be 0"

    def forward(self, outputs, targets):
        """
        Performs the matching between outputs and targets.
        :param outputs: A dict that contains all the proposals in 1 batch:
               "pred_logits": Tensor of dim [bs, n_queries, n_classes] with the classification category
               "pred_boxes": Tensor of dim [bs, n_queries, 4] with the predicted box coordinates
               dict{"pred_logits":list[tensor, tensor, ..., tensor], "pred_boxes":list[tensor, tensor, ..., tensor]}

        :param targets: A list that contains targets dict of img in 1 batch, where each targets dict contains:
               "labels": Tensor of dim [n_target_boxes] with the class labels
               "boxes": Tensor of dim [n_target_boxes, 4] with the target box coordinates
               list[dict{"labels","boxes"}, dict{"labels","boxes"}, ..., dict{"labels","boxes"}]

        :return:A list that contains tuples(index_i, index_j) of img in 1 batch, where each tuple contains:
                -index_i is Tensor of dim n_target_boxes[indices of the selected predictions (in order)]
                -index_j is Tensor of dim n_target_boxes[indices of the corresponding selected targets (in order)]
                For each img in 1 batch, it holds: len(index_i) = len(index_j) = min(n_queries, n_target_boxes)
                list[(index_i,index_j),(index_i,index_j),...,(index_i,index_j)]
        """
        # 计算loss，禁用梯度
        with torch.no_grad():
            bs, num_queries = outputs["pred_logits"].shape[:2]

            # We flatten to compute the cost matrices in a batch
            # out_prob:(b*n_queries,n_classes)
            # out_bbox:(b*n_queries,4)
            out_prob = outputs["pred_logits"].flatten(0, 1).sigmoid()
            out_bbox = outputs["pred_boxes"].flatten(0, 1)  # [batch_size * num_queries, 4]

            # Also concat the target labels and boxes
            # tgt_ids:(b*n_targets,1)
            # tgt_bbox:(b*n_targets,4)
            tgt_ids = torch.cat([v["labels"] for v in targets])
            tgt_bbox = torch.cat([v["boxes"] for v in targets])

            # Compute the classification cost.
            alpha = 0.25
            gamma = 2.0
            neg_cost_class = (1 - alpha) * (out_prob ** gamma) * (-(1 - out_prob + 1e-8).log())
            pos_cost_class = alpha * ((1 - out_prob) ** gamma) * (-(out_prob + 1e-8).log())
            cost_class = pos_cost_class[:, tgt_ids] - neg_cost_class[:, tgt_ids]

            # Compute the L1 cost between boxes
            cost_bbox = torch.cdist(out_bbox, tgt_bbox, p=1)

            # Compute the giou cost betwen boxes
            cost_giou = -generalized_box_iou(box_cxcywh_to_xyxy(out_bbox),
                                             box_cxcywh_to_xyxy(tgt_bbox))

            # Final cost matrix
            # C:(b,n_queries,sum(n_targets) in the batch) 二分图匹配的开销矩阵
            C = self.cost_bbox * cost_bbox + self.cost_class * cost_class + self.cost_giou * cost_giou
            C = C.view(bs, num_queries, -1).cpu()

            # sizes:(n_targets)
            sizes = [len(v["boxes"]) for v in targets]
            # 将总开销矩阵划分为batch中每个img的cost矩阵:c[i]:(n_queries,n_targets),利用匈牙利算法求解最佳分配(i,j)
            indices = [linear_sum_assignment(c[i]) for i, c in enumerate(C.split(sizes, -1))]
            # i:query索引;j:target索引
            return [(torch.as_tensor(i, dtype=torch.int64), torch.as_tensor(j, dtype=torch.int64)) for i, j in indices]


def build_matcher(args):
    return HungarianMatcher(cost_class=args.set_cost_class,
                            cost_bbox=args.set_cost_bbox,
                            cost_giou=args.set_cost_giou)
