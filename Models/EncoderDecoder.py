import copy
import torch
import torch.nn.functional as F
from torch import nn, Tensor
from .MSDeformableAttn import MSDeformAttn

def inverse_sigmoid(x, eps=1e-5):
    x = x.clamp(min=0, max=1)
    x1 = x.clamp(min=eps)
    x2 = (1 - x).clamp(min=eps)
    return torch.log(x1/x2)

def _get_clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for i in range(N)])

def _get_activation_fn(activation):
    """Return an activation function given a string"""
    if activation == "relu":
        return F.relu
    if activation == "gelu":
        return F.gelu
    if activation == "glu":
        return F.glu
    raise RuntimeError(F"activation should be relu/gelu, not {activation}.")

class FFN(nn.Module):
    def __init__(self, d_model, d_fc, activation, dropout):
        self.nn1 = nn.Linear(d_model, d_fc)
        self.af = _get_activation_fn(activation)
        self.do = nn.Dropout(dropout)
        self.nn2 = nn.Linear(d_fc, d_model)

    def forward(self, x):
        return self.nn2(self.do(self.af(self.nn1(x))))

class Residual(nn.Module):
    def __init__(self, fn, dropout):
        super().__init__()
        self.fn = fn
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, **kwargs):
        return self.dropout(self.fn(x, **kwargs)) + x

class AddNorm(nn.Module):
    def __init__(self, d_model, block):
        super(AddNorm, self).__init__()
        self.residual = Residual(block)
        self.norm = nn.LayerNormalize(d_model)

    def forward(self, x, **kwargs):
        return self.norm(self.residual(x), **kwargs)

class EncoderLayer(nn.Module):
    def __init__(self,
                 d_model=256, d_fc=1024,
                 dropout=0.1, activation="relu",
                 n_levels=4, n_heads=8, n_points=4):

        super().__init__()
        # Self-Attention Block, output:(b, h*w*lvl, d_model)
        self.self_attn = AddNorm(d_model, MSDeformAttn(d_model, n_levels, n_heads, n_points))
        # FFN Block, output:(b, h*w*lvl, d_model)
        self.ffn = AddNorm(d_model, FFN(d_model, d_fc, activation, dropout))

    @staticmethod
    def with_pos_embed(tensor, pos):
        return tensor if pos is None else tensor + pos

    def forward(self, src, pos, reference_points, spatial_shapes, level_start_index, padding_mask=None):
        src = self.self_attn(self.with_pos_embed(src, pos), reference_points, src, spatial_shapes, level_start_index, padding_mask)
        src = self.forward_ffn(src)
        return src

class _make_encoder(nn.Module):
    def __init__(self, encoder_layer, depth):
        super().__init__()
        self.layers = _get_clones(encoder_layer, depth)

    @staticmethod
    def get_reference_points(spatial_shapes, valid_ratios, device):
        # reference_points (N, Length_{query}, n_levels, 2), range in [0, 1], top-left (0,0), bottom-right (1, 1), including padding area
        reference_points_list = []
        for lvl, (H_, W_) in enumerate(spatial_shapes):
            # linesopace:生成一个start到end之间的间隔为step的一维张量
            # ref_y = [[0.5,..., 0.5],
            #          [1.5,..., 1.5],
            #          ,...,
            #          [H-0.5,...,H-0.5]]
            # ref_y（H_ * H_）共有H_行，其中每一行是W_个从0.5到H_ - 0.5的均匀值
            ref_y, ref_x = torch.meshgrid(torch.linspace(0.5, H_ - 0.5, H_, dtype=torch.float32, device=device),
                                          torch.linspace(0.5, W_ - 0.5, W_, dtype=torch.float32, device=device))
            # ref_y = ref_y.reshape(-1)[None]把 ref_y的H_ * W_个网格坐标放在第一维的一个数组里，外面再加一维度，size是（1, H_ * W_）
            # 通过不同尺度的掩码矩阵得到每张图片在不同尺度下宽和高的有效比率（不被mask的像素所占的整个特征图的比例）,valid_ratios.size(bs,lvl,2)
            # ref_y，ref_x为在该特征图上每个像素的中心点的坐标，除以有效高度，有效宽度（实际值 * 有效比例）来归一化到(0，1)之间
            ref_y = ref_y.reshape(-1)[None] / (valid_ratios[:, None, lvl, 1] * H_)
            ref_x = ref_x.reshape(-1)[None] / (valid_ratios[:, None, lvl, 0] * W_)
            # ref.size(bs, H_ * W_, 2)，保存了特征图上每个像素点的相对位置坐标对
            ref = torch.stack((ref_x, ref_y), -1)
            reference_points_list.append(ref)\
        # reference_points:(b, h*w*lvl，2)
        reference_points = torch.cat(reference_points_list, 1)
        # reference_points:(b, h*w*lvl, lvl, 2), 同一参考点在不同尺度下对应不同的坐标
        reference_points = reference_points[:, :, None] * valid_ratios[:, None]
        return reference_points

    def forward(self, src, spatial_shapes, level_start_index, valid_ratios, pos=None, padding_mask=None):
        output = src
        reference_points = self.get_reference_points(spatial_shapes, valid_ratios, device=src.device)
        for layer in self.layers:
            output = layer(output, pos, reference_points, spatial_shapes, level_start_index, padding_mask)
        return output

class DecoderLayer(nn.Module):
    def __init__(self, d_model=256, d_fc=1024,
                 dropout=0.1, activation="relu",
                 n_levels=4, n_heads=8, n_points=4):
        super().__init__()

        # Self-Attention Block, output:(b, h*w*lvl, d_model)
        self.self_attn = AddNorm(d_model, nn.MultiheadAttention(d_model, n_heads, dropout=dropout))
        # Cross-Attention Block, output:(b, h*w*lvl, d_model)
        self.cross_attn = AddNorm(d_model, MSDeformAttn(d_model, n_levels, n_heads, n_points))
        # FFN Block, output:(b, h*w*lvl, d_model)
        self.ffn = AddNorm(d_model, FFN(d_model, d_fc, activation, dropout))

    @staticmethod
    def with_pos_embed(tensor, pos):
        return tensor if pos is None else tensor + pos

    def forward(self, tgt, query_pos, reference_points, src, src_spatial_shapes, level_start_index, src_padding_mask=None):
        q = k = self.with_pos_embed(tgt, query_pos)
        tgt = self.self_attn(q.transpose(0, 1), k.transpose(0, 1), tgt.transpose(0, 1))[0].transpose(0, 1)
        tgt = self.cross_attn(self.with_pos_embed(tgt, query_pos), reference_points, src,
                              src_spatial_shapes, level_start_index, src_padding_mask)
        tgt = self.ffn(tgt)
        return tgt


class _make_decoder(nn.Module):
    def __init__(self, DecoderLayer, n_layers, return_intermediate=False):
        super().__init__()
        self.layers = _get_clones(DecoderLayer, n_layers)
        self.n_layers = n_layers
        self.return_intermediate = return_intermediate
        # hack implementation for iterative bounding box refinement and two-stage Deformable DETR
        self.bbox_embed = None
        self.class_embed = None

    # DecoderLayer:forward(tgt, reference_points, memory,spatial_shapes, level_start_index, valid_ratios, query_embed, mask_flatten)
    def forward(self, tgt, reference_points, src, src_spatial_shapes, src_level_start_index, src_valid_ratios,
                query_pos=None, src_padding_mask=None):
        output = tgt
        # 存储中间结果
        intermediate = []
        intermediate_reference_points = []
        for _, layer in enumerate(self.layers):
            # 从第二层开始, reference points 为上一层的bonding box:(x,y,h,w)
            if reference_points.shape[-1] == 4:
                reference_points_input = reference_points[:, :, None] \
                                         * torch.cat([src_valid_ratios, src_valid_ratios], -1)[:, None]
            # 第一层,reference_points为query_pos预测的点坐标:(x,y)
            else:
                assert reference_points.shape[-1] == 2
                # reference_points:(b,num_queries,2); valid_ratios:(b,lvl,2)
                reference_points_input = reference_points[:, :, None] * src_valid_ratios[:, None]

            output = layer(output, query_pos, reference_points_input, src, src_spatial_shapes, src_level_start_index,
                           src_padding_mask)

            if self.bbox_embed is not None:
                # query=[tgt,pos], tgt编码query的内容, pos编码query的位置
                # 提取这一层decoder输出的hidden，根据hidden预测一个bonding box
                # 将上一层hidden预测的bonding box作为reference point, 和这一层的box的中心坐标相加
                tmp = self.bbox_embed[_](output)
                # 第二层开始
                if reference_points.shape[-1] == 4:
                    new_reference_points = tmp + inverse_sigmoid(reference_points)
                    new_reference_points = new_reference_points.sigmoid()
                # 第一层
                else:
                    assert reference_points.shape[-1] == 2
                    new_reference_points = tmp
                    new_reference_points[..., :2] = tmp[..., :2] + inverse_sigmoid(reference_points)
                    new_reference_points = new_reference_points.sigmoid()
                # 将这一层预测的bonding box作为下一层的reference piints
                reference_points = new_reference_points.detach()

            # 保存每一层decoder的输出
            if self.return_intermediate:
                intermediate.append(output)
                intermediate_reference_points.append(reference_points)

        if self.return_intermediate:
            return torch.stack(intermediate), torch.stack(intermediate_reference_points)

        return output, reference_points


