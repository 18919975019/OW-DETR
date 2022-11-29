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
        super(FFN, self).__init__()
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
    def __init__(self, d_model, block, dropout):
        super(AddNorm, self).__init__()
        self.residual = Residual(block, dropout)
        self.norm = nn.LayerNorm(d_model)

    def forward(self, x, **kwargs):
        return self.norm(self.residual(x), **kwargs)

class EncoderLayer(nn.Module):
    def __init__(self,
                 d_model=256, d_fc=1024,
                 dropout=0.1, activation="relu",
                 n_levels=4, n_heads=8, n_points=4):
        super().__init__()
        self.self_attn = AddNorm(d_model, MSDeformAttn(d_model, n_levels, n_heads, n_points), dropout)
        self.ffn = AddNorm(d_model, FFN(d_model, d_fc, activation, dropout), dropout)

    @staticmethod
    def with_pos_embed(tensor, pos):
        return tensor if pos is None else tensor + pos

    def forward(self, src, pos, reference_points, spatial_shapes, level_start_index, mask=None):
        """
        :param src: img展平的多尺度features, of size[bs, lvl*h*w, d_model]
        :param pos: img展平的多尺度位置编码, of size[bs, lvl*h*w, d_model]
        :param mask: img展平的掩码矩阵, of size[bs, lvl*h*w]
        :param reference_points: 编码阶段的参考点, 像素中心点坐标在不同尺度下的坐标, of size[bs, h*w*lvl, lvl, 2]
        :param spatial_shapes: 多尺度特征图的大小, of size[lvl,2]
        :param level_start_index: 多尺度特征图的索引, ex.[0,4,12,28]

        :return: 单层encoder输出的隐层表示, of size[bs, lvl*h*w, d_model]
        """
        src = self.self_attn(self.with_pos_embed(src, pos), reference_points, src, spatial_shapes, level_start_index, mask)
        src = self.forward_ffn(src)
        return src

class _make_encoder(nn.Module):
    def __init__(self, encoder_layer, depth):
        super().__init__()
        self.layers = _get_clones(encoder_layer, depth)

    @staticmethod
    def get_reference_points(spatial_shapes, valid_ratios, device):
        """
        :param spatial_shapes:
        :param valid_ratios:
        :param device:
        :return: reference_points img像素中心点坐标,同一参考点在不同尺度下映射的不同的坐标, of size[bs, h*w*lvl, n_levels, 2]
                 range in [0, 1], top-left (0, 0), bottom-right (1, 1)
        """
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
            # of size[bs, h*w, 2]，保存了特征图上每个像素点的相对位置坐标对
            ref = torch.stack((ref_x, ref_y), -1)
            reference_points_list.append(ref)
        # of size[bs, h*w*lvl，2]
        reference_points = torch.cat(reference_points_list, 1)
        # of size[bs, h*w*lvl, lvl, 2], 同一参考点在不同尺度下对应不同的坐标
        reference_points = reference_points[:, :, None] * valid_ratios[:, None]
        return reference_points

    def forward(self, src, spatial_shapes, level_start_index, valid_ratios, pos=None, mask=None):
        """
        :param src:
        :param spatial_shapes:
        :param level_start_index:
        :param valid_ratios:
        :param pos:
        :param padding_mask:

        :return: 多层encoder输出的隐层表示, of size[bs, lvl*h*w, d_model]
        """
        output = src
        reference_points = self.get_reference_points(spatial_shapes, valid_ratios, device=src.device)
        for layer in self.layers:
            output = layer(output, pos, reference_points, spatial_shapes, level_start_index, mask)
        return output

class DecoderLayer(nn.Module):
    def __init__(self, d_model=256, d_fc=1024,
                 dropout=0.1, activation="relu",
                 n_levels=4, n_heads=8, n_points=4):
        super().__init__()

        self.self_attn = AddNorm(d_model, nn.MultiheadAttention(d_model, n_heads, dropout=dropout), dropout)
        self.cross_attn = AddNorm(d_model, MSDeformAttn(d_model, n_levels, n_heads, n_points), dropout)
        self.ffn = AddNorm(d_model, FFN(d_model, d_fc, activation, dropout), dropout)

    @staticmethod
    def with_pos_embed(tensor, pos):
        return tensor if pos is None else tensor + pos

    def forward(self, query_tgt, query_pos, reference_points, src, src_spatial_shapes, level_start_index, src_mask=None):
        """
        The queries first compute their self attention, and then the self attention representation is used to compute
        the cross attention with output of the Encoder.
        :param query_tgt: query的内容编码, of size[bs, n_queries, c=d_model]
        :param query_pos: query的位置编码, of size[bs, n_queries, c=d_model]
        :param reference_points: 解码阶段的参考点, 通过线性投影pos embed生成两个坐标来产生, of size[bs, n_queries, 2]
        :param src: Encoder输出的隐层表示, of size[bs, lvl*h*w, d_model]
        :param src_mask: backbone输出的掩码矩阵, of size[bs, lvl*h*w]
        :param src_spatial_shapes: 多尺度特征图的大小, of size[lvl,2]
        :param level_start_index: 多尺度特征图的索引, ex.[0,4,12,28]

        :return: 单层decoder输出的隐层表示, of size[bs, n_queries, d_model]
        """
        q = k = self.with_pos_embed(query_tgt, query_pos)
        query_tgt = self.self_attn(q.transpose(0, 1), k.transpose(0, 1), query_tgt.transpose(0, 1))[0].transpose(0, 1)
        query_tgt = self.cross_attn(self.with_pos_embed(query_tgt, query_pos), reference_points, src,
                              src_spatial_shapes, level_start_index, src_mask)
        query_tgt = self.ffn(query_tgt)
        return query_tgt


class _make_decoder(nn.Module):
    def __init__(self, DecoderLayer, n_layers, return_intermediate=False):
        super().__init__()
        self.layers = _get_clones(DecoderLayer, n_layers)
        self.n_layers = n_layers
        self.return_intermediate = return_intermediate
        self.bbox_embed = None

    # DecoderLayer:forward(tgt, reference_points, memory,spatial_shapes, level_start_index, valid_ratios, query_embed, mask_flatten)
    def forward(self, query_tgt, reference_points, src, src_spatial_shapes, src_level_start_index, src_valid_ratios,
                query_pos=None, src_mask=None):
        """
        :param query_tgt: query的内容编码, of size[bs, n_queries, c=d_model]
        :param query_pos: query的位置编码, of size[bs, n_queries, c=d_model]
        :param reference_points: 解码阶段的参考点, 通过线性投影pos embed生成两个坐标来产生, of size[bs, n_queries, 2]
        :param src: Encoder输出的隐层表示, of size[bs, lvl*h*w, d_model]
        :param src_mask: backbone输出的掩码矩阵, of size[bs, lvl*h*w]
        :param src_spatial_shapes: 多尺度特征图的大小, of size[lvl,2]
        :param level_start_index: 多尺度特征图的索引, ex.[0,4,12,28]
        :param src_valid_ratios:

        :return output: queries经过decoder解码后输出的隐层表示, of size[n_layer,bs,n_queries,c=d_model]
                reference_point: 用decoder每层输出预测bbox坐标,优化下一层的参考点坐标, of size[n_layer,bs,n_queries,4]
        """
        output = query_tgt
        # 存储中间结果
        intermediate = []
        intermediate_reference_points = []
        for layer_index, layer in enumerate(self.layers):
            # 从第二层开始, reference points 为上一层的bonding box:(x,y,h,w)
            if reference_points.shape[-1] == 4:
                reference_points_input = reference_points[:, :, None] \
                                         * torch.cat([src_valid_ratios, src_valid_ratios], -1)[:, None]
            # 第一层,reference_points为query_pos预测的点坐标:(x,y)
            else:
                assert reference_points.shape[-1] == 2
                # reference_points:(bs,n_queries,2); valid_ratios:(bs,lvl,2)
                reference_points_input = reference_points[:, :, None] * src_valid_ratios[:, None]

            output = layer(output, query_pos, reference_points_input, src, src_spatial_shapes, src_level_start_index,
                           src_mask)

            if self.bbox_embed is not None:
                tmp = self.bbox_embed[layer_index](output) # 提取这一层decoder输出的hidden，根据hidden预测一个bonding box坐标
                # 第二层开始
                if reference_points.shape[-1] == 4:
                    # 将上一层预测的bbox坐标作为reference point, 和这一层的box的中心坐标相加
                    new_reference_points = tmp + inverse_sigmoid(reference_points)
                    new_reference_points = new_reference_points.sigmoid()
                # 第一层
                else:
                    assert reference_points.shape[-1] == 2
                    new_reference_points = tmp
                    new_reference_points[..., :2] = tmp[..., :2] + inverse_sigmoid(reference_points)
                    new_reference_points = new_reference_points.sigmoid()
                # 将这一层预测的bonding box作为下一层的reference points
                reference_points = new_reference_points.detach()

            # 保存每一层decoder的输出
            if self.return_intermediate:
                intermediate.append(output)
                intermediate_reference_points.append(reference_points)

        if self.return_intermediate:
            return torch.stack(intermediate), torch.stack(intermediate_reference_points)

        return output, reference_points


