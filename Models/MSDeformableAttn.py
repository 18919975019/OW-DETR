from __future__ import absolute_import
from __future__ import print_function
from __future__ import division
import warnings
import math
from einops import rearrange
import torch
from torch import nn
import torch.nn.functional as F
from torch.nn.init import xavier_uniform_, constant_


def _is_power_of_2(n):
    if (not isinstance(n, int)) or (n < 0):
        raise ValueError("invalid input for _is_power_of_2: {} (type: {})".format(n, type(n)))
    return (n & (n-1) == 0) and n != 0


class MSDeformAttn(nn.Module):
    def __init__(self, d_model=256, n_levels=4, n_heads=8, n_points=4):
        super().__init__()

        if d_model % n_heads != 0:
            raise ValueError('d_model must be divisible by n_heads, but got {} and {}'.format(d_model, n_heads))
        self.d_head = d_model // n_heads
        if not _is_power_of_2(self.d_head):
            warnings.warn("You'd better set d_model in MSDeformAttn to make the dimension of each attention head a power of 2 "
                          "which is more efficient in our CUDA implementation.")

        self.im2col_step = 64

        self.d_model = d_model
        self.n_levels = n_levels
        self.n_heads = n_heads
        self.n_points = n_points

        # 全连接层：用于为每一个参考点【在多个注意力头、多个尺度的特征图上】采样n对offset（x轴/y轴的offset）,输出维度为（头数*尺度数*采样数*2）
        self.sampling = nn.Linear(d_model, n_heads * n_levels * n_points * 2)
        # 全连接层：用于为每一个参考点【在多个注意力头、多个尺度的特征图上】的n个采样点学习注意力得分，输出维度为（头数*尺度数*采样数）
        self.attention_weights = nn.Linear(d_model, n_heads * n_levels * n_points)
        # 全连接层：对参考点在一个注意力头的输入向量进行投影，得到参考点的value向量
        self.v = nn.Linear(d_model, d_model)
        # 全连接层：对参考点在一个注意力头的输出向量进行投影，得到参考点的输出向量
        self.fc = nn.Linear(d_model, d_model)

        # 初始化网络的权重和偏置
        self._reset_parameters()

    def _reset_parameters(self):
        # 将采样offset的网络的权重初始化为0，计算并固定每个采样点的偏置
        constant_(self.sampling.weight.data, 0.)
        # theta = 2pi * (head_id/n_heads)
        thetas = torch.arange(self.n_heads, dtype=torch.float32) * (2.0 * math.pi / self.n_heads)
        # 为每一个注意力头计算[cos theta, sin theta],grid_init.size:(注意力头数 * 2（一对三角函数）)
        grid_init = torch.stack([thetas.cos(), thetas.sin()], -1)
        # 归一化每个注意力头的三角函数对（除以所有头中绝对值最大的三角函数值），并把每个头的三角函数对复制到每个尺度、每个采样点
        # grid_init.size:(注意力头数 * 尺度数 * 采样点的个数 * 2)
        grid_init = (grid_init / grid_init.abs().max(-1, keepdim=True)[0]).view(self.n_heads, 1, 1, 2).repeat(1, self.n_levels, self.n_points, 1)
        # 对采样点1，2，3，分别为它们的三角函数对乘上1，2，3，使得三个采样点的初始偏置不同
        for i in range(self.n_points):
            grid_init[:, :, i, :] *= i + 1
        # 将每个注意力头、每个尺度上的所有采样点的三角函数对展开成一列作为采样offset的网络的权重的偏置，并且这些偏置没有梯度，在反向传播中不修改值
        # 每个头的每个尺度的第i个采样点的偏置相同
        with torch.no_grad():
            self.sampling.bias = nn.Parameter(grid_init.view(-1))

        # 将学习注意力得分的网络的权重、偏置初始化为0
        constant_(self.attention_weights.weight.data, 0.)
        constant_(self.attention_weights.bias.data, 0.)

        # 将投影得到value的网络的偏置初始化为0，权重初始化为均匀分布，使得该层输入和输出的方差相近
        xavier_uniform_(self.v.weight.data)
        constant_(self.v.bias.data, 0.)

        # 将投影得到注意力头输出的网络的偏置初始化为0，权重初始化为均匀分布，使得该层输入和输出的方差相近
        xavier_uniform_(self.fc.weight.data)
        constant_(self.fc.bias.data, 0.)

    @staticmethod
    def attention(value, value_spatial_shapes, sampling_locations, attention_weights):
        # for debug and test only,
        # need to use cuda version instead
        N_, S_, M_, D_ = value.shape
        _, Lq_, M_, L_, P_, _ = sampling_locations.shape
        value_list = value.split([H_ * W_ for H_, W_ in value_spatial_shapes], dim=1)
        sampling_grids = 2 * sampling_locations - 1
        sampling_value_list = []
        for lid_, (H_, W_) in enumerate(value_spatial_shapes):
            # N_, H_*W_, M_, D_ -> N_, H_*W_, M_*D_ -> N_, M_*D_, H_*W_ -> N_*M_, D_, H_, W_
            value_l_ = value_list[lid_].flatten(2).transpose(1, 2).reshape(N_ * M_, D_, H_, W_)
            # N_, Lq_, M_, P_, 2 -> N_, M_, Lq_, P_, 2 -> N_*M_, Lq_, P_, 2
            sampling_grid_l_ = sampling_grids[:, :, :, lid_].transpose(1, 2).flatten(0, 1)
            # N_*M_, D_, Lq_, P_
            sampling_value_l_ = F.grid_sample(value_l_, sampling_grid_l_,
                                              mode='bilinear', padding_mode='zeros', align_corners=False)
            sampling_value_list.append(sampling_value_l_)
        # (N_, Lq_, M_, L_, P_) -> (N_, M_, Lq_, L_, P_) -> (N_, M_, 1, Lq_, L_*P_)
        attention_weights = attention_weights.transpose(1, 2).reshape(N_ * M_, 1, Lq_, L_ * P_)
        output = (torch.stack(sampling_value_list, dim=-2).flatten(-2) * attention_weights).sum(-1).view(N_, M_ * D_,
                                                                                                         Lq_)
        return output.transpose(1, 2).contiguous()

    def forward(self, query, reference_points, input_flatten, input_spatial_shapes, input_level_start_index, input_padding_mask=None):

        # N:batch_size
        # Length_{query}:
        # C:channel
        # Len_in: h * w * n_levels,所有尺度的所有像素展开成一行

        """
        :param query                       (N, Length_{query}, C)
        :param reference_points            (N, Length_{query}, n_levels, 2), range in [0, 1], top-left (0,0), bottom-right (1, 1), including padding area
                                        or (N, Length_{query}, n_levels, 4), add additional (w, h) to form reference boxes
        :param input_flatten               (N, \sum_{l=0}^{L-1} H_l \cdot W_l, C)
        :param input_spatial_shapes        (n_levels, 2), [(H_0, W_0), (H_1, W_1), ..., (H_{L-1}, W_{L-1})]
        :param input_level_start_index     (n_levels, ), [0, H_0*W_0, H_0*W_0+H_1*W_1, H_0*W_0+H_1*W_1+H_2*W_2, ..., H_0*W_0+H_1*W_1+...+H_{L-1}*W_{L-1}]
        :param input_padding_mask          (N, \sum_{l=0}^{L-1} H_l \cdot W_l), True for padding elements, False for non-padding elements

        :return output                     (N, Length_{query}, C)
        """
        b, num_queries, c = query.shape
        b, num_inputs, c = input_flatten.shape
        # 确保图像输入向量包含每个尺度的每个像素
        assert (input_spatial_shapes[:, 0] * input_spatial_shapes[:, 1]).sum() == num_inputs

        # 将图像输入通过self.value_proj线性层投影成value向量
        value = self.v(input_flatten)

        # 如果掩码矩阵不为空，利用掩码过滤value向量
        if input_padding_mask is not None:
            value = value.masked_fill(input_padding_mask[..., None], float(0))
        # value.size(样本数 * pixel_num * 头数 * 每一个头上分得的向量维度)
        value = rearrange(value, "b n (h d) -> b n h d ", h=self.n_heads)
        # sampling_offsets.size(样本数 * Len_q * 头数 * 尺度数 * 采样点个数 * 坐标维度（2）)
        sampling_offsets = self.sampling(query)
        sampling_offsets = rearrange(sampling_offsets, "b n (h l p o) -> b n h l p o",
                                     h=self.n_heads, l=self.n_levels, p=self.n_points, o=2)
        attention_weights = self.attention_weights(query)
        attention_weights = rearrange(attention_weights, "b n (h l p) -> b n h lp",
                                      h=self.n_heads, l=self.n_levels, p=self.n_points)
        attention_weights = F.softmax(attention_weights, -1)
        attention_weights = rearrange(attention_weights, "b n h (l p) -> b n h l p",
                                      h=self.n_heads, l=self.n_levels, p=self.n_points)
        # N, Len_q, n_heads, n_levels, n_points, 2
        # 如果reference_points是参考点，只包含两个坐标
        if reference_points.shape[-1] == 2:
            # 各个尺度的w，h参数，除以它们以归一化采样偏移sampling_offsets
            offset_normalizer = torch.stack([input_spatial_shapes[..., 1], input_spatial_shapes[..., 0]], -1)
            # 采样位置由参考点坐标+offset得到，size（样本数 * query个数 * 头数 * 尺度数 * 采样点个数 * 2）
            sampling_locations = reference_points[:, :, None, :, None, :] \
                                 + sampling_offsets / offset_normalizer[None, None, None, :, None, :]
        # 如果query是以参考点为中心的候选框，包含两个坐标、w、H四个参数
        # 采样点的偏移除以（采样数 * 采样框的高/宽 * 0.5）
        elif reference_points.shape[-1] == 4:
            sampling_locations = reference_points[:, :, None, :, None, :2] \
                                 + sampling_offsets / self.n_points * reference_points[:, :, None, :, None, 2:] * 0.5
        else:
            raise ValueError(
                'Last dim of reference_points must be 2 or 4, but get {} instead.'.format(reference_points.shape[-1]))
        output = self.attention(value, input_spatial_shapes, sampling_locations, attention_weights)
        output = self.fc(output)

        return output