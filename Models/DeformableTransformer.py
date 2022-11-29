import math
import torch
from torch import nn
from torch.nn.init import xavier_uniform_, constant_, uniform_, normal_
from einops import rearrange
from .MSDeformableAttn import MSDeformAttn

from .EncoderDecoder import EncoderLayer, DecoderLayer, _make_encoder, _make_decoder


class DeformableTransformer(nn.Module):
    def __init__(self, d_model=256, n_heads=8, d_fc=1024, n_levels=4,
                 dec_n_points=4, enc_n_points=4,
                 dropout=0.1, activation="relu",
                 num_encoder_layers=6, num_decoder_layers=6,
                 return_intermediate_dec=False,
                 ):
        """
        :param d_model: embedding的维度
        :param n_heads: 头数
        :param d_fc: 全连接层输出维度
        :param n_levels: 特征图的尺度数
        :param dec_n_points: 解码器参考点采样数
        :param enc_n_points: 编码器参考点采样数
        :param dropout: dropout的比率
        :param activation: 激活函数
        :param num_encoder_layers: 编码器层数
        :param num_decoder_layers: 解码器层数
        :param return_intermediate_dec: 是否返回解码器中间层结果
        """
        super().__init__()

        self.d_model = d_model
        self.n_head = n_heads
        # 将多个相同的encoder块组装为encoder
        encoder_layer = EncoderLayer(d_model, d_fc, dropout, activation, n_levels, n_heads, enc_n_points)
        self.encoder = _make_encoder(encoder_layer, num_encoder_layers)
        # 将多个相同的decoder块组装为encoder
        decoder_layer = DecoderLayer(d_model, d_fc, dropout, activation, n_levels, n_heads, dec_n_points)
        self.decoder = _make_decoder(decoder_layer, num_decoder_layers, return_intermediate_dec)

        # 创建特征图尺度的embedding, size:[lvl, d_model]
        self.level_embed = nn.Parameter(torch.Tensor(n_levels, d_model))
        # 通过一个线性投影层产生点P的两个偏移（x轴和y轴）delta P，生成P的参考点
        self.sample = nn.Linear(d_model, 2)
        # backbone的参数
        self._reset_parameters()

    def _reset_parameters(self):
        for p in self.parameters():
            # p为某一层的权重参数矩阵
            if p.dim() > 1:
                # nn.init.xavier_uniform_在每一层网络保证输入和输出的方差相同，包括前向传播和后向传播
                nn.init.xavier_uniform_(p)
        # 递归地在transformer每一个模块调用xavier_uniform
        for m in self.modules():
            if isinstance(m, MSDeformAttn):
                m._reset_parameters()
        # 归一化产生参考点的线性投影层的权重，使得输入和输出的方差相同
        xavier_uniform_(self.sample.weight.data, gain=1.0)
        # 固定产生参考点的线性投影层的bias为0，且不反向传播梯度
        constant_(self.sample.bias.data, 0.)
        # 用从正态分布中采样的参数来填充表示特征图尺度的embedding
        normal_(self.level_embed)

    # for two stage training
    def get_proposal_pos_embed(self, proposals):
        num_pos_feats = 128
        temperature = 10000
        scale = 2 * math.pi

        # dim_t[0,1,2,3,...,127]
        dim_t = torch.arange(num_pos_feats, dtype=torch.float32, device=proposals.device)
        # dim_t[10000**0,10000**0,10000**2,10000**2,...,10000**126,10000**126]
        dim_t = temperature ** (2 * (dim_t // 2) / num_pos_feats)
        # N, L, 4
        proposals = proposals.sigmoid() * scale
        # N, L, 4, 128
        pos = proposals[:, :, :, None] / dim_t
        # N, L, 4, 64, 2
        pos = torch.stack((pos[:, :, :, 0::2].sin(), pos[:, :, :, 1::2].cos()), dim=4).flatten(2)
        return pos
    # for two stage training
    def gen_encoder_output_proposals(self, memory, memory_padding_mask, spatial_shapes):
        N_, S_, C_ = memory.shape
        # memory_padding_mask.size(bs, h*W*lvl)
        base_scale = 4.0
        proposals = []
        _cur = 0

        for lvl, (H_, W_) in enumerate(spatial_shapes):
            # 取出这个level特征图的掩码矩阵，形式为（bs,h,w,1）
            mask_flatten_ = memory_padding_mask[:, _cur:(_cur + H_ * W_)].view(N_, H_, W_, 1)
            # 统计求和第一列不被遮掩的像素数，作为有效高度,size(bs)
            valid_H = torch.sum(~mask_flatten_[:, :, 0, 0], 1)
            # 统计求和第一行不被遮掩的像素数，作为有效高度,size(bs)
            valid_W = torch.sum(~mask_flatten_[:, 0, :, 0], 1)

            grid_y, grid_x = torch.meshgrid(torch.linspace(0, H_ - 1, H_, dtype=torch.float32, device=memory.device),
                                            torch.linspace(0, W_ - 1, W_, dtype=torch.float32, device=memory.device))
            # grid.size(h,w,2)
            grid = torch.cat([grid_x.unsqueeze(-1), grid_y.unsqueeze(-1)], -1)

            # scale.size(bs,1,1,2),最后一维是这个尺度特征图的有效高度和有效宽度
            scale = torch.cat([valid_W.unsqueeze(-1), valid_H.unsqueeze(-1)], 1).view(N_, 1, 1, 2)
            # 将grid中的坐标位移至该尺度特征图每个像素的中心点坐标，除以有效高度和宽度,归一化坐标，被mask掉的像素的中心点坐标可能大于1
            grid = (grid.unsqueeze(0).expand(N_, -1, -1, -1) + 0.5) / scale
            # 候选框的宽和高设置为0.05*2**lvl
            wh = torch.ones_like(grid) * 0.05 * (2.0 ** lvl)
            # 把候选框的中心点坐标和宽高放在一起，size(bs,W*H,4),最后一维为x,y,w,h
            proposal = torch.cat((grid, wh), -1).view(N_, -1, 4)
            proposals.append(proposal)
            _cur += (H_ * W_)
        # 各个尺度特征图的候选框参数,size(bs, w*h*lvl, 4)
        output_proposals = torch.cat(proposals, 1)
        # 将x,y,w,h限定在(0.01,0.99)间，不满足的候选框剔除掉
        output_proposals_valid = ((output_proposals > 0.01) & (output_proposals < 0.99)).all(-1, keepdim=True)
        output_proposals = torch.log(output_proposals / (1 - output_proposals))
        # 用掩码矩阵过滤掉中心点被mask的候选框
        output_proposals = output_proposals.masked_fill(memory_padding_mask.unsqueeze(-1), float('inf'))
        # 用output_proposals_valid过滤掉参数不在（0.01，0.99）之间的候选框
        output_proposals = output_proposals.masked_fill(~output_proposals_valid, float('inf'))

        output_memory = memory
        # 用掩码矩阵过滤掉被mask的像素的attention向量，被过滤像素的attention=0
        output_memory = output_memory.masked_fill(memory_padding_mask.unsqueeze(-1), float(0))

        # 用output_proposals_valid过滤掉参数不在（0.01，0.99）之间的候选框，被过滤的候选框参数为0
        output_proposals = output_proposals.masked_fill(~output_proposals_valid, float(0))

        # 用output_proposals_valid过滤掉像素坐标不在（0.01，0.99）之间的像素的attention向量，被过滤像素的attention=0
        output_memory = output_memory.masked_fill(~output_proposals_valid, float(0))

        # 筛选后的attention通过一个线性映射层和一个layernorm
        output_memory = self.enc_output_norm(self.enc_output(output_memory))
        return output_memory, output_proposals

    def get_valid_ratio(self, mask):
        # _:bs
        _, H, W = mask.shape
        # valid_H.size(bs,H)统计特征图第一列像素中不被mask的个数，即该特征图的有效高度
        # valid_w.size(bs,W)统计特征图第一行像素中不被mask的个数，即该特征图的有效宽度
        valid_H = torch.sum(~mask[:, :, 0], 1)
        valid_W = torch.sum(~mask[:, 0, :], 1)
        # 将有效高度和宽度归一化到[0,1]之间，称为有效比率
        valid_ratio_h = valid_H.float() / H
        valid_ratio_w = valid_W.float() / W
        valid_ratio = torch.stack([valid_ratio_w, valid_ratio_h], -1) \
            # valid_ratio.size(bs,2)最后一个维度存储每个特征图宽和高的有效比率，形如[0.3, 0.6]
        return valid_ratio

    def get_flatten_embed(self, srcs, masks, pos_embeds):
        """
        :param srcs:       backbone输出的多尺度特征图, of size[level, bs, c=d_model, h, w]
        :param pos_embeds: backbone输出的多尺度位置编码, of size[level, bs, c=d_model, h, w]
        :param masks:      backbone输出的多尺度掩码, of size[level, bs, h, w]
        :return src_flatten:          (b, h*w*lvl, d_model)
                lvl_pos_embed_flatten:(b, h*w*lvl, d_model)
                mask_flatten:         (b, h*w*lvl)
        """
        src_flatten = []
        mask_flatten = []
        lvl_pos_embed_flatten = []
        spatial_shapes = []

        # src: 特征图; (b, d_model, h, w)
        # mask:掩码矩阵; (b, h, w)
        # pos_embed: 位置embedding; size：(b, d_model, h, w)，和特征图大小一致
        # lvl: 特征图尺度 1/2/3/4
        for lvl, (src, mask, pos_embed) in enumerate(zip(srcs, masks, pos_embeds)):
            b, c, h, w = src.shape
            spatial_shape = (h, w)
            spatial_shapes.append(spatial_shape)

            # src:(b,h,w,c)->(b, hw, c)
            src = rearrange(src, "b c h w -> b hw c")
            src = rearrange(src, "b c h w -> b (h w) c")
            # mask:(b,h,w)->(b, hw)
            mask = rearrange(mask, "b h w -> b (h w)")
            # pos_embed:(b,h,w,c)->(b, hw, c)
            pos_embed = rearrange(pos_embed, "b h w c -> b (h w) c")
            # lvl_embed:(c)->(b, hw, c)
            lvl_embed = self.level_embed[lvl].view(1, 1, -1)
            # lvl_pos_embed:(b, hW, c)最后一维是位置编码和尺度编码的求和结果
            lvl_pos_embed = pos_embed + lvl_embed

            src_flatten.append(src)
            mask_flatten.append(mask)
            lvl_pos_embed_flatten.append(lvl_pos_embed)

        # 把多个尺度上的特征图输入、掩码矩阵、尺度位置编码拼接起来
        # src_flatten:          (b, lvl, h*w, d_model)
        # lvl_pos_embed_flatten:(b, lvl, h*w, d_model)
        # mask_flatten:         (b, lvl*h*w)
        src_flatten = torch.cat(src_flatten, 1)
        mask_flatten = torch.cat(mask_flatten, 1)
        lvl_pos_embed_flatten = torch.cat(lvl_pos_embed_flatten, 1)
        # 将spatial_shapes转化为tensor,最后一维度存储一对(h,w): (lvl,2)
        spatial_shapes = torch.as_tensor(spatial_shapes, dtype=torch.long, device=src_flatten.device)

        return spatial_shapes, src_flatten, mask_flatten, lvl_pos_embed_flatten

    def forward(self, srcs, masks, pos_embeds, query_embed=None):
        """
        (1)Encode each pixel in each level with embed[d_model] + pos_level_embed[d_model], and flatten [level,c,h,w]
           to [c,h*w*level].Prepare valid_ratios, level_start index for encoder.
        (2)Encoding
        (3)Split the query embed into tgt embed and pos embed, and generate initial reference points from pos embed.
        (4)Decoding

        :param srcs:        backbone输出的多尺度特征图, of size[level, bs, c=d_model, h, w]
        :param pos_embeds:  backbone输出的多尺度位置编码, of size[level, bs, c=d_model, h, w]
        :param masks:       backbone输出的多尺度掩码, of size[level, bs, h, w]
        :param query_embed: decoder的输入query, 包括内容编码(d_model)和位置编码(d_model), of size[n_queries, c=d_model*2]

        :return DEC_Hidden: queries经过decoder解码后的隐层表示, 用于预测box坐标和类别, of size[bs, n_queries, c=d_model]
                init_reference_out: 由queries的位置编码通过线性映射得到的初始参考点
                inter_references_out: 利用decoder中间层输出修正的参考点
        """
        # 确保query_embed不为空
        assert query_embed is not None
        """
        (1) Prepared Input
            Encode each pixel in each level with embed[d_model] + pos_level_embed[d_model], and flatten [level,c,h,w] 
            to [c,h*w*level].Prepare valid_ratios, level_start index for encoder. 
        """
        spatial_shapes, src_flatten, mask_flatten, lvl_pos_embed_flatten = self.get_flatten_embed(srcs, masks, pos_embeds)
        # spatial_shapes.new_zeros((1, )).size(1,1)形如([0])
        # spatial_shapes.prod(1).cumsum(0)[:-1].size(lvl-1),存储每个尺度像素数的累计求和结果，如果四个尺度的像素数为[4,8,16,32],返回[4,12,28]
        # level_start_index形如[0,4,12,28],投入到linear(4, d_model), 得到每个level的大小为d_model的尺度编码
        level_start_index = torch.cat((spatial_shapes.new_zeros((1,)), spatial_shapes.prod(1).cumsum(0)[:-1]))
        # 通过不同尺度的掩码矩阵得到每张图片在不同尺度下宽和高的有效比率(不被mask的像素所占的整个特征图的比例),of size[bs,lvl,2]
        valid_ratios = torch.stack([self.get_valid_ratio(m) for m in masks], 1)
        """
        (2) Encoding
        """
        ENC_Hidden = self.encoder(src_flatten, spatial_shapes, level_start_index,
                              valid_ratios, lvl_pos_embed_flatten, mask_flatten)
        """
        (3) Prepared Queries
            generate pos embed and tgt embed of queries
            generate reference points for queries using pos embed
        """
        # prepare input for decoder
        bs, _, c = ENC_Hidden.shape
        # 将query_embed切分成pos_embed和target
        pos_embed, tgt_embed = torch.split(query_embed, c, dim=1)
        pos_embed = pos_embed.unsqueeze(0).expand(bs, -1, -1) # of size[n_queries, c=d_model]
        tgt_embed = tgt_embed.unsqueeze(0).expand(bs, -1, -1) # of size[n_queries, c=d_model]
        # encoder阶段，query（其实是各个尺度特征图的所有像素）的reference_points是每个像素的中心点
        # decoder阶段，query的reference_points通过投影query的pos embedding生成两个坐标来产生
        reference_points = self.sample(pos_embed).sigmoid() # of size[bs, n_queries, 2]
        init_reference_out = reference_points
        """
        (4) Decoding
        """
        DEC_Hidden, inter_references = self.decoder(tgt_embed, reference_points, ENC_Hidden, spatial_shapes,
                                            level_start_index, valid_ratios, query_embed, mask_flatten)
        inter_references_out = inter_references

        return DEC_Hidden, init_reference_out, inter_references_out, None, None

def build_deforamble_transformer(args):
    return DeformableTransformer(
        d_model=args.hidden_dim,
        n_heads=args.nheads,
        num_encoder_layers=args.enc_layers,
        num_decoder_layers=args.dec_layers,
        d_fc=args.dim_feedforward,
        dropout=args.dropout,
        activation="relu",
        return_intermediate_dec=True,
        n_levels=args.num_feature_levels,
        dec_n_points=args.dec_n_points,
        enc_n_points=args.enc_n_points)