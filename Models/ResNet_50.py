import torch
import torch.nn.functional as F
import torchvision
from torch import nn
from torchvision.models._utils import IntermediateLayerGetter
from typing import Dict, List
from util.misc import NestedTensor, is_main_process
from .PositionEmebdding import build_position_encoding


class FrozenBatchNorm2d(torch.nn.Module):
    """
    BatchNorm2d where the batch statistics and the affine parameters are fixed.

    Copy-paste from torchvision.misc.ops with added eps before rqsrt,
    without which any other models than torchvision.models.resnet[18,34,50,101]
    produce nans.
    """

    def __init__(self, n, eps=1e-5):
        super(FrozenBatchNorm2d, self).__init__()
        self.register_buffer("weight", torch.ones(n))
        self.register_buffer("bias", torch.zeros(n))
        self.register_buffer("running_mean", torch.zeros(n))
        self.register_buffer("running_var", torch.ones(n))
        self.eps = eps

    def _load_from_state_dict(self, state_dict, prefix, local_metadata, strict,
                              missing_keys, unexpected_keys, error_msgs):
        num_batches_tracked_key = prefix + 'num_batches_tracked'
        if num_batches_tracked_key in state_dict:
            del state_dict[num_batches_tracked_key]

        super(FrozenBatchNorm2d, self)._load_from_state_dict(
            state_dict, prefix, local_metadata, strict,
            missing_keys, unexpected_keys, error_msgs)

    def forward(self, x):
        # move reshapes to the beginning
        # to make it fuser-friendly
        w = self.weight.reshape(1, -1, 1, 1)
        b = self.bias.reshape(1, -1, 1, 1)
        rv = self.running_var.reshape(1, -1, 1, 1)
        rm = self.running_mean.reshape(1, -1, 1, 1)
        eps = self.eps
        scale = w * (rv + eps).rsqrt()
        bias = b - rm * scale
        return x * scale + bias


class BackboneBase(nn.Module):

    def __init__(self, backbone: nn.Module, train_backbone: bool, return_interm_layers: bool):
        super().__init__()
        for name, parameter in backbone.named_parameters():
            if not train_backbone or 'layer2' not in name and 'layer3' not in name and 'layer4' not in name:
                # ???????????????????????????
                parameter.requires_grad_(False)

        # ???????????????2???3???4?????????
        if return_interm_layers:
            # return_layers = {"layer1": "0", "layer2": "1", "layer3": "2", "layer4": "3"}
            return_layers = {"layer2": "0", "layer3": "1", "layer4": "2"}
            self.strides = [8, 16, 32]
            self.n_channels = [512, 1024, 2048]
        # ??????????????????(???4???)?????????
        else:
            return_layers = {'layer4': "0"}
            self.strides = [32]
            self.n_channels = [2048]

        # ??????backbone??????????????????
        self.body = IntermediateLayerGetter(backbone, return_layers=return_layers)

    def forward(self, tensor_list: NestedTensor):
        """
        The input is a NestedTensor of img, where consists of the tensor and the mask. The mask is only for semantic segmentation.
        We put input into the intermediate layer of resnet50, and get the output tensor of each layer, and then combine them with their
        corresponding scaled mask.
        :param tensor_list:
        :return out of each layer in backbone, out:NestedTensor(tensor,mask)
        """
        # xs.items() : [('0',tensor(),'1',tensor(),...]
        xs = self.body(tensor_list.tensors)
        out: Dict[str, NestedTensor] = {}
        for name, tensor in xs.items():
            # x = tensor_list.tensor
            # m???img?????????mask
            m = tensor_list.mask
            assert m is not None
            # ???img?????????mask????????????,?????????out_tensor?????????????????????mask
            mask = F.interpolate(m[None].float(), size=tensor.shape[-2:]).to(torch.bool)[0]
            # tensor:(b,c,h,w),mask:(b,h,w)
            # out:['0':NestedTensor(tensor,mask),'1':NestedTensor(tensor,mask),...,]
            out[name] = NestedTensor(tensor, mask)
        return out


class Backbone(BackboneBase):
    """
    Here is a 3-hidden-layer backbone(resnet-50), the input is a NestedTensor of img, where consists of the tensor and
    the mask. The mask is only for semantic segmentation.We put input into the intermediate layer of resnet50, and get
    the output tensor of each layer, and then combine them with their corresponding scaled mask.

    The 3-hidden-layer network structure is as followed:
    self.strides:[8, 16, 32]
    self.n_channels:[512, 1024, 2048]
    self.body:{"layer2": "0", "layer3": "1", "layer4": "2"}

    The forward process is as followed:
    input: tensor_list:NestedTensor(tensor,mask)
    output:out of each hidden layer in backbone:NestedTensor(tensor,mask)
    """
    def __init__(self, name: str,
                 train_backbone: bool,
                 return_interm_layers: bool,
                 dilation: bool):
        norm_layer = FrozenBatchNorm2d
        # ??????default='resnet50'????????????????????????????????????name = 'resnet50'????????????init??????resnet50??????
        backbone = getattr(torchvision.models, name)(
            replace_stride_with_dilation=[False, False, dilation],
            pretrained=is_main_process(), norm_layer=norm_layer)
        # backbone??????resnet50????????????nn.Module
        assert name not in ('resnet18', 'resnet34'), "number of channels are hard coded"
        # ??????BackboneBase????????????????????????self.strides, self.n_channels, self.body
        super().__init__(backbone, train_backbone, return_interm_layers)
        # ?????????????????????1/2??????????????????size??????????????????
        if dilation:
            self.strides[-1] = self.strides[-1] // 2


class Joiner(nn.Sequential):
    """
    Here is a 2-block sequential, which consists of a backbone and a position encoder. The img as NestedTensor is put
    into the backbone to get the output of each layer as feature maps of each level, and then these feature maps are put
    into the position encoder to get the position embedding of each level.

    The 3-hidden-layer network structure is as followed:
        self.strides:[8, 16, 32]
        self.n_channels:[512, 1024, 2048]
        self.n_resolutions:3
        self[0]:resnet-50{"layer2": "0", "layer3": "1", "layer4": "2"}
        self[1]:position encoder

        The forward process is as followed:
        input: tensor_list:NestedTensor(tensor,mask)
        output:out of each hidden layer in backbone:NestedTensor(tensor,mask)
              :pos embedding of each hidden layer in backbone
    """
    def __init__(self, backbone, position_encoder):
        # ??????backbone???????????????, ??????self.strides, self.n_channels, self.body
        super().__init__(backbone, position_encoder)
        self.strides = backbone.strides
        self.n_channels = backbone.n_channels
        self.n_resolutions = 3

    def forward(self, tensor_list: NestedTensor):
        # resnet: ????????????tensor_list???tensor+mask????????????out:['0':(tensor,mask),'1':(tensor,mask),...,]
        # self[0]=backbone,self[1]=pos
        xs = self[0](tensor_list)
        out = []
        pos = []
        # x???resnet50?????????????????????name????????????:"layer1": "0", "layer2": "1", "layer3": "2", "layer4": "3"
        for name, nestedtensor in sorted(xs.items()):
            out.append(nestedtensor)
        # ??????????????????tensor??????position_encoder,??????????????????position embed
        for nestedtensor in out:
            pos_embed = self[1](nestedtensor).to(nestedtensor.tensors.dtype)
            pos.append(pos_embed)

        return out, pos


def build_backbone(args):
    position_encoder = build_position_encoding(args)
    if_train = args.lr_backbone > 0
    if_return_interm_layers = (args.num_feature_levels > 1)
    backbone = Backbone(args.backbone, if_train, if_return_interm_layers, args.dilation)
    model = Joiner(backbone, position_encoder)
    return model
