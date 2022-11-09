import torch
import torch.nn.functional as F
from einops import rearrange
from torch import nn
import torch.nn.init as init

def _weights_init(m):
    classname = m.__class__.__name__
    #print(classname)
    if isinstance(m, nn.Linear) or isinstance(m, nn.Conv2d):
        init.kaiming_normal_(m.weight)

# define a network prototype
class LambdaLayer(nn.Module):
    def __init__(self, lambd):
        super(LambdaLayer, self).__init__()
        self.lambd = lambd

    def forward(self, x):
        return self.lambd(x)

# define a residual block
class Residual(nn.Module):
    def __init__(self, net):
        super().__init__()
        self.fn = net

    def forward(self, x, **kwargs):
        return self.net(x, **kwargs) + x

# define a layernorm layer
class LayerNormalize(nn.Module):
    def __init__(self, dim, net):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.fn = net

    def forward(self, x, **kwargs):
        return self.net(self.norm(x), **kwargs)

# define a basic Residual Block: 2 conv
class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_channels, out_channels, stride=1, option='A'):
        super(BasicBlock, self).__init__()

        # conv1 reduce the scale, while deepen the net
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)

        # conv2 maintain the scale, while deepen the net
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)

        self.shortcut = nn.Sequential()
        # if the output.scale < input.scale/output.depth > input.depth, pad to make output.size = input.size
        if stride != 1 or in_channels != out_channels:
            if option == 'A':
                #x:(b,c,h,w)
                #reduce h,w to half, pad c by 1/2*
                #to make the x and the out of the residual being the same size
                self.shortcut = LambdaLayer(lambda x:
                                            F.pad(x[:, :, ::2, ::2], (0, 0, 0, 0, out_channels//4, out_channels//4), "constant", 0))
            elif option == 'B':
                self.shortcut = nn.Sequential(
                     nn.Conv2d(in_channels, self.expansion * out_channels, kernel_size=1, stride=stride, bias=False),
                     nn.BatchNorm2d(self.expansion * out_channels)
                )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        #print(out.size())
        return out


# define a MLP Block: 2 fc
class MLP_Block(nn.Module):
    def __init__(self, dim, hidden_dim, dropout=0.1):
        super().__init__()
        self.nn1 = nn.Linear(dim, hidden_dim)
        torch.nn.init.xavier_uniform_(self.nn1.weight)
        torch.nn.init.normal_(self.nn1.bias, std=1e-6)
        self.af1 = nn.GELU()
        self.do1 = nn.Dropout(dropout)
        self.nn2 = nn.Linear(hidden_dim, dim)
        torch.nn.init.xavier_uniform_(self.nn2.weight)
        torch.nn.init.normal_(self.nn2.bias, std=1e-6)
        self.do2 = nn.Dropout(dropout)

    def forward(self, x):
        x = self.nn1(x)
        x = self.af1(x)
        x = self.do1(x)
        x = self.nn2(x)
        x = self.do2(x)

        return x


# define a Backbone
class Backbone(nn.Module):
    def __init__(self, block, num_blocks, num_classes=10, dim=128, num_tokens=8, mlp_dim=256, heads=8, depth=6,
                 emb_dropout=0.1, dropout=0.1):
        super(Backbone, self).__init__()
        self.in_channels = 16
        self.L = num_tokens
        self.cT = dim

        # convert RGB channels into 16 channels, maintain the scale
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(16)

        # reduce the scale，deepen the channels
        self.layer1 = self._make_layer(block, 16, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 32, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 64, num_blocks[2], stride=2)  # 8x8 feature maps (64 in total)
        self.apply(_weights_init)

    def _make_layer(self, BasicBlock, out_channels, num_blocks, stride):
        # num_blocks=3
        # expansion = 1

        # strides = [stride, 1, 1]
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(BasicBlock(self.in_channels, out_channels, stride))
            self.in_channels = out_channels * BasicBlock.expansion

        return nn.Sequential(*layers)

    def forward(self, img, mask=None):
        x = F.relu(self.bn1(self.conv1(img)))
        x_16 = self.layer1(x)
        x_32 = self.layer2(x_16)
        x_64 = self.layer3(x_32)

        x_16 = rearrange(x_16, 'b c h w -> b (h w) c')  # vectors each with 16 points.
        x_32 = rearrange(x_32, 'b c h w -> b (h w) c')  # vectors each with 32 points.
        x_64 = rearrange(x_64, 'b c h w -> b (h w) c')  # vectors each with 64 points.

        return (x_16, x_32, x_64)