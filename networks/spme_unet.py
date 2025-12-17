
import torch
import torch.nn as nn
import torchvision
import torch.nn.functional as F
from functools import partial
from timm.layers import DropPath, to_2tuple, trunc_normal_
import math
from mmseg.registry import MODELS
from mmengine.logging import MMLogger
from mmengine.runner.checkpoint import load_checkpoint
import swattention
from einops import rearrange
from timm.layers import trunc_normal_tf_
from timm.models import named_apply
from einops.layers.torch import Rearrange, Reduce
from torchvision.ops import DeformConv2d
CUDA_NUM_THREADS = 128


class DynamicKernel(nn.Module):
    """Dynamic Kernel branch """

    def __init__(self, dim, kernel_sizes=[3, 5], groups=4):
        super().__init__()
        self.kernel_sizes = kernel_sizes
        self.groups = groups
        self.bn_g = nn.BatchNorm2d(dim)
        self.bn_x = nn.BatchNorm2d(dim)
        self.GC_x = nn.Conv2d(dim, dim, kernel_size=3, padding=1, groups=groups)

        self.conv_attn = nn.Sequential(
            nn.Conv2d(dim, 1, kernel_size=1),
            nn.BatchNorm2d(1),
            nn.Sigmoid()
        )
        self.conv_branches = nn.ModuleList([
            nn.Conv2d(
                dim, dim,
                kernel_size=k,
                padding=k // 2,
                groups=groups
            ) for k in kernel_sizes
        ])


        self.weight_net = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(dim, len(kernel_sizes), kernel_size=1),
            nn.Softmax(dim=1)
        )

    def forward(self, g, x):

        branch_outputs = [conv(g) for conv in self.conv_branches]

        weights = self.weight_net(g)


        fused_g = torch.stack(branch_outputs, dim=1)
        weighted_g = (fused_g * weights.unsqueeze(2)).sum(dim=1)


        x_conv = self.GC_x(x)
        fused = torch.relu(self.bn_g(weighted_g) + self.bn_x(x_conv))
        attn = self.conv_attn(fused)
        return x * attn
class DeformableGroupConv(nn.Module):
    """DeformableGroupConv branch"""
    def __init__(self, in_channels, groups, kernel_size=3, padding=1):
        super().__init__()
        self.offset_conv = nn.Conv2d(
            in_channels,
            2 * kernel_size * kernel_size * groups,
            kernel_size=3,
            padding=1
        )
        self.deform_conv = DeformConv2d(
            in_channels,
            in_channels,
            kernel_size=kernel_size,
            padding=padding,
            groups=groups
        )

    def forward(self, x):
        offset = self.offset_conv(x)
        return self.deform_conv(x, offset)
class DeformConv(nn.Module):

    def __init__(self, in_channels, groups, kernel_size=(3, 3), padding=1, stride=1, dilation=1, bias=True):
        super(DeformConv, self).__init__()

        self.offset_net = nn.Conv2d(in_channels=in_channels,
                                    out_channels=2 * kernel_size[0] * kernel_size[1],
                                    kernel_size=kernel_size,
                                    padding=padding,
                                    stride=stride,
                                    dilation=dilation,
                                    bias=True)

        self.deform_conv = torchvision.ops.DeformConv2d(in_channels=in_channels,
                                                        out_channels=in_channels,
                                                        kernel_size=kernel_size,
                                                        padding=padding,
                                                        groups=groups,
                                                        stride=stride,
                                                        dilation=dilation,
                                                        bias=False)

    def forward(self, x):
        offsets = self.offset_net(x)
        out = self.deform_conv(x, offsets)
        return out


class DeformConv_3x3(nn.Module):

    def __init__(self, in_channels, groups, kernel_size=(3, 3), padding=1, stride=1, dilation=1, bias=True):
        super(DeformConv_3x3, self).__init__()

        self.offset_net = nn.Conv2d(in_channels=in_channels,
                                    out_channels=2 * kernel_size[0] * kernel_size[1],
                                    kernel_size=3,
                                    padding=1,
                                    stride=1,
                                    bias=True)

        self.deform_conv = torchvision.ops.DeformConv2d(in_channels=in_channels,
                                                        out_channels=in_channels,
                                                        kernel_size=kernel_size,
                                                        padding=padding,
                                                        groups=groups,
                                                        stride=stride,
                                                        dilation=dilation,
                                                        bias=False)

    def forward(self, x):
        offsets = self.offset_net(x)
        out = self.deform_conv(x, offsets)
        return out





class deformable_LKA(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.conv0 = DeformConv(dim, kernel_size=(5, 5), padding=2, groups=dim)
        self.conv_spatial = DeformConv(dim, kernel_size=(7, 7), stride=1, padding=9, groups=dim, dilation=3)
        self.conv1 = nn.Conv2d(dim, dim, 1)

    def forward(self, x):
        u = x.clone()
        attn = self.conv0(x)
        attn = self.conv_spatial(attn)
        attn = self.conv1(attn)

        return u * attn





class deformable_LKA_Attention(nn.Module):
    def __init__(self, d_model):
        super().__init__()

        self.proj_1 = nn.Conv2d(d_model, d_model, 1)
        self.activation = nn.GELU()
        self.spatial_gating_unit = deformable_LKA(d_model)
        self.proj_2 = nn.Conv2d(d_model, d_model, 1)

    def forward(self, x):
        shorcut = x.clone()
        x = self.proj_1(x)
        x = self.activation(x)
        x = self.spatial_gating_unit(x)
        x = self.proj_2(x)
        x = x + shorcut
        return x





class sw_qkrpb_cuda(torch.autograd.Function):
    @staticmethod
    def forward(ctx, query, key, rpb, height, width, kernel_size):
        attn_weight = swattention.qk_rpb_forward(query, key, rpb, height, width, kernel_size, CUDA_NUM_THREADS)

        ctx.save_for_backward(query, key)
        ctx.height, ctx.width, ctx.kernel_size = height, width, kernel_size

        return attn_weight

    @staticmethod
    def backward(ctx, d_attn_weight):
        query, key = ctx.saved_tensors
        height, width, kernel_size = ctx.height, ctx.width, ctx.kernel_size

        d_query, d_key, d_rpb = swattention.qk_rpb_backward(d_attn_weight.contiguous(), query, key, height, width,
                                                            kernel_size, CUDA_NUM_THREADS)

        return d_query, d_key, d_rpb, None, None, None


class sw_av_cuda(torch.autograd.Function):
    @staticmethod
    def forward(ctx, attn_weight, value, height, width, kernel_size):
        output = swattention.av_forward(attn_weight, value, height, width, kernel_size, CUDA_NUM_THREADS)

        ctx.save_for_backward(attn_weight, value)
        ctx.height, ctx.width, ctx.kernel_size = height, width, kernel_size

        return output

    @staticmethod
    def backward(ctx, d_output):
        attn_weight, value = ctx.saved_tensors
        height, width, kernel_size = ctx.height, ctx.width, ctx.kernel_size

        d_attn_weight, d_value = swattention.av_backward(d_output.contiguous(), attn_weight, value, height, width,
                                                         kernel_size, CUDA_NUM_THREADS)

        return d_attn_weight, d_value, None, None, None





class PatchExpand(nn.Module):
    def __init__(self, input_resolution, dim, dim_scale=2, norm_layer=nn.LayerNorm):
        super().__init__()
        self.input_resolution = input_resolution
        self.dim = dim
        self.dim_scale = dim_scale
        self.expand = nn.Linear(dim, dim_scale * dim, bias=False)
        self.norm = norm_layer(dim // dim_scale)

    def forward(self, x):
        B, C, H, W = x.shape
        x = x.permute(0, 2, 3, 1).reshape(B, H * W, C)
        x = self.expand(x)
        x = x.reshape(B, H, W, self.dim_scale * C)


        p = self.dim_scale
        c = (self.dim_scale * C) // (p ** 2)
        x = rearrange(x, 'b h w (p1 p2 c)-> b (h p1) (w p2) c', p1=p, p2=p, c=c)
        x = self.norm(x)
        x = x.permute(0, 3, 1, 2)
        return x


class FinalPatchExpand_X4(nn.Module):
    def __init__(self, input_resolution, dim, norm_layer=nn.LayerNorm):
        super().__init__()
        self.input_resolution = input_resolution
        self.dim = dim
        self.expand = nn.Linear(dim, 16 * dim, bias=False)
        self.norm = norm_layer(dim)

    def forward(self, x):
        H, W = self.input_resolution
        x = x.permute(0, 2, 3, 1).reshape(x.size(0), H * W, -1)
        x = self.expand(x)
        x = x.view(x.size(0), H, W, 16 * self.dim)


        x = rearrange(x, 'b h w (p1 p2 c)-> b (h p1) (w p2) c', p1=4, p2=4, c=self.dim)
        x = self.norm(x)
        x = x.permute(0, 3, 1, 2)
        return x




class Dropsample(nn.Module):
    def __init__(self, prob=0):
        super().__init__()
        self.prob = prob

    def forward(self, x):
        device = x.device

        if self.prob == 0. or (not self.training):
            return x

        keep_mask = torch.FloatTensor((x.shape[0], 1, 1, 1), device=device).uniform_() > self.prob
        return x * keep_mask / (1 - self.prob)

class FeedForward(nn.Module):
    def __init__(self, dim, mult = 4, dropout = 0.):
        super().__init__()
        inner_dim = int(dim * mult)
        self.net = nn.Sequential(
            nn.Linear(dim, inner_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(inner_dim, dim),
            nn.Dropout(dropout)
        )
    def forward(self, x):
        return self.net(x)

class SqueezeExcitation(nn.Module):
    """SqueezeExcitation plus"""
    def __init__(self, dim, shrinkage_rate = 0.25):
        super().__init__()
        hidden_dim = int(dim * shrinkage_rate)

        self.gap_pool = nn.AdaptiveAvgPool2d(1)
        self.gmp_pool = nn.AdaptiveMaxPool2d(1)



        self.fc = nn.Sequential(
            nn.Conv2d(dim * 2, hidden_dim, 1),
            nn.GELU(),
            nn.Conv2d(hidden_dim, dim, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()

        gap = self.gap_pool(x)
        gmp = self.gmp_pool(x)

        combined = torch.cat([gap, gmp], dim=1)
        return x * self.fc(combined)

class MBConv(nn.Module):
    def __init__(
            self,
            dim_in,
            dim_out,
            *,
            downsample,
            expansion_rate=4,
            shrinkage_rate=0.25,
            dropout=0. ,kernel_size):
        super().__init__()
        self.dim_in = dim_in
        self.dim_out = dim_out
        self.downsample = downsample
        self.dropout = dropout
        self.kernel_size = kernel_size

        hidden_dim = int(expansion_rate * dim_out)
        stride = 2 if downsample else 1

        self.net = nn.Sequential(
            nn.Conv2d(dim_in, dim_in, kernel_size, padding=kernel_size // 2, stride=stride, groups=dim_in),
            nn.Conv2d(dim_in, hidden_dim, 1),
            nn.BatchNorm2d(hidden_dim),
            nn.GELU(),
            SqueezeExcitation(hidden_dim, shrinkage_rate=shrinkage_rate),
            nn.Conv2d(hidden_dim, dim_out, 1),
            nn.BatchNorm2d(dim_out),
        )
        if dim_in == dim_out and not downsample:
            self.dropsample = Dropsample(self.dropout)

    def forward(self, x):
        out = self.net(x)
        if self.dropsample is not None:
            out = self.dropsample(out)
        return out +x
def gcd(a, b):
    while b:
        a, b = b, a % b
    return a

def _init_weights(module, name, scheme=''):
    if isinstance(module, nn.Conv2d) or isinstance(module, nn.Conv3d):
        if scheme == 'normal':
            nn.init.normal_(module.weight, std=.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif scheme == 'trunc_normal':
            trunc_normal_tf_(module.weight, std=.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif scheme == 'xavier_normal':
            nn.init.xavier_normal_(module.weight)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif scheme == 'kaiming_normal':
            nn.init.kaiming_normal_(module.weight, mode='fan_out', nonlinearity='relu')
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        else:

            fan_out = module.kernel_size[0] * module.kernel_size[1] * module.out_channels
            fan_out //= module.groups
            nn.init.normal_(module.weight, 0, math.sqrt(2.0 / fan_out))
            if module.bias is not None:
                nn.init.zeros_(module.bias)
    elif isinstance(module, nn.BatchNorm2d) or isinstance(module, nn.BatchNorm3d):
        nn.init.constant_(module.weight, 1)
        nn.init.constant_(module.bias, 0)
    elif isinstance(module, nn.LayerNorm):
        nn.init.constant_(module.weight, 1)
        nn.init.constant_(module.bias, 0)
def act_layer(act, inplace=False, neg_slope=0.2, n_prelu=1):
    # activation layer
    act = act.lower()
    if act == 'relu':
        layer = nn.ReLU(inplace)
    elif act == 'relu6':
        layer = nn.ReLU6(inplace)
    elif act == 'leakyrelu':
        layer = nn.LeakyReLU(neg_slope, inplace)
    elif act == 'prelu':
        layer = nn.PReLU(num_parameters=n_prelu, init=neg_slope)
    elif act == 'gelu':
        layer = nn.GELU()
    elif act == 'hswish':
        layer = nn.Hardswish(inplace)
    else:
        raise NotImplementedError('activation layer [%s] is not found' % act)
    return layer

def channel_shuffle(x, groups):
    batchsize, num_channels, height, width = x.data.size()
    channels_per_group = num_channels // groups
    # reshape
    x = x.view(batchsize, groups,
               channels_per_group, height, width)
    x = torch.transpose(x, 1, 2).contiguous()
    # flatten
    x = x.view(batchsize, -1, height, width)
    return x
class MSDC(nn.Module):
    def __init__(self, in_channels, kernel_sizes, stride, activation='relu6', dw_parallel=True
                 ,mbconv_shrinkage_rate = 0.25, mbconv_expansion_rate = 4):
        super(MSDC, self).__init__()

        self.in_channels = in_channels
        self.kernel_sizes = kernel_sizes
        self.activation = activation
        self.dw_parallel = dw_parallel
        self.conv1 = MBConv(in_channels, in_channels, kernel_size=kernel_sizes[0],  downsample=False, expansion_rate=mbconv_expansion_rate,
                            shrinkage_rate=mbconv_shrinkage_rate)
        self.conv2 = MBConv(in_channels, in_channels, kernel_size=kernel_sizes[1], downsample=False, expansion_rate=mbconv_expansion_rate,
                            shrinkage_rate=mbconv_shrinkage_rate)
        self.conv3 = MBConv(in_channels, in_channels, kernel_size=kernel_sizes[2], downsample=False, expansion_rate=mbconv_expansion_rate,
                            shrinkage_rate=mbconv_shrinkage_rate)

        self.init_weights('normal')

    def init_weights(self, scheme=''):
        named_apply(partial(_init_weights, scheme=scheme), self)

    def forward(self, x):
        x1 = self.conv1(x)
        x2 = self.conv2(x)
        x3 = self.conv3(x)
        outputs = [x1, x2, x3]

        return outputs

class ChannelAttention(nn.Module):
    """
    Input -> x: [B, N, C]
    Output -> [B, N, C]
    """
    def __init__(self, dim, num_heads=8, qkv_bias=False, attn_drop=0, proj_drop=0):
        super().__init__()
        self.num_heads = num_heads
        self.temperature = nn.Parameter(torch.ones(num_heads, 1, 1))

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x):
        """x: [B, N, C]"""
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads)
        qkv = qkv.permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]

        q = q.transpose(-2, -1)
        k = k.transpose(-2, -1)
        v = v.transpose(-2, -1)

        q = F.normalize(q, dim=-1)
        k = F.normalize(k, dim=-1)

        attn = (q @ k.transpose(-2, -1)) * self.temperature
        # -------------------
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).permute(0, 3, 1, 2).reshape(B, N, C)
        # ------------------
        x = self.proj(x)
        x = self.proj_drop(x)

        return x
class MSCB(nn.Module):
    """
    Multi-scale convolution block (MSCB)
    """

    def __init__(self, in_channels, out_channels, stride, kernel_sizes=[1, 3, 5], expansion_factor=1, dw_parallel=True,
                 add=True, activation='relu6'):
        super(MSCB, self).__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.stride = stride
        self.kernel_sizes = kernel_sizes
        self.expansion_factor = expansion_factor
        self.dw_parallel = dw_parallel
        self.add = add
        self.activation = activation
        self.n_scales = len(self.kernel_sizes)
        # check stride value
        assert self.stride in [1, 2]
        # Skip connection if stride is 1
        self.use_skip_connection = True if self.stride == 1 else False

        # expansion factor
        self.ex_channels = int(self.in_channels * self.expansion_factor)
        self.pconv1 = nn.Sequential(
            # pointwise convolution
            nn.Conv2d(self.in_channels, self.ex_channels, 1, 1, 0, bias=False),
            nn.BatchNorm2d(self.ex_channels),
            act_layer(self.activation, inplace=True)
        )
        self.msdc = MSDC(self.ex_channels, self.kernel_sizes, self.stride, self.activation,
                         dw_parallel=self.dw_parallel)
        if self.add == True:
            self.combined_channels = self.ex_channels * 1
        else:
            self.combined_channels = self.ex_channels * self.n_scales
        self.pconv2 = nn.Sequential(
            # pointwise convolution
            nn.Conv2d(self.combined_channels, self.out_channels, 1, 1, 0, bias=False),
            nn.BatchNorm2d(self.out_channels),
        )
        if self.use_skip_connection and (self.in_channels != self.out_channels):
            self.conv1x1 = nn.Conv2d(self.in_channels, self.out_channels, 1, 1, 0, bias=False)
        self.init_weights('normal')
        self.channel_attn = ChannelAttention(3 * self.in_channels)
        self.norm = nn.LayerNorm(3 * self.in_channels)
        self.linear_downsample = nn.Linear(3 * self.in_channels, self.in_channels, bias=False)
    def init_weights(self, scheme=''):
        named_apply(partial(_init_weights, scheme=scheme), self)

    def forward(self, x):
        pout1 = self.pconv1(x)

        msdc_outs = self.msdc(pout1)
        embed = msdc_outs[0 ]+ msdc_outs[1 ] +msdc_outs[2]
        dout = channel_shuffle(embed, gcd(self.combined_channels, self.out_channels))
        out = self.pconv2(dout)
        if self.use_skip_connection:
            if self.in_channels != self.out_channels:
                x = self.conv1x1(x)
            return x + out
        else:
            return out


def MSCBLayer(in_channels, out_channels, n=1, stride=1, kernel_sizes=[1, 3, 5], expansion_factor=1, dw_parallel=True,
              add=True, activation='relu6'):
    """
    create a series of multi-scale convolution blocks.
    """
    convs = []
    mscb = MSCB(in_channels, out_channels, stride, kernel_sizes=kernel_sizes, expansion_factor=expansion_factor,
                dw_parallel=dw_parallel, add=add, activation=activation)
    convs.append(mscb)
    if n > 1:
        for i in range(1, n):
            mscb = MSCB(out_channels, out_channels, 1, kernel_sizes=kernel_sizes, expansion_factor=expansion_factor,
                        dw_parallel=dw_parallel, add=add, activation=activation)
            convs.append(mscb)
    conv = nn.Sequential(*convs)
    return conv

class CAB(nn.Module):
    def __init__(self, in_channels, out_channels=None, ratio=16, activation='relu'):
        super(CAB, self).__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        if self.in_channels < ratio:
            ratio = self.in_channels
        self.reduced_channels = self.in_channels // ratio
        if self.out_channels == None:
            self.out_channels = in_channels

        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.activation = act_layer(activation, inplace=True)
        self.fc1 = nn.Conv2d(self.in_channels, self.reduced_channels, 1, bias=False)
        self.fc2 = nn.Conv2d(self.reduced_channels, self.out_channels, 1, bias=False)

        self.sigmoid = nn.Sigmoid()

        self.init_weights('normal')

    def init_weights(self, scheme=''):
        named_apply(partial(_init_weights, scheme=scheme), self)

    def forward(self, x):
        avg_pool_out = self.avg_pool(x)
        avg_out = self.fc2(self.activation(self.fc1(avg_pool_out)))

        max_pool_out = self.max_pool(x)
        max_out = self.fc2(self.activation(self.fc1(max_pool_out)))

        out = avg_out + max_out
        return self.sigmoid(out)



"""Spatial attention block (SAB)"""
class SAB(nn.Module):
    def __init__(self, kernel_size=7):
        super(SAB, self).__init__()

        assert kernel_size in (3, 7, 11), 'kernel must be 3 or 7 or 11'
        padding = kernel_size // 2

        self.conv = nn.Conv2d(2, 1, kernel_size, padding=padding, bias=False)

        self.sigmoid = nn.Sigmoid()

        self.init_weights('normal')

    def init_weights(self, scheme=''):
        named_apply(partial(_init_weights, scheme=scheme), self)

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x = torch.cat([avg_out, max_out], dim=1)
        x = self.conv(x)
        return self.sigmoid(x)
class MFEM(nn.Module):
    def __init__(self,  channels=512, kernel_sizes=[1 ,3 ,5], expansion_factor=1, dw_parallel=True, add=True,  activation='relu6'):
        super(MFEM, self).__init__()
        self.mscb = MSCBLayer(channels, channels ,n=1, stride=1, kernel_sizes=kernel_sizes, expansion_factor=expansion_factor, dw_parallel=dw_parallel, add=add, activation=activation)
        self.cab = CAB(channels)
        self.sab = SAB()
    def forward(self, x):

        d = self.cab(x ) *x
        d = self.sab(d ) *d
        d = self.mscb(d)
        return d


class SPM(nn.Module):
    def __init__(self, F_g, F_l, F_int, kernel_size=3, groups=1, activation='relu'):
        super(SPM, self).__init__()
        self.deform_conv = DeformableGroupConv(F_l, groups=F_l)
        self.dynamic_kernel = DynamicKernel(F_l, kernel_sizes=[1 ,3, 5])


        self.GC_x = nn.Conv2d(F_l, F_l, kernel_size=3, padding=1, groups=F_l)


        self.conv_attn = nn.Sequential(
            nn.Conv2d(F_l, F_l//2, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(F_l//2, 1, kernel_size=1),
            nn.Sigmoid()
        )
        if kernel_size == 1:
            groups = 1
        self.F_g = F_g
        self.W_g = nn.Sequential(
            DeformConv_3x3(in_channels=F_l ,groups=F_l),
            nn.BatchNorm2d(F_g)
        )
        self.W_x = nn.Sequential(
            DeformConv_3x3(in_channels=F_l ,groups=F_l),
            nn.BatchNorm2d(F_l)
        )
        self.psi = nn.Sequential(
            nn.Conv2d(F_l, 1, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(1),
            nn.Sigmoid()
        )
        self.activation = act_layer(activation, inplace=True)

        self.init_weights('normal')

    def init_weights(self, scheme=''):
        named_apply(partial(_init_weights, scheme=scheme), self)

    def forward(self, g, x):
        g_deform = self.deform_conv(g)
        g_dynamic = self.dynamic_kernel(g_deform, x)

        return g_dynamic




from monai.networks.blocks.unetr_block import UnetrBasicBlock, UnetrUpBlock, UnetrPrUpBlock



class TransNeXtDecoder(nn.Module):
    def __init__(self ,img_size=224, patch_size=4,
                 norm_name: str = "instance",
                 res_block: bool = True,
                 spatial_dims: int = 2,
                 encoder_dims: list = [96, 192, 384, 768],
                 decoder_dims: list = [ 768 ,384 ,192 ,96]
                 ):

        super().__init__()
        self.mfem4 = MFEM(channels=encoder_dims[3])
        self.decoder4 = UnetrPrUpBlock(
            spatial_dims=spatial_dims,
            in_channels=encoder_dims[3],
            out_channels=decoder_dims[1],
            num_layer=0,
            stride=1,
            kernel_size=3,
            upsample_kernel_size=2,
            norm_name=norm_name,
            res_block=res_block,
        )
        self.spm3 = SPM(F_g=decoder_dims[1], F_l=decoder_dims[1], F_int=decoder_dims[1 ]//2, kernel_size=3, groups=decoder_dims[1 ]//2)
        self.mfem3 = MFEM(channels=encoder_dims[2])
        self.decoder3 = UnetrPrUpBlock(
            spatial_dims=spatial_dims,
            in_channels=decoder_dims[1],
            out_channels=decoder_dims[2],
            num_layer=0,
            stride=1,
            kernel_size=3,
            upsample_kernel_size=2,
            norm_name=norm_name,
            res_block=res_block,
        )
        self.spm2 = SPM(F_g=decoder_dims[2], F_l=decoder_dims[2], F_int=decoder_dims[2] // 2, kernel_size=3,
                          groups=decoder_dims[2] // 2)

        self.mfem2 = MFEM(channels=encoder_dims[1])
        self.decoder2 = UnetrPrUpBlock(
            spatial_dims=spatial_dims,
            in_channels=decoder_dims[2],
            out_channels=decoder_dims[3],
            num_layer=0,
            stride=1,
            kernel_size=3,
            upsample_kernel_size=2,
            norm_name=norm_name,
            res_block=res_block,
        )
        self.spm1 = SPM(F_g=decoder_dims[3], F_l=decoder_dims[3], F_int=decoder_dims[3] // 2, kernel_size=3,
                          groups=decoder_dims[3] // 2)
        self.mfem1 = MFEM(channels=encoder_dims[0])

        self.final_upsample = FinalPatchExpand_X4(
            input_resolution=(img_size // patch_size, img_size // patch_size),
            dim=decoder_dims[3],
            norm_layer=nn.LayerNorm
        )

        self.seg_head = nn.Conv2d(decoder_dims[3], 9, kernel_size=1)

    def forward(self, encoder_features):

        x = encoder_features[3]
        x = self.mfem4(x)
        x2 = self.decoder4(x)
        x3 = self.spm3(x2 ,encoder_features[2])
        x = x3 + x2
        x = self.mfem3(x)
        x2 = self.decoder3(x)
        x3 = self.spm2(x2, encoder_features[1])
        x = x2 + x3
        x = self.mfem2(x)
        x2 = self.decoder2(x)
        x3 = self.spm1(x2, encoder_features[0])
        x = x3 + x2
        x = self.mfem1(x)
        x = self.final_upsample(x)

        x = self.seg_head(x)

        return x




class DWConv(nn.Module):
    def __init__(self, dim=768):
        super(DWConv, self).__init__()
        self.dwconv = nn.Conv2d(dim, dim, kernel_size=3, stride=1, padding=1, bias=True, groups=dim)

    def forward(self, x, H, W):
        B, N, C = x.shape
        x = x.transpose(1, 2).view(B, C, H, W).contiguous()
        x = self.dwconv(x)
        x = x.flatten(2).transpose(1, 2)

        return x


class ConvolutionalGLU(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        hidden_features = int(2 * hidden_features / 3)
        self.fc1 = nn.Linear(in_features, hidden_features * 2)
        self.dwconv = DWConv(hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x, H, W):
        x, v = self.fc1(x).chunk(2, dim=-1)
        x = self.act(self.dwconv(x, H, W)) * v
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


@torch.no_grad()
def get_relative_position_cpb(query_size, key_size, pretrain_size=None,
                              device=torch.device('cuda')):
    pretrain_size = pretrain_size or query_size
    axis_qh = torch.arange(query_size[0], dtype=torch.float32, device=device)
    axis_kh = F.adaptive_avg_pool1d(axis_qh.unsqueeze(0), key_size[0]).squeeze(0)
    axis_qw = torch.arange(query_size[1], dtype=torch.float32, device=device)
    axis_kw = F.adaptive_avg_pool1d(axis_qw.unsqueeze(0), key_size[1]).squeeze(0)
    axis_kh, axis_kw = torch.meshgrid(axis_kh, axis_kw)
    axis_qh, axis_qw = torch.meshgrid(axis_qh, axis_qw)

    axis_kh = torch.reshape(axis_kh, [-1])
    axis_kw = torch.reshape(axis_kw, [-1])
    axis_qh = torch.reshape(axis_qh, [-1])
    axis_qw = torch.reshape(axis_qw, [-1])

    relative_h = (axis_qh[:, None] - axis_kh[None, :]) / (pretrain_size[0] - 1) * 8
    relative_w = (axis_qw[:, None] - axis_kw[None, :]) / (pretrain_size[1] - 1) * 8
    relative_hw = torch.stack([relative_h, relative_w], dim=-1).view(-1, 2)

    relative_coords_table, idx_map = torch.unique(relative_hw, return_inverse=True, dim=0)

    relative_coords_table = torch.sign(relative_coords_table) * torch.log2(
        torch.abs(relative_coords_table) + 1.0) / torch.log2(torch.tensor(8, dtype=torch.float32))

    return idx_map, relative_coords_table


@torch.no_grad()
def get_seqlen_scale(input_resolution, window_size, device):
    return torch.nn.functional.avg_pool2d(
        torch.ones(1, input_resolution[0], input_resolution[1], device=device) * (window_size ** 2), window_size,
        stride=1, padding=window_size // 2, ).reshape(-1, 1)





class AggregatedAttention(nn.Module):
    def __init__(self, dim, input_resolution, num_heads=8, window_size=3, qkv_bias=True,
                 attn_drop=0., proj_drop=0., sr_ratio=1, is_extrapolation=False):
        super().__init__()
        assert dim % num_heads == 0, f"dim {dim} should be divided by num_heads {num_heads}."

        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads

        self.sr_ratio = sr_ratio

        self.is_extrapolation = is_extrapolation

        if not is_extrapolation:
            # The estimated training resolution is used for bilinear interpolation of the generated relative position bias.
            self.trained_H, self.trained_W = input_resolution
            self.trained_len = self.trained_H * self.trained_W
            self.trained_pool_H, self.trained_pool_W = input_resolution[0] // self.sr_ratio, input_resolution[
                1] // self.sr_ratio
            self.trained_pool_len = self.trained_pool_H * self.trained_pool_W

        assert window_size % 2 == 1, "window size must be odd"
        self.window_size = window_size
        self.local_len = window_size ** 2

        self.temperature = nn.Parameter(
            torch.log((torch.ones(num_heads, 1, 1) / 0.24).exp() - 1))  # Initialize softplus(temperature) to 1/0.24.

        self.q = nn.Linear(dim, dim, bias=qkv_bias)
        self.query_embedding = nn.Parameter(
            nn.init.trunc_normal_(torch.empty(self.num_heads, 1, self.head_dim), mean=0, std=0.02))
        self.kv = nn.Linear(dim, dim * 2, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

        self.sr = nn.Conv2d(dim, dim, kernel_size=1, stride=1, padding=0)
        self.norm = nn.LayerNorm(dim)
        self.act = nn.GELU()

        # mlp to generate continuous relative position bias
        self.cpb_fc1 = nn.Linear(2, 512, bias=True)
        self.cpb_act = nn.ReLU(inplace=True)
        self.cpb_fc2 = nn.Linear(512, num_heads, bias=True)

        # relative_bias_local:
        self.relative_pos_bias_local = nn.Parameter(
            nn.init.trunc_normal_(torch.empty(num_heads, self.local_len), mean=0,
                                  std=0.0004))

        # dynamic_local_bias:
        self.learnable_tokens = nn.Parameter(
            nn.init.trunc_normal_(torch.empty(num_heads, self.head_dim, self.local_len), mean=0, std=0.02))
        self.learnable_bias = nn.Parameter(torch.zeros(num_heads, 1, self.local_len))

    def forward(self, x, H, W, relative_pos_index, relative_coords_table, seq_length_scale):
        B, N, C = x.shape
        pool_H, pool_W = H // self.sr_ratio, W // self.sr_ratio
        pool_len = pool_H * pool_W

        # Generate queries, normalize them with L2, add query embedding, and then magnify with sequence length scale and temperature.
        # Use softplus function ensuring that the temperature is not lower than 0.
        q_norm = F.normalize(self.q(x).reshape(B, N, self.num_heads, self.head_dim).permute(0, 2, 1, 3), dim=-1)
        q_norm_scaled = (q_norm + self.query_embedding) * F.softplus(self.temperature) * seq_length_scale

        # Generate unfolded keys and values and l2-normalize them
        k_local, v_local = self.kv(x).reshape(B, N, 2 * self.num_heads, self.head_dim).permute(0, 2, 1, 3).chunk(2,
                                                                                                                 dim=1)

        # Compute local similarity
        attn_local = sw_qkrpb_cuda.apply(q_norm_scaled.contiguous(), F.normalize(k_local, dim=-1).contiguous(),
                                         self.relative_pos_bias_local,
                                         H, W, self.window_size)

        # Generate pooled features
        x_ = x.permute(0, 2, 1).reshape(B, -1, H, W).contiguous()
        x_ = F.adaptive_avg_pool2d(self.act(self.sr(x_)), (pool_H, pool_W)).reshape(B, -1, pool_len).permute(0, 2, 1)
        x_ = self.norm(x_)

        # Generate pooled keys and values
        kv_pool = self.kv(x_).reshape(B, pool_len, 2 * self.num_heads, self.head_dim).permute(0, 2, 1, 3)
        k_pool, v_pool = kv_pool.chunk(2, dim=1)

        if self.is_extrapolation:
            ##Use MLP to generate continuous relative positional bias for pooled features.
            pool_bias = self.cpb_fc2(self.cpb_act(self.cpb_fc1(relative_coords_table))).transpose(0, 1)[:,
                        relative_pos_index.view(-1)].view(-1, N, pool_len)
        else:
            ##Use MLP to generate continuous relative positional bias for pooled features.
            pool_bias = self.cpb_fc2(self.cpb_act(self.cpb_fc1(relative_coords_table))).transpose(0, 1)[:,
                        relative_pos_index.view(-1)].view(-1, self.trained_len, self.trained_pool_len)

            # bilinear interpolation:
            pool_bias = pool_bias.reshape(-1, self.trained_len, self.trained_pool_H, self.trained_pool_W)
            pool_bias = F.interpolate(pool_bias, (pool_H, pool_W), mode='bilinear')
            pool_bias = pool_bias.reshape(-1, self.trained_len, pool_len).transpose(-1, -2).reshape(-1, pool_len,
                                                                                                    self.trained_H,
                                                                                                    self.trained_W)
            pool_bias = F.interpolate(pool_bias, (H, W), mode='bilinear').reshape(-1, pool_len, N).transpose(-1, -2)

        # Compute pooled similarity
        attn_pool = q_norm_scaled @ F.normalize(k_pool, dim=-1).transpose(-2, -1) + pool_bias

        # Concatenate local & pooled similarity matrices and calculate attention weights through the same Softmax
        attn = torch.cat([attn_local, attn_pool], dim=-1).softmax(dim=-1)
        attn = self.attn_drop(attn)

        # Split the attention weights and separately aggregate the values of local & pooled features
        attn_local, attn_pool = torch.split(attn, [self.local_len, pool_len], dim=-1)
        attn_local = (q_norm @ self.learnable_tokens) + self.learnable_bias + attn_local
        x_local = sw_av_cuda.apply(attn_local.type_as(v_local), v_local.contiguous(), H, W, self.window_size)
        x_pool = attn_pool @ v_pool
        x = (x_local + x_pool).transpose(1, 2).reshape(B, N, C)

        x = self.proj(x)
        x = self.proj_drop(x)

        return x


class Attention(nn.Module):
    def __init__(self, dim, input_resolution, num_heads=8, qkv_bias=True, attn_drop=0.,
                 proj_drop=0., is_extrapolation=False):
        super().__init__()
        assert dim % num_heads == 0, f"dim {dim} should be divided by num_heads {num_heads}."

        self.is_extrapolation = is_extrapolation

        if not is_extrapolation:
            # The estimated training resolution is used for bilinear interpolation of the generated relative position bias.
            self.trained_H, self.trained_W = input_resolution
            self.trained_len = self.trained_H * self.trained_W

        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.temperature = nn.Parameter(
            torch.log((torch.ones(num_heads, 1, 1) / 0.24).exp() - 1))  # Initialize softplus(temperature) to 1/0.24.

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.query_embedding = nn.Parameter(
            nn.init.trunc_normal_(torch.empty(self.num_heads, 1, self.head_dim), mean=0, std=0.02))
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

        # mlp to generate continuous relative position bias
        self.cpb_fc1 = nn.Linear(2, 512, bias=True)
        self.cpb_act = nn.ReLU(inplace=True)
        self.cpb_fc2 = nn.Linear(512, num_heads, bias=True)

    def forward(self, x, H, W, relative_pos_index, relative_coords_table, seq_length_scale):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, -1, 3 * self.num_heads, self.head_dim).permute(0, 2, 1, 3)
        q, k, v = qkv.chunk(3, dim=1)

        if self.is_extrapolation:
            # Use MLP to generate continuous relative positional bias
            rel_bias = self.cpb_fc2(self.cpb_act(self.cpb_fc1(relative_coords_table))).transpose(0, 1)[:,
                       relative_pos_index.view(-1)].view(-1, N, N)
        else:
            # Use MLP to generate continuous relative positional bias
            rel_bias = self.cpb_fc2(self.cpb_act(self.cpb_fc1(relative_coords_table))).transpose(0, 1)[:,
                       relative_pos_index.view(-1)].view(-1, self.trained_len, self.trained_len)
            # bilinear interpolation:
            rel_bias = rel_bias.reshape(-1, self.trained_len, self.trained_H, self.trained_W)
            rel_bias = F.interpolate(rel_bias, (H, W), mode='bilinear')
            rel_bias = rel_bias.reshape(-1, self.trained_len, N).transpose(-1, -2).reshape(-1, N, self.trained_H,
                                                                                           self.trained_W)
            rel_bias = F.interpolate(rel_bias, (H, W), mode='bilinear').reshape(-1, N, N).transpose(-1, -2)

        attn = ((F.normalize(q, dim=-1) + self.query_embedding) * F.softplus(
            self.temperature) * seq_length_scale) @ F.normalize(k, dim=-1).transpose(-2, -1) + rel_bias
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)
        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class Block(nn.Module):

    def __init__(self, dim, num_heads, input_resolution, window_size=3, mlp_ratio=4.,
                 qkv_bias=False, drop=0., attn_drop=0.,
                 drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm, sr_ratio=1, is_extrapolation=False):
        super().__init__()
        self.norm1 = norm_layer(dim)
        if sr_ratio == 1:
            self.attn = Attention(
                dim,
                input_resolution,
                num_heads=num_heads,
                qkv_bias=qkv_bias,
                attn_drop=attn_drop,
                proj_drop=drop,
                is_extrapolation=is_extrapolation)
        else:
            self.attn = AggregatedAttention(
                dim,
                input_resolution,
                window_size=window_size,
                num_heads=num_heads,
                qkv_bias=qkv_bias,
                attn_drop=attn_drop,
                proj_drop=drop,
                sr_ratio=sr_ratio,
                is_extrapolation=is_extrapolation)
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = ConvolutionalGLU(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

        # NOTE: drop path for stochastic depth, we shall see if this is better than dropout here
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()

    def forward(self, x, H, W, relative_pos_index, relative_coords_table, seq_length_scale):
        x = x + self.drop_path(
            self.attn(self.norm1(x), H, W, relative_pos_index, relative_coords_table, seq_length_scale))
        x = x + self.drop_path(self.mlp(self.norm2(x), H, W))

        return x


class OverlapPatchEmbed(nn.Module):
    """ Image to Patch Embedding
    """

    def __init__(self, patch_size=7, stride=4, in_chans=3, embed_dim=768):
        super().__init__()

        patch_size = to_2tuple(patch_size)

        assert max(patch_size) > stride, "Set larger patch_size than stride"
        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=stride,
                              padding=(patch_size[0] // 2, patch_size[1] // 2))
        self.norm = nn.LayerNorm(embed_dim)

    def forward(self, x):
        x = self.proj(x)
        _, _, H, W = x.shape
        x = x.flatten(2).transpose(1, 2)
        x = self.norm(x)

        return x, H, W


class TransNeXt(nn.Module):
    '''
    The parameter "img size" is primarily utilized for generating relative spatial coordinates,
    which are used to compute continuous relative positional biases. As this TransNeXt implementation can accept multi-scale inputs,
    it is recommended to set the "img size" parameter to a value close to the resolution of the inference images.
    It is not advisable to set the "img size" parameter to a value exceeding 800x800.
    The "pretrain size" refers to the "img size" used during the initial pre-training phase,
    which is used to scale the relative spatial coordinates for better extrapolation by the MLP.
    For models trained on ImageNet-1K at a resolution of 224x224,
    as well as downstream task models fine-tuned based on these pre-trained weights,
    the "pretrain size" parameter should be set to 224x224.
    '''

    def __init__(self, img_size=224, pretrain_size=None, window_size=[3, 3, 3, None],
                 patch_size=16, in_chans=3, num_classes=1000, embed_dims=[64, 128, 256, 512],
                 num_heads=[1, 2, 4, 8], mlp_ratios=[4, 4, 4, 4], qkv_bias=False, drop_rate=0.,
                 attn_drop_rate=0., drop_path_rate=0., norm_layer=nn.LayerNorm,
                 depths=[3, 4, 6, 3], sr_ratios=[8, 4, 2, 1], num_stages=4, pretrained=None, is_extrapolation=False):
        super().__init__()
        # self.num_classes = num_classes
        self.depths = depths
        self.num_stages = num_stages
        self.window_size = window_size
        self.sr_ratios = sr_ratios
        self.is_extrapolation = is_extrapolation
        self.pretrain_size = pretrain_size or img_size

        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]  # stochastic depth decay rule
        cur = 0

        for i in range(num_stages):
            if not self.is_extrapolation:
                relative_pos_index, relative_coords_table = get_relative_position_cpb(
                    query_size=to_2tuple(img_size // (2 ** (i + 2))),
                    key_size=to_2tuple(img_size // ((2 ** (i + 2)) * sr_ratios[i])),
                    pretrain_size=to_2tuple(pretrain_size // (2 ** (i + 2))))

                self.register_buffer(f"relative_pos_index{i + 1}", relative_pos_index, persistent=False)
                self.register_buffer(f"relative_coords_table{i + 1}", relative_coords_table, persistent=False)

            patch_embed = OverlapPatchEmbed(patch_size=patch_size * 2 - 1 if i == 0 else 3,
                                            stride=patch_size if i == 0 else 2,
                                            in_chans=in_chans if i == 0 else embed_dims[i - 1],
                                            embed_dim=embed_dims[i])

            block = nn.ModuleList([Block(
                dim=embed_dims[i], input_resolution=to_2tuple(img_size // (2 ** (i + 2))), window_size=window_size[i],
                num_heads=num_heads[i], mlp_ratio=mlp_ratios[i], qkv_bias=qkv_bias,
                drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[cur + j], norm_layer=norm_layer,
                sr_ratio=sr_ratios[i], is_extrapolation=is_extrapolation)
                for j in range(depths[i])])
            norm = norm_layer(embed_dims[i])
            cur += depths[i]

            setattr(self, f"patch_embed{i + 1}", patch_embed)
            setattr(self, f"block{i + 1}", block)
            setattr(self, f"norm{i + 1}", norm)

        # classification head
        # self.head = nn.Linear(embed_dims[3], num_classes) if num_classes > 0 else nn.Identity()

        for n, m in self.named_modules():
            self._init_weights(m, n)
        if pretrained:
            self.init_weights(pretrained)

    def _init_weights(self, m: nn.Module, name: str = ''):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.Conv2d):
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            fan_out //= m.groups
            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if m.bias is not None:
                m.bias.data.zero_()
        elif isinstance(m, (nn.LayerNorm, nn.GroupNorm, nn.BatchNorm2d)):
            nn.init.zeros_(m.bias)
            nn.init.ones_(m.weight)

    def init_weights(self, pretrained=None):
        if isinstance(pretrained, str):
            logger = MMLogger.get_current_instance()
            load_checkpoint(self, pretrained, strict=False, logger=logger)

    @torch.jit.ignore
    def no_weight_decay(self):
        return {}

    @torch.jit.ignore
    def no_weight_decay_keywords(self):
        return {'query_embedding', 'relative_pos_bias_local', 'cpb', 'temperature'}

    def get_classifier(self):
        return self.head

    def reset_classifier(self, num_classes, global_pool=''):
        self.num_classes = num_classes
        self.head = nn.Linear(self.embed_dim, num_classes) if num_classes > 0 else nn.Identity()

    def forward_features(self, x):
        B = x.shape[0]
        outs = []

        for i in range(self.num_stages):
            patch_embed = getattr(self, f"patch_embed{i + 1}")
            block = getattr(self, f"block{i + 1}")
            norm = getattr(self, f"norm{i + 1}")
            x, H, W = patch_embed(x)
            sr_ratio = self.sr_ratios[i]
            if self.is_extrapolation:
                relative_pos_index, relative_coords_table = get_relative_position_cpb(query_size=(H, W),
                                                                                      key_size=(
                                                                                          H // sr_ratio,
                                                                                          W // sr_ratio),
                                                                                      pretrain_size=to_2tuple(
                                                                                          self.pretrain_size // (
                                                                                                  2 ** (i + 2))),
                                                                                      device=x.device)
            else:
                relative_pos_index = getattr(self, f"relative_pos_index{i + 1}")
                relative_coords_table = getattr(self, f"relative_coords_table{i + 1}")

            with torch.no_grad():
                if i != (self.num_stages - 1):
                    local_seq_length = get_seqlen_scale((H, W), self.window_size[i], device=x.device)
                    seq_length_scale = torch.log(local_seq_length + (H // sr_ratio) * (W // sr_ratio))
                else:
                    seq_length_scale = torch.log(torch.as_tensor((H // sr_ratio) * (W // sr_ratio), device=x.device))

            for blk in block:
                x = blk(x, H, W, relative_pos_index, relative_coords_table, seq_length_scale)

            x = norm(x)
            x = x.reshape(B, H, W, -1).permute(0, 3, 1, 2).contiguous()
            outs.append(x)


        return outs

    def forward(self, x):
        x = self.forward_features(x)
        # x = self.head(x)

        return x


@MODELS.register_module()
class transnext_tiny(TransNeXt):
    def __init__(self, **kwargs):
        super().__init__(window_size=[3, 3, 3, None],
                         patch_size=4, embed_dims=[72, 144, 288, 576], num_heads=[3, 6, 12, 24],
                         mlp_ratios=[8, 8, 4, 4], qkv_bias=True,
                         norm_layer=partial(nn.LayerNorm, eps=1e-6), depths=[2, 2, 15, 2], sr_ratios=[8, 4, 2, 1],
                         drop_rate=0.0, drop_path_rate=0.3,
                         pretrained=kwargs['pretrained'], img_size=kwargs['img_size'],
                         pretrain_size=kwargs['pretrain_size'])


@MODELS.register_module()
class transnext_small(TransNeXt):
    def __init__(self, **kwargs):
        super().__init__(window_size=[3, 3, 3, None],
                         patch_size=4, embed_dims=[72, 144, 288, 576], num_heads=[3, 6, 12, 24],
                         mlp_ratios=[8, 8, 4, 4], qkv_bias=True,
                         norm_layer=partial(nn.LayerNorm, eps=1e-6), depths=[5, 5, 22, 5], sr_ratios=[8, 4, 2, 1],
                         drop_rate=0.0, drop_path_rate=0.5,
                         pretrained=kwargs['pretrained'], img_size=kwargs['img_size'],
                         pretrain_size=kwargs['pretrain_size'])


@MODELS.register_module()
class transnext_base(TransNeXt):
    def __init__(self, **kwargs):
        super().__init__(window_size=[3, 3, 3, None],
                         patch_size=4, embed_dims=[96, 192, 384, 768], num_heads=[4, 8, 16, 32],
                         mlp_ratios=[8, 8, 4, 4], qkv_bias=True,
                         norm_layer=partial(nn.LayerNorm, eps=1e-6), depths=[5, 5, 10, 5], sr_ratios=[8, 4, 2, 1],
                         drop_rate=0.0, drop_path_rate=0.6,
                         pretrained=kwargs['pretrained'], img_size=kwargs['img_size'],
                         pretrain_size=kwargs['pretrain_size'])
