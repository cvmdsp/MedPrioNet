import timm
import torch
import torch.nn as nn
from functools import partial
import torch.nn.functional as F
from timm.models.vision_transformer import VisionTransformer
from timm.models.registry import register_model
from timm.models.layers import trunc_normal_, PatchEmbed, DropPath
from timm.models.helpers import named_apply
import math
from torch.autograd import Variable

def init_weights_vit_timm(module: nn.Module, name: str = ''):
    """ ViT weight initialization, original timm impl (for reproducibility) """
    if isinstance(module, nn.Linear):
        trunc_normal_(module.weight, std=.02)
        if module.bias is not None:
            nn.init.zeros_(module.bias)
    elif hasattr(module, 'init_weights'):
        module.init_weights()


class PatchEmbedConv(nn.Module):

    def __init__(self, img_size=224, patch_size=16, in_chans=3, embed_dim=768, norm_layer=None, stem_channel=32):
        super().__init__()
        img_size = (img_size, img_size)
        patch_size_origin = (patch_size, patch_size)

        self.img_size = img_size
        self.patch_size = patch_size_origin
        self.grid_size = (img_size[0] // patch_size_origin[0], img_size[1] // patch_size_origin[1])
        self.num_patches = self.grid_size[0] * self.grid_size[1]

        self.stem_conv1 = nn.Conv2d(3, embed_dim, kernel_size=4, stride=4)
        self.stem_norm1 = norm_layer(embed_dim, eps=1e-6, data_format="channels_first") if norm_layer else nn.Identity()

        # self.stem_conv1 = nn.Conv2d(3, stem_channel, kernel_size=3, stride=2, padding=1, bias=True)
        # self.stem_relu1 = nn.GELU()
        # self.stem_norm1 = nn.BatchNorm2d(stem_channel, eps=1e-5)
        #
        # self.stem_conv2 = nn.Conv2d(stem_channel, stem_channel, kernel_size=3, stride=1, padding=1, bias=True)
        # self.stem_relu2 = nn.GELU()
        # self.stem_norm2 = nn.BatchNorm2d(stem_channel, eps=1e-5)
        #
        # self.stem_conv3 = nn.Conv2d(stem_channel, stem_channel, kernel_size=3, stride=1, padding=1, bias=True)
        # self.stem_relu3 = nn.GELU()
        # self.stem_norm3 = nn.BatchNorm2d(stem_channel, eps=1e-5)
        #
        # self.proj = nn.Conv2d(stem_channel, embed_dim, kernel_size=7, stride=2, padding=3)
        # self.norm = norm_layer(embed_dim, eps=1e-6, data_format="channels_first") if norm_layer else nn.Identity()

    def forward(self, x):
        B, C, H, W = x.shape
        assert H == self.img_size[0], f"Input image height ({H}) doesn't match model ({self.img_size[0]})."
        assert W == self.img_size[1], f"Input image width ({W}) doesn't match model ({self.img_size[1]})."
        # x = self.stem_conv1(x)
        # x = self.stem_relu1(x)
        # x = self.stem_norm1(x)
        # x = self.stem_conv2(x)
        # x = self.stem_relu2(x)
        # x = self.stem_norm2(x)
        # x = self.stem_conv3(x)
        # x = self.stem_relu3(x)
        # x = self.stem_norm3(x)
        # x = self.proj(x)
        # # x = x.flatten(2).transpose(1, 2)  # BCHW -> BNC
        # x = self.norm(x)
        # return x
        x = self.stem_conv1(x)
        x = self.stem_norm1(x)
        return x

class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0., bn=0):
        super().__init__()
        self.bn = bn
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_features, hidden_features, 1, 1, 0, bias=True),
            nn.GELU(),
            nn.BatchNorm2d(hidden_features, eps=1e-5),
        )
        self.proj = nn.Conv2d(hidden_features, hidden_features, 3, 1, 1, groups=hidden_features)
        self.proj_act = nn.GELU()
        self.proj_bn = nn.BatchNorm2d(hidden_features, eps=1e-5)
        self.conv2 = nn.Sequential(
            nn.Conv2d(hidden_features, out_features, 1, 1, 0, bias=True),
            nn.BatchNorm2d(out_features, eps=1e-5),
        )
        self.drop = nn.Dropout(drop)

    def forward(self, x, H, W):
        # IRFFN
        B, N, C = x.shape
        bottleneck = x[:, -self.bn:, :]
        x = x[:, :-self.bn, :] if self.bn != 0 else x
        x = x.permute(0, 2, 1).contiguous().reshape(B, C, H, W)
        x = self.conv1(x)
        x = self.drop(x)
        x = self.proj(x) + x
        x = self.proj_act(x)
        x = self.proj_bn(x)
        x = self.conv2(x)
        x = x.flatten(2).permute(0, 2, 1).contiguous()
        x = torch.cat([x, bottleneck], dim=1) if self.bn != 0 else x
        x = self.drop(x)
        return x


class Attention(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None,
                 attn_drop=0., proj_drop=0., qk_ratio=1, sr_ratio=1, bn=0):
        super().__init__()
        self.bn = bn
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5
        self.qk_dim = dim // qk_ratio

        self.q = nn.Linear(dim, self.qk_dim, bias=qkv_bias)
        self.k = nn.Linear(dim, self.qk_dim, bias=qkv_bias)
        self.v = nn.Linear(dim, dim, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

        self.sr_ratio = sr_ratio
        # same as PVTv1
        if self.sr_ratio > 1:
            self.sr = nn.Sequential(
                nn.Conv2d(dim, dim, kernel_size=sr_ratio, stride=sr_ratio, groups=dim, bias=True),
                nn.BatchNorm2d(dim, eps=1e-5),
            )

    def forward(self, x, H, W, relative_pos):
        B, N, C = x.shape
        q = self.q(x).reshape(B, N, self.num_heads, self.qk_dim // self.num_heads).permute(0, 2, 1, 3).contiguous()

        if self.sr_ratio > 1:
            # adapt to cross phase tokens
            bottleneck = x[:, -self.bn:, :]
            x = x[:, :-self.bn, :] if self.bn != 0 else x
            x_ = x.permute(0, 2, 1).contiguous().reshape(B, C, H, W)
            x_ = self.sr(x_).reshape(B, C, -1).permute(0, 2, 1).contiguous()
            x_ = torch.cat([x_, bottleneck], dim=1) if self.bn != 0 else x_
            k = self.k(x_).reshape(B, -1, self.num_heads, self.qk_dim // self.num_heads).permute(0, 2, 1, 3).contiguous()
            v = self.v(x_).reshape(B, -1, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3).contiguous()
        else:
            k = self.k(x).reshape(B, N, self.num_heads, self.qk_dim // self.num_heads).permute(0, 2, 1, 3).contiguous()
            v = self.v(x).reshape(B, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3).contiguous()

        attn = (q @ k.transpose(-2, -1)) * self.scale + relative_pos
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)
        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class Block1(nn.Module):
    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm, qk_ratio=1, sr_ratio=1, bn=0, dwflag=True):
        super().__init__()
        self.bn = bn
        self.dwflag = dwflag
        self.norm1 = norm_layer(dim)
        self.attn = Attention(
            dim, num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale,
            attn_drop=attn_drop, proj_drop=drop, qk_ratio=qk_ratio, sr_ratio=sr_ratio, bn=bn)

        # NOTE: drop path for stochastic depth, we shall see if this is better than dropout here
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop, bn=bn)
        if dwflag:
            self.proj = nn.Conv2d(dim, dim, 3, 1, 1, groups=dim)

    def forward(self, x, H, W, relative_pos):
        x = x + self.drop_path(self.attn(self.norm1(x), H, W, relative_pos))
        x = x + self.drop_path(self.mlp(self.norm2(x), H, W))
        # a part of conv down sampler
        if self.dwflag:
            bottleneck = x[:, -self.bn:, :]
            x = x[:, :-self.bn, :] if self.bn != 0 else x
            B, N, C = x.shape
            cnn_feat = x.permute(0, 2, 1).contiguous().reshape(B, C, H, W)
            x = self.proj(cnn_feat) + cnn_feat
            x = x.flatten(2).permute(0, 2, 1).contiguous()
            x = torch.cat([x, bottleneck], dim=1) if self.bn != 0 else x
        return x


class LayerNorm(nn.Module):
    r""" LayerNorm that supports two data formats: channels_last (default) or channels_first.
    The ordering of the dimensions in the inputs. channels_last corresponds to inputs with
    shape (batch_size, height, width, channels) while channels_first corresponds to inputs
    with shape (batch_size, channels, height, width).
    """

    def __init__(self, normalized_shape, eps=1e-6, data_format="channels_last"):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.eps = eps
        self.data_format = data_format
        if self.data_format not in ["channels_last", "channels_first"]:
            raise NotImplementedError
        self.normalized_shape = (normalized_shape,)

    def forward(self, x):
        if self.data_format == "channels_last":
            return F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
        elif self.data_format == "channels_first":
            u = x.mean(1, keepdim=True)
            s = (x - u).pow(2).mean(1, keepdim=True)
            x = (x - u) / torch.sqrt(s + self.eps)
            x = self.weight[:, None, None] * x + self.bias[:, None, None]
            return x

class Block(nn.Module):
    r""" ConvNeXt Block. There are two equivalent implementations:
    (1) DwConv -> LayerNorm (channels_first) -> 1x1 Conv -> GELU -> 1x1 Conv; all in (N, C, H, W)
    (2) DwConv -> Permute to (N, H, W, C); LayerNorm (channels_last) -> Linear -> GELU -> Linear; Permute back
    We use (2) as we find it slightly faster in PyTorch

    Args:
        dim (int): Number of input channels.
        drop_path (float): Stochastic depth rate. Default: 0.0
        layer_scale_init_value (float): Init value for Layer Scale. Default: 1e-6.
    """

    def __init__(self, dim, drop_path=0., layer_scale_init_value=1e-6):
        super().__init__()
        self.dwconv = nn.Conv2d(dim, dim, kernel_size=7, padding=3, groups=dim)  # depthwise conv
        self.norm = LayerNorm(dim, eps=1e-6)
        self.pwconv1 = nn.Linear(dim, 4 * dim)  # pointwise/1x1 convs, implemented with linear layers
        self.act = nn.GELU()
        self.pwconv2 = nn.Linear(4 * dim, dim)
        self.gamma = nn.Parameter(layer_scale_init_value * torch.ones((dim)),
                                  requires_grad=True) if layer_scale_init_value > 0 else None
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()

    def forward(self, x):
        input = x
        x = self.dwconv(x)
        x = x.permute(0, 2, 3, 1)  # (N, C, H, W) -> (N, H, W, C)
        x = self.norm(x)
        x = self.pwconv1(x)
        x = self.act(x)
        x = self.pwconv2(x)
        if self.gamma is not None:
            x = self.gamma * x
        x = x.permute(0, 3, 1, 2)  # (N, H, W, C) -> (N, C, H, W)

        x = input + self.drop_path(x)
        return x

class InceptionBlock(nn.Module):

    def __init__(self, dim, drop_path=0., layer_scale_init_value=1e-6, square_kernel_size=3, band_kernel_size=11):
        super().__init__()
        self.dwconv = InceptionDWConv2d(dim, square_kernel_size, band_kernel_size)  # InceptionDWConv2d
        self.norm = LayerNorm(dim, eps=1e-6)
        self.pwconv1 = nn.Linear(dim, 4 * dim)  # pointwise/1x1 convs, implemented with linear layers
        self.act = nn.GELU()
        self.pwconv2 = nn.Linear(4 * dim, dim)
        self.gamma = nn.Parameter(layer_scale_init_value * torch.ones((dim)),
                                  requires_grad=True) if layer_scale_init_value > 0 else None
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()

    def forward(self, x):
        input = x
        x = self.dwconv(x)
        x = x.permute(0, 2, 3, 1)  # (N, C, H, W) -> (N, H, W, C)
        x = self.norm(x)
        x = self.pwconv1(x)
        x = self.act(x)
        x = self.pwconv2(x)
        if self.gamma is not None:
            x = self.gamma * x
        x = x.permute(0, 3, 1, 2)  # (N, H, W, C) -> (N, C, H, W)

        x = input + self.drop_path(x)
        return x

class InceptionDWConv2d(nn.Module):
    """ Inception depthwise convolution with dilated convolution
    """

    def __init__(self, in_channels, square_kernel_size=3, band_kernel_size=11, branch_ratio=0.125, dilation=1):
        super().__init__()

        gc = int(in_channels * branch_ratio)  # channel numbers of a convolution branch

        # # 定义空洞卷积
        self.dwconv_hw_dilated = nn.Conv2d(gc, gc, square_kernel_size,
                                           padding=(dilation * (square_kernel_size - 1) // 2),
                                           groups=gc, dilation=dilation)
        # 原始深度可分离卷积
        self.dwconv_hw = nn.Conv2d(gc, gc, square_kernel_size, padding=square_kernel_size // 2, groups=gc)
        self.dwconv_w = nn.Conv2d(gc, gc, kernel_size=(1, band_kernel_size), padding=(0, band_kernel_size // 2),
                                  groups=gc)
        self.dwconv_h = nn.Conv2d(gc, gc, kernel_size=(band_kernel_size, 1), padding=(band_kernel_size // 2, 0),
                                  groups=gc)

        self.split_indexes = (in_channels - 3 * gc, gc, gc, gc)

    def forward(self, x):
        x_id, x_hw, x_w, x_h = torch.split(x, self.split_indexes, dim=1)

        # 应用空洞卷积
        hw_out = self.dwconv_hw_dilated(x_hw)
        # w_out = self.dwconv_w_dilated(x_w)
        # h_out = self.dwconv_h_dilated(x_h)

        # 然后再应用原始的深度可分离卷积
        return torch.cat(
            (x_id, self.dwconv_hw(hw_out), self.dwconv_w(x_w), self.dwconv_h(x_h)),
            dim=1,
        )

class IRMLP(nn.Module):
    def __init__(self, inp_dim, out_dim):
        super(IRMLP, self).__init__()
        self.conv1 = Conv(inp_dim, inp_dim, 3, relu=False, bias=False, group=inp_dim)
        self.conv2 = Conv(inp_dim, inp_dim * 4, 1, relu=False, bias=False)
        self.conv3 = Conv(inp_dim * 4, out_dim, 1, relu=False, bias=False, bn=True)
        self.gelu = nn.GELU()
        self.bn1 = nn.BatchNorm2d(inp_dim)

    def forward(self, x):

        residual = x
        out = self.conv1(x)
        out = self.gelu(out)
        out += residual

        out = self.bn1(out)
        out = self.conv2(out)
        out = self.gelu(out)
        out = self.conv3(out)

        return out


import torch
import torch.nn as nn
import torch.nn.functional as F


class FeatureFusion(nn.Module):
    def __init__(self):
        super(FeatureFusion, self).__init__()

        self.conv_x1 = nn.Sequential(
            nn.Conv2d(64, 512, kernel_size=1),
            nn.ReLU()  # 添加激活函数
        )
        self.conv_x2 = nn.Sequential(
            nn.Conv2d(128, 512, kernel_size=1),
            nn.ReLU()  # 添加激活函数
        )
        self.conv_x3 = nn.Sequential(
            nn.Conv2d(256, 512, kernel_size=1),
            nn.ReLU()  # 添加激活函数
        )

    def forward(self, x1, x2, x3, x4):
        x1_out = self.conv_x1(x1)  # (B, 512, 56, 56)
        x2_out = self.conv_x2(x2)  # (B, 512, 28, 28)
        x3_out = self.conv_x3(x3)  # (B, 512, 14, 14)

        # 特征调整至相同尺寸
        x1_resized = F.interpolate(x1_out, size=x4.shape[2:], mode='bilinear', align_corners=False)
        x2_resized = F.interpolate(x2_out, size=x4.shape[2:], mode='bilinear', align_corners=False)
        x3_resized = F.interpolate(x3_out, size=x4.shape[2:], mode='bilinear', align_corners=False)

        # 进行特征拼接
        fused_features = torch.cat((x1_resized, x2_resized, x3_resized, x4), dim=1)  # (B, 2048, 7, 7)

        return fused_features


class Conv(nn.Module):
    def __init__(self, inp_dim, out_dim, kernel_size=3, stride=1, bn=False, relu=True, bias=True, group=1):
        super(Conv, self).__init__()
        self.inp_dim = inp_dim
        self.conv = nn.Conv2d(inp_dim, out_dim, kernel_size, stride, padding=(kernel_size-1)//2, bias=bias)
        self.relu = None
        self.bn = None
        if relu:
            self.relu = nn.ReLU(inplace=True)
        if bn:
            self.bn = nn.BatchNorm2d(out_dim)

    def forward(self, x):
        assert x.size()[1] == self.inp_dim, "{} {}".format(x.size()[1], self.inp_dim)
        x = self.conv(x)
        if self.bn is not None:
            x = self.bn(x)
        if self.relu is not None:
            x = self.relu(x)
        return x

class HFF_block(nn.Module):
    def __init__(self, ch_1, ch_2, ch_int, drop_rate=0.):
        super(HFF_block, self).__init__()
        self.maxpool=nn.AdaptiveMaxPool2d(1)
        self.avgpool=nn.AdaptiveAvgPool2d(1)
        # self.se=nn.Sequential(
        #     nn.Conv2d(ch_2, ch_2 // r_2, 1,bias=False),
        #     nn.ReLU(),
        #     nn.Conv2d(ch_2 // r_2, ch_2, 1,bias=False)
        # )
        self.sigmoid = nn.Sigmoid()
        self.spatial = Conv(2, 1, 7, bn=True, relu=False, bias=False)
        self.W_l = Conv(ch_1, ch_int, 1, bn=True, relu=False)
        self.W_g = Conv(ch_2, ch_int, 1, bn=True, relu=False)
        self.Avg = nn.AvgPool2d(2, stride=2)
        self.Updim = Conv(ch_int//2, ch_int, 1, bn=True, relu=True)
        self.norm1 = LayerNorm(ch_int * 3, eps=1e-6, data_format="channels_first")
        self.norm2 = LayerNorm(ch_int * 2, eps=1e-6, data_format="channels_first")
        self.norm3 = LayerNorm(ch_1 + ch_2 + ch_int, eps=1e-6, data_format="channels_first")
        self.W3 = Conv(ch_int * 3, ch_int, 1, bn=True, relu=False)
        self.W = Conv(ch_int * 2, ch_int, 1, bn=True, relu=False)

        self.gelu = nn.GELU()

        # self.residual = IRMLP(ch_1 + ch_2 + ch_int, ch_out)
        self.drop_path = DropPath(drop_rate) if drop_rate > 0. else nn.Identity()

    def forward(self, l, g):

        T_l = self.Updim(l)
        T_l = self.Avg(T_l)
        T_g = self.W_g(g)
        X_f = torch.cat([T_l, T_g], 1)
        X_f = self.norm2(X_f)
        X_f = self.W(X_f)
        X_f = self.gelu(X_f)


        # spatial attention for ConvNeXt branch
        X_f_jump = X_f
        max_result, _ = torch.max(X_f, dim=1, keepdim=True)
        avg_result = torch.mean(X_f, dim=1, keepdim=True)
        result = torch.cat([max_result, avg_result], 1)
        X_f = self.spatial(result)
        X_f = self.sigmoid(X_f) * X_f_jump

        # # channel attetion for transformer branch
        # g_jump = g
        # max_result=self.maxpool(g)
        # avg_result=self.avgpool(g)
        # max_out=self.se(max_result)
        # avg_out=self.se(avg_result)
        # g = self.sigmoid(max_out+avg_out) * g_jump

        X_f = self.drop_path(X_f)

        return X_f

class TemporalAttentionModule(nn.Module):
    """Large Kernel Attention for SimVP"""

    def __init__(self, dim, kernel_size, dilation=3, reduction=16):
        super().__init__()
        d_k = 2 * dilation - 1
        d_p = (d_k - 1) // 2
        dd_k = kernel_size // dilation + ((kernel_size // dilation) % 2 - 1)
        dd_p = (dilation * (dd_k - 1) // 2)

        self.conv0 = nn.Conv2d(dim, dim, d_k, padding=d_p, groups=dim)
        self.conv_spatial = nn.Conv2d(
            dim, dim, dd_k, stride=1, padding=dd_p, groups=dim, dilation=dilation)
        self.conv1 = nn.Conv2d(dim, dim, 1)

        self.reduction = max(dim // reduction, 4)
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(dim, dim // self.reduction, bias=False), # reduction
            nn.ReLU(True),
            nn.Linear(dim // self.reduction, dim, bias=False), # expansion
            nn.Sigmoid()
        )

    def forward(self, x):
        B, T, C, H, W = x.shape
        x = x.reshape(B, T*C, H, W)
        u = x.clone()
        attn = self.conv0(x)           # depth-wise conv
        attn = self.conv_spatial(attn) # depth-wise dilation convolution
        f_x = self.conv1(attn)         # 1x1 conv
        # append a se operation
        b, c, _, _ = x.size()
        se_atten = self.avg_pool(x).view(b, c)
        se_atten = self.fc(se_atten).view(b, c, 1, 1)
        out = se_atten * f_x * u
        out = out.reshape(B, T, C, H, W)
        return out

class PositionalEncoder(nn.Module):
    def __init__(self, d_model, max_seq_len=16):
        super(PositionalEncoder, self).__init__()
        self.d_model = d_model
        pe = torch.zeros(max_seq_len, d_model)
        for pos in range(max_seq_len):
            for i in range(0, d_model, 2):
                pe[pos, i] = \
                    math.sin(pos / (10000 ** ((2 * i) / d_model)))
                pe[pos, i + 1] = \
                    math.cos(pos / (10000 ** ((2 * (i + 1)) / d_model)))
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        """
        :param x: [B, T, C, H, W]
        :param self.pe [1, max_seq_len, d_model]
        :return:
        """
        # make embeddings relatively larger
        x = x * math.sqrt(self.d_model)
        # add constant to embedding
        seq_len = x.size(1)
        batch_size = x.size(0)
        num_feature = x.size(2)
        spatial_h = x.size(3)
        spatial_w = x.size(4)
        z = Variable(self.pe[:, :seq_len], requires_grad=False)
        z = z.unsqueeze(-1).unsqueeze(-1)  # [1, T, d_model, 1, 1]
        z = z.expand(batch_size, seq_len, num_feature, spatial_h, spatial_w)
        x = x + z
        return x

class AttentionTF(nn.Module):
    """
    Self-attention based temporal fusion
    """

    def __init__(self, in_channels, out_channels, grid=(4, 4), original_scale=(7, 7), bias=False):
        super(AttentionTF, self).__init__()

        self.posenc = PositionalEncoder(d_model=in_channels, max_seq_len=16)
        self.out_channels = out_channels
        self.grids = grid
        self.key_conv = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=bias)
        self.query_conv = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=bias)
        self.value_conv = nn.Conv2d(in_channels, in_channels, kernel_size=1, bias=bias)
        self.avg = nn.AdaptiveAvgPool2d(grid)
        self.upsample = nn.Upsample(size=original_scale, mode='nearest')

    # standard attention layer
    def attention(self, q, k, v, d_k):
        # [B,T,C,H,W]
        scores = torch.sum(q * k, 2) / math.sqrt(d_k)
        scores = F.softmax(scores, dim=1)
        scores = self.upsample(scores)  # [B,T,1, H,W]
        scores = scores.unsqueeze(2).expand_as(v)
        output = scores * v
        output = torch.sum(output, 1)
        return output

    def forward(self, x, query_x):
        batch, t, channels, height, width = x.size()
        dim = self.out_channels
        DPRANk = query_x
        # value maps [B,T,C,H,W]
        v_out = self.value_conv(x.view((-1,) + x.shape[2:]))
        v = v_out.view((-1, t) + v_out.shape[1:])
        # key
        x = self.avg(x.view((-1,) + x.shape[2:]))
        x = self.posenc(x.view((-1, t) + x.shape[1:]))  # temporal position encoding
        k_out = self.key_conv(x.view((-1,) + x.shape[2:]))
        k = k_out.view((-1, t) + k_out.shape[1:])
        # query [B,C,H,W]
        q_out = self.query_conv(self.avg(query_x))
        q = q_out.unsqueeze(1).expand_as(k)

        out = self.attention(q, k, v, d_k=dim)
        out += DPRANk

        return out


class max_q(nn.Module):
    def __init__(self, inchannel, in_features):
        super(max_q, self).__init__()
        self.fc1 = nn.Linear(in_features, in_features//4, bias=True)
        self.fc2 = nn.Linear(in_features//4, in_features,bias=True)
        self.act = nn.ReLU(inplace=True)

    def forward(self, x):
        x_mean = torch.mean(x, dim=(3, 4))
        x_fc1 = self.fc1(x_mean)
        x_relu = self.act(x_fc1)
        x_fc2 = self.fc2(x_relu)

        att_weight = nn.functional.softmax(x_fc2, dim=1)
        max_index = torch.argmax(att_weight, dim=1)
        # 调整 max_index 的形状
        max_index = max_index.unsqueeze(1)
        # 使用 gather 从 scale_1 中获取对应的张量
        # selected = torch.gather(x*att_weight, dim=1, index=max_index)
        selected = torch.gather(x, dim=1, index=max_index)
        selected = selected.squeeze(1)
        return selected

def extract_important_frames(features):
    """
    提取最重要的关键帧。

    参数:
        features: numpy.ndarray, 特征数组，形状为 (B, T, C, H, W)

    返回:
        key_frames: numpy.ndarray, 选出的关键帧，形状为 (B, C, H, W)
    """
    # 计算每一帧的方差
    variances = torch.var(features, dim=(2, 3, 4), unbiased=False)  # 沿着C, H, W轴计算方差，结果形状为(B, T)

    # 找到方差最大的帧的索引
    key_frame_indices = torch.argmax(variances, dim=1)  # 获取每个视频中方差最大帧的索引

    # 提取关键帧
    B, T, C, H, W = features.shape
    key_frames = features[torch.arange(B).unsqueeze(1),key_frame_indices.unsqueeze(1)].squeeze(1)

    return key_frames

class Updownblock(nn.Module):
    def __init__(self, n_feats):
        super(Updownblock, self).__init__()
        self.encoder = mixblock(n_feats)
        self.decoder_high = mixblock(n_feats)
        self.decoder_low = nn.Sequential(mixblock(n_feats), mixblock(n_feats), mixblock(n_feats))

        self.alise = nn.Conv2d(n_feats,n_feats,1,1,0,bias=False)  # one_module(n_feats)
        self.alise2 = nn.Conv2d(n_feats*2,n_feats,3,1,1,bias=False)  # one_module(n_feats)
        self.down = nn.AvgPool2d(kernel_size=2)
        self.att = CALayer(n_feats)
        self.raw_alpha=nn.Parameter(torch.ones(1))
        # fill 0
        self.raw_alpha.data.fill_(0)
        self.ega=selfAttention(n_feats, n_feats)

    def forward(self, x):
        x1 = self.encoder(x)
        x2 = self.down(x1)
        high = x1 - F.interpolate(x2, size=x.size()[-2:], mode='bilinear', align_corners=True)
        high=high+self.ega(high,high)*self.raw_alpha
        x2=self.decoder_low(x2)
        x3 = x2
        high1 = self.decoder_high(high)
        x4 = F.interpolate(x3, size=x.size()[-2:], mode='bilinear', align_corners=True)
        return self.alise(self.att(self.alise2(torch.cat([x4, high1], dim=1)))) + x

class mixblock(nn.Module):
    def __init__(self, n_feats):
        super(mixblock, self).__init__()
        self.conv1 = nn.Sequential(nn.Conv2d(n_feats,n_feats,3,1,1,bias=False),nn.GELU())

    def forward(self, x):
        # return self.alpha*self.conv1(x)+self.beta*self.conv2(x)
        return self.conv1(x)
class CALayer(nn.Module):
    def __init__(self, channel, reduction=16):
        super(CALayer, self).__init__()
        # global average pooling: feature --> point
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        # feature channel downscale and upscale --> channel weight
        self.conv_du = nn.Sequential(
            nn.Conv2d(channel, channel // reduction, 1, padding=0, bias=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(channel // reduction, channel, 1, padding=0, bias=True),
            nn.Sigmoid()
        )

    def forward(self, x):
        y = self.avg_pool(x)
        y = self.conv_du(y)
        return x * y
class selfAttention(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(selfAttention, self).__init__()
        self.query_conv = nn.Conv2d(in_channels, in_channels, kernel_size=1)
        self.key_conv = nn.Conv2d(in_channels, in_channels, kernel_size=1)
        self.value_conv = nn.Conv2d(in_channels, in_channels, kernel_size=1)
        self.scale = 1.0 / (out_channels ** 0.5)

    def forward(self, feature, feature_map):
        query = self.query_conv(feature)
        key = self.key_conv(feature)
        value = self.value_conv(feature)
        attention_scores = torch.matmul(query, key.transpose(-2, -1))
        attention_scores = attention_scores * self.scale

        attention_weights = F.softmax(attention_scores, dim=-1)

        attended_values = torch.matmul(attention_weights, value)

        output_feature_map = (feature_map + attended_values)

        return output_feature_map

class MultiPhaseVisionTransformer(VisionTransformer):

    def __init__(self,
                 use_bottleneck=True,
                 bottleneck_n=8,  # cross phase tokens number
                 phase_num=8,  # phase number
                 fusion_layer=9,  # fusion layer
                 depth=12,  # depth of the last 2 stages
                 drop_rate=0.,
                 attn_drop_rate=0.1,
                 drop_path_rate=0.2,
                 num_classes=2,
                 num_heads=8,
                 img_size=224,
                 patch_size=16,
                 embed_dim=512,
                 in_chans=1,
                 pre_norm=False,
                 *args, **kwargs):
        weight_init = kwargs.pop('weight_init', '')
        super().__init__(depth=depth, drop_rate=drop_rate, num_classes=num_classes, num_heads=num_heads, img_size=img_size,
                         patch_size=patch_size, embed_dim=embed_dim, in_chans=in_chans, *args, **kwargs, weight_init='')
        norm_layer = partial(LayerNorm, eps=1e-6)
        layer_scale_init_value = 1e-6
        self.bottleneck_n = bottleneck_n
        self.phase_num = phase_num
        self.fusion_layer = fusion_layer
        self.depth = depth
        self.in_chans = in_chans
        self.img_size = img_size
        sr_ratios = [8, 4, 2, 1]
        embed_dim_y = 512
        # init the patch embedding layer, which includes the conv encoder
        del self.patch_embed
        self.patch_embeds = []
        for i in range(phase_num):
            self.patch_embeds.append(PatchEmbedConv(img_size=img_size, patch_size=patch_size, in_chans=in_chans,
                                     embed_dim=64, norm_layer=norm_layer))
        self.patch_embeds = nn.ModuleList(self.patch_embeds)

        self.temporal_dims = [0]*phase_num

        # transliver does not use absolute position embedding and class token
        del self.pos_embed
        del self.cls_token

        self.norm_pres = []
        for i in range(phase_num):
            self.norm_pres.append(norm_layer(self.embed_dim) if pre_norm else nn.Identity())
        self.norm_pres = nn.ModuleList(self.norm_pres)

        # init cross phase tokens
        if use_bottleneck:
            # remember tile (bs,1,1) when forward
            self.bottleneck = nn.Parameter(torch.Tensor(1, bottleneck_n, self.embed_dim))
        else:
            self.bottleneck = None

        # init conv downsampler
        self.down_sample_a = []
        self.down_sample_b = []
        self.down_sample_c = []
        stem_a = nn.Sequential(
            LayerNorm(64, eps=1e-6, data_format="channels_first"),
            PatchEmbed(img_size=img_size // 4, patch_size=2, in_chans=64, embed_dim=128, flatten=None),
            # LayerNorm(128, eps=1e-6, data_format="channels_first")
        )
        stem_b = nn.Sequential(
            LayerNorm(128, eps=1e-6, data_format="channels_first"),
            PatchEmbed(img_size=img_size // 8, patch_size=2, in_chans=128, embed_dim=256, flatten=None)
            # LayerNorm(256, eps=1e-6, data_format="channels_first")
        )
        stem_c = nn.Sequential(
            LayerNorm(256, eps=1e-6, data_format="channels_first"),
            PatchEmbed(img_size=img_size // 16, patch_size=2, in_chans=256, embed_dim=embed_dim, flatten=None)
            # LayerNorm(embed_dim, eps=1e-6, data_format="channels_first")
        )
        for i in range(phase_num):
            self.down_sample_a.append(stem_a)
            self.down_sample_b.append(stem_b)
            self.down_sample_c.append(stem_c)
        self.down_sample_a = nn.ModuleList(self.down_sample_a)
        self.down_sample_b = nn.ModuleList(self.down_sample_b)
        self.down_sample_c = nn.ModuleList(self.down_sample_c)

        # init relative position embedding in each stage
        self.relative_pos_a = []
        self.relative_pos_b = []
        self.relative_pos_c = []
        self.relative_pos_d = []
        for i in range(phase_num):
            self.relative_pos_a.append(nn.Parameter(torch.randn(
                1, self.patch_embeds[0].num_patches*16,
                self.patch_embeds[0].num_patches*16 // sr_ratios[0] // sr_ratios[0])))
            self.relative_pos_b.append(nn.Parameter(torch.randn(
                2, self.patch_embeds[0].num_patches*4,
                self.patch_embeds[0].num_patches*4 // sr_ratios[1] // sr_ratios[1])))
            self.relative_pos_c.append(nn.Parameter(torch.randn(
                4, self.patch_embeds[0].num_patches,
                self.patch_embeds[0].num_patches // sr_ratios[2] // sr_ratios[2])))
            # only use cross phase tokens in the last stage
            self.relative_pos_d.append(nn.Parameter(torch.randn(
                8, self.patch_embeds[0].num_patches // 4 + bottleneck_n,
                self.patch_embeds[0].num_patches // 4 // sr_ratios[3] // sr_ratios[3] + bottleneck_n)))
        self.relative_pos_a = nn.ParameterList(self.relative_pos_a)
        self.relative_pos_b = nn.ParameterList(self.relative_pos_b)
        self.relative_pos_c = nn.ParameterList(self.relative_pos_c)
        self.relative_pos_d = nn.ParameterList(self.relative_pos_d)

        # init transformer blocks of the first 2 stages
        self.depth_b = 3  # depth of stage a and b
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth+2*self.depth_b)]  # stochastic depth decay rule
        self.blocks_a = []
        self.blocks_b = []
        for l in range(self.depth_b):
            self.blocks_a.append([])
            self.blocks_b.append([])
            for i in range(phase_num):
                self.blocks_a[l].append(InceptionBlock(dim=64, drop_path=dpr[l], layer_scale_init_value=layer_scale_init_value))
                self.blocks_b[l].append(InceptionBlock(dim=128, drop_path=dpr[l], layer_scale_init_value=layer_scale_init_value))
            self.blocks_a[l] = nn.ModuleList(self.blocks_a[l])
            self.blocks_b[l] = nn.ModuleList(self.blocks_b[l])
        self.blocks_a = nn.ModuleList(self.blocks_a)
        self.blocks_b = nn.ModuleList(self.blocks_b)

        # init transformer blocks of the last 2 stages
        del self.blocks
        self.blocks = []
        for l in range(depth):
            self.blocks.append([])
            for i in range(phase_num):
                if l < fusion_layer:
                    self.blocks[l].append(InceptionBlock(dim=256, drop_path=dpr[l], layer_scale_init_value=layer_scale_init_value))
                else:
                    self.blocks[l].append(InceptionBlock(dim=self.embed_dim, drop_path=dpr[l], layer_scale_init_value=layer_scale_init_value, band_kernel_size=7))
            self.blocks[l] = nn.ModuleList(self.blocks[l])
        self.blocks = nn.ModuleList(self.blocks)

        self.fu1 = HFF_block(ch_1=64, ch_2=128, ch_int=128)
        self.fu2 = HFF_block(ch_1=128, ch_2=256, ch_int=256)
        self.fu3 = HFF_block(ch_1=256, ch_2=512, ch_int=512)

        self.max_frame1 = max_q(64,64)
        self.max_frame2 = max_q(128, 128)
        self.max_frame3 = max_q(256, 256)
        self.max_frame4 = max_q(512, 512)

        self.updown1 = Updownblock(64)
        self.updown2 = Updownblock(128)
        # self.updown3 = Updownblock(256)
        # self.updown4 = Updownblock(512)

        self.tau1 = TemporalAttentionModule(512, 3)
        self.tau2 = TemporalAttentionModule(1024, 3)
        self.tau3 = TemporalAttentionModule(2048, 3)
        self.tau4 = TemporalAttentionModule(4096, 3)

        self.CAT1 = AttentionTF(64, 64, original_scale=(56, 56))
        self.CAT2 = AttentionTF(128, 128, original_scale=(28, 28))
        self.CAT3 = AttentionTF(256, 256, original_scale=(14, 14))
        self.CAT4 = AttentionTF(512, 512, original_scale=(7, 7))

        self.Fusion = FeatureFusion()

        # 2 fc
        self.norm_head = nn.LayerNorm(embed_dim_y, eps=1e-6)
        fc_mid_dim = 64
        self.head0 = nn.Linear(embed_dim_y, fc_mid_dim)
        self.head1 = nn.Linear(fc_mid_dim, num_classes)
        self.dropout = nn.Dropout(p=0.5)
        self.init_weights_custom()

    def init_weights_custom(self):
        if self.bottleneck is not None:
            nn.init.normal_(self.bottleneck, std=.02)
        named_apply(init_weights_vit_timm, self)

    def forward_features(self, x):
        # patch embedding
        for i in range(self.phase_num):
            x[i] = self.patch_embeds[i](x[i])

        # stage a
        for l in range(self.depth_b):
            for i in range(self.phase_num):
                x[i] = self.blocks_a[l][i](x[i])
        # y_a = x[0]
        # for i in range(1, len(x)):
        #     y_a += x[i]
        # y_a = torch.cat(x, 1)
        y_a = torch.stack(x, 1)

        # downsample
        for i in range(self.phase_num):
            # x[i] = x[i].reshape(x[i].shape[0], self.img_size//4, self.img_size//4, -1).permute(0, 3, 1, 2).contiguous()
            x[i] = self.down_sample_a[i](x[i])

        # stage b
        for l in range(self.depth_b):
            for i in range(self.phase_num):
                x[i] = self.blocks_b[l][i](x[i])
        # y_b = x[0]
        # for i in range(1, len(x)):
        #     y_b += x[i]
        # y_b = torch.cat(x, 1)
        y_b = torch.stack(x, 1)
        # downsample
        for i in range(self.phase_num):
            # x[i] = x[i].reshape(x[i].shape[0], self.img_size//8, self.img_size//8, -1).permute(0, 3, 1, 2).contiguous()
            x[i] = self.down_sample_b[i](x[i])

        # cross phase tokens
        if self.bottleneck is not None:
            batch_bottleneck = self.bottleneck.expand(x[0].shape[0], -1, -1)
        else:
            batch_bottleneck = None

        for i in range(self.phase_num):
            x[i] = self.norm_pres[i](x[i])

        for l in range(self.depth):
            # stage c
            if l < self.fusion_layer:
                for i in range(self.phase_num):
                    x[i] = self.blocks[l][i](x[i])
                if l == self.fusion_layer - 1:
                    # y_c = x[0]
                    # for i in range(1, len(x)):
                    #     y_c += x[i]
                    # y_c = torch.cat(x, 1)
                    y_c = torch.stack(x, 1)
                    for i in range(self.phase_num):
                        # x[i] = x[i].reshape(x[i].shape[0], self.img_size // 16, self.img_size // 16, -1).permute(0, 3, 1, 2).contiguous()
                        x[i] = self.down_sample_c[i](x[i])
                        self.temporal_dims[i] = x[i].shape[2]*x[i].shape[3]
            # stage d
            else:
                bottle = []
                for i in range(self.phase_num):
                    t_mod = x[i].shape[1]
                    out_mod = self.blocks[l][i](x[i])
                    x[i] = out_mod[:, :t_mod, ...]
                    bottle.append(out_mod[:, t_mod:, ...])

        y_d = torch.stack(x, 1)

        x_a = extract_important_frames(y_a)
        x_b = extract_important_frames(y_b)
        x_c = extract_important_frames(y_c)
        x_d = extract_important_frames(y_d)

        x_a = self.updown1(x_a)
        x_b = self.updown2(x_b)

        y_a = self.tau1(y_a)
        y_b = self.tau2(y_b)
        y_c = self.tau3(y_c)
        y_d = self.tau4(y_d)
        out1 = self.CAT1(y_a, x_a)
        out2 = self.CAT2(y_b, x_b)
        out3 = self.CAT3(y_c, x_c)
        out4 = self.CAT4(y_d, x_d)

        y_f_1 = self.fu1(out1, out2)
        y_f_2 = self.fu2(y_f_1, out3)
        y_f_3 = self.fu3(y_f_2, out4)

        return y_f_3

    def forward_head(self, x, pre_logits: bool = False):
        x_out = []
        counter = 0
        # global average for each phase
        for i in range(self.phase_num):
            x_out.append(x[:, counter:counter+self.temporal_dims[i], ...].mean(dim=1))
            counter += self.temporal_dims[i]

        if pre_logits:
            return x_out
        for i in range(self.phase_num):
            # 2 fc
            x_out[i] = self.head0(x_out[i])
            x_out[i] = self.head1(x_out[i])
        x_pool = torch.zeros_like(x_out[0])
        for i in range(self.phase_num):
            x_pool += x_out[i]
        return x_pool / len(x_out)

    def forward_head_y(self, x):

        x = self.norm_head(x.mean([-2, -1]))
        # x = self.dropout(F.relu(self.head0(x)))
        # x = F.relu(self.head0(x))
        x = self.head0(x)
        x = self.head1(x)

        # return F.log_softmax(x, dim=1)
        return x

    def forward(self, x):
        xs = []
        for i in range(self.phase_num):
            xs.append(x[:, i, ...])
        x = xs
        x = self.forward_features(x)
        x = self.forward_head_y(x)
        return x


@register_model
def mbt_base_phase8_bottleneck8_vit(pretrained=False, pretrain_path=None, pretrained_cfg=None, **kwargs):
    def find_keys(d, s):
        keys = []
        for key in d:
            if s in key:
                keys.append(key)
        return keys

    def find_phase_position(s, dots=1):
        cnt = 0
        for ind in range(len(s)):
            if s[ind] == '.':
                if cnt == dots:
                    return ind+1
                else:
                    cnt += 1
        return len(s)

    phase_num = 8
    bottleneck_n = 8
    use_bottleneck = True
    model = MultiPhaseVisionTransformer(phase_num=phase_num, bottleneck_n=bottleneck_n,
                                        use_bottleneck=use_bottleneck, **kwargs)

    # load the pretrain model into the new model
    if pretrained:
        pre_model = timm.create_model("vit_small_patch16_224", pretrained=True, **kwargs)
        model_dict = model.state_dict()
        new_dict = {}
        pre_dict = pre_model.state_dict()
        pre_dict_cmt = torch.load(pretrain_path)["model"]  # may be changed for your pretrain model path
        para_dict = {}
        for k in pre_dict:
            if k in model_dict and model_dict[k].shape == pre_dict[k].shape:
                para_dict[k] = k
        for k in pre_dict_cmt:
            if "stem_conv" in k or "stem_norm" in k:
                for mk in find_keys(model_dict, k):
                    if model_dict[mk].shape == pre_dict_cmt[k].shape:
                        new_dict[mk] = pre_dict_cmt[k]
                        para_dict[mk] = k
                    else:
                        new_dict[mk] = torch.sum(pre_dict_cmt[k], dim=1).unsqueeze(1)
                        para_dict[mk] = k
            elif "patch_embed" in k:
                for mk in find_keys(model_dict, "down_sample")+find_keys(model_dict, "patch_embed"):
                    if mk.split('.')[2] == k.split('.')[1] and mk.split('.')[3] == k.split('.')[2]:
                        if model_dict[mk].shape == pre_dict_cmt[k].shape:
                            new_dict[mk] = pre_dict_cmt[k]
                            para_dict[mk] = k
            elif "block" in k:
                for i in range(phase_num):
                    if "blocks_c" in k:
                        nk = k.replace("blocks_c", "blocks")
                    elif "blocks_d" in k:
                        nk = k.replace("blocks_d", "blocks")
                        l = int(k.split('.')[1])
                        nl = l + fusion_layer
                        nk = nk.replace(str(l), str(nl), 1)
                    else:
                        nk = k
                    pos = find_phase_position(nk)
                    mk = nk[0:pos] + str(i) + "." + nk[pos:]
                    if len(find_keys(model_dict, mk)) > 0:
                        if model_dict[mk].shape == pre_dict_cmt[k].shape:
                            new_dict[mk] = pre_dict_cmt[k]
                            para_dict[mk] = k
            elif "relative_pos" in k:
                for mk in find_keys(model_dict, k):
                    if model_dict[mk].shape == pre_dict_cmt[k].shape:
                        new_dict[mk] = pre_dict_cmt[k]
                        para_dict[mk] = k
                    elif model_dict[mk].shape[1] == pre_dict_cmt[k].shape[1]+bottleneck_n:
                        # relative pos containing with cross-phase-token
                        new_dict[mk] = model_dict[mk].clone()
                        new_dict[mk][:, :-bottleneck_n, :-bottleneck_n] = pre_dict_cmt[k]
                        para_dict[mk] = k

        model_dict.update(new_dict)
        model.load_state_dict(model_dict)

    return model


def create_mbt(model_name, pretrained, pretrain_path, **kwargs):
    return timm.create_model(model_name, pretrained=pretrained, pretrain_path=pretrain_path, **kwargs)
