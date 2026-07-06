import warnings
import logging
from collections import OrderedDict
from copy import deepcopy

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint as cp
from torch.nn.init import trunc_normal_


# =====================================================
# utils
# =====================================================

def to_2tuple(x):
    if isinstance(x, tuple):
        return x
    if isinstance(x, list):
        return tuple(x)
    return (x, x)


ModuleList = nn.ModuleList


def constant_init(module, val, bias=0):
    if hasattr(module, "weight") and module.weight is not None:
        nn.init.constant_(module.weight, val)
    if hasattr(module, "bias") and module.bias is not None:
        nn.init.constant_(module.bias, bias)


def trunc_normal_init(module, mean=0., std=1., bias=0.):
    if hasattr(module, "weight") and module.weight is not None:
        trunc_normal_(module.weight, mean=mean, std=std)
    if hasattr(module, "bias") and module.bias is not None:
        nn.init.constant_(module.bias, bias)


def load_checkpoint(filename, map_location="cpu"):
    return torch.load(filename, map_location=map_location)


# =====================================================
# builders
# =====================================================

def build_norm_layer(cfg, num_features):
    if cfg is None:
        return "id", nn.Identity()

    t = cfg["type"]

    if t in ["LN", "LayerNorm"]:
        return "ln", nn.LayerNorm(num_features)

    elif t in ["BN", "BN2d", "BatchNorm2d"]:
        return "bn", nn.BatchNorm2d(num_features)

    raise KeyError(t)


class DropPath(nn.Module):
    def __init__(self, drop_prob=0.):
        super().__init__()
        self.drop_prob = float(drop_prob)

    def forward(self, x):
        if self.drop_prob == 0. or not self.training:
            return x

        keep_prob = 1 - self.drop_prob
        shape = (x.shape[0],) + (1,) * (x.ndim - 1)
        random_tensor = keep_prob + torch.rand(
            shape, dtype=x.dtype, device=x.device
        )
        random_tensor.floor_()
        return x.div(keep_prob) * random_tensor


def build_dropout(cfg):
    if cfg is None:
        return nn.Identity()

    t = cfg["type"]

    if t == "DropPath":
        return DropPath(cfg.get("drop_prob", 0.))

    if t == "Dropout":
        return nn.Dropout(cfg.get("p", 0.))

    return nn.Identity()


# =====================================================
# FFN
# =====================================================

class FFN(nn.Module):
    def __init__(self,
                 embed_dims,
                 feedforward_channels,
                 num_fcs=2,
                 ffn_drop=0.,
                 dropout_layer=None,
                 act_cfg=dict(type="GELU"),
                 add_identity=True,):
        super().__init__()

        self.add_identity = add_identity

        if act_cfg["type"] == "GELU":
            act = nn.GELU()
        else:
            act = nn.ReLU(inplace=True)

        self.layers = nn.Sequential(
            nn.Sequential(
                nn.Linear(embed_dims, feedforward_channels),
                act,
                nn.Dropout(ffn_drop),
            ),
            nn.Linear(feedforward_channels, embed_dims),
            nn.Dropout(ffn_drop)
        )

        self.drop = build_dropout(dropout_layer)

    def forward(self, x, identity=None):
        out = self.layers(x)
        out = self.drop(out)

        if self.add_identity:
            if identity is None:
                identity = x
            out = out + identity

        return out


# =====================================================
# PatchEmbed / PatchMerging
# =====================================================

class PatchEmbed(nn.Module):
    def __init__(self,
                 in_channels=3,
                 embed_dims=96,
                 kernel_size=4,
                 stride=4,
                 norm_cfg=None,
                 **kwargs):
        super().__init__()

        self.proj = nn.Conv2d(
            in_channels,
            embed_dims,
            kernel_size=kernel_size,
            stride=stride
        )

        self.norm = build_norm_layer(norm_cfg, embed_dims)[1] \
            if norm_cfg else None

    def forward(self, x):
        x = self.proj(x)
        H, W = x.shape[2], x.shape[3]

        x = x.flatten(2).transpose(1, 2)

        if self.norm is not None:
            x = self.norm(x)

        return x, (H, W)


class PatchMerging(nn.Module):
    def __init__(self,
                 in_channels,
                 out_channels,
                 stride=2,
                 norm_cfg=dict(type="LN"),
                 **kwargs):
        super().__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels

        self.norm = build_norm_layer(norm_cfg, 4 * in_channels)[1]
        self.reduction = nn.Linear(4 * in_channels, out_channels, bias=False)

    def forward(self, x, hw_shape):
        B, L, C = x.shape
        H, W = hw_shape

        # Match MMDetection/MMRotate PatchMerging exactly.  Their
        # implementation uses nn.Unfold on BCHW tensors, so the flattened
        # 2x2 patch order is channel-major:
        #   [c0_tl, c0_tr, c0_bl, c0_br, c1_tl, ...]
        # The previous manual concat used position-major ordering
        #   [tl_all_channels, bl_all_channels, tr_all_channels, br_all_channels]
        # which is incompatible with original Swin checkpoint weights.
        x = x.reshape(B, H, W, C).permute(0, 3, 1, 2).contiguous()

        pad_h = H % 2
        pad_w = W % 2
        if pad_h or pad_w:
            x = F.pad(x, (0, pad_w, 0, pad_h))
            H = H + pad_h
            W = W + pad_w

        x = F.unfold(x, kernel_size=2, stride=2).transpose(1, 2)
        H2, W2 = H // 2, W // 2
        x = self.norm(x)
        x = self.reduction(x)

        return x, (H2, W2)


# =====================================================
# Window Attention
# =====================================================

class WindowMSA(nn.Module):
    def __init__(self,
                 embed_dims,
                 num_heads,
                 window_size,
                 qkv_bias=True,
                 qk_scale=None,
                 attn_drop_rate=0.,
                 proj_drop_rate=0.,):
        super().__init__()

        self.embed_dims = embed_dims
        self.window_size = window_size
        self.num_heads = num_heads

        head_dim = embed_dims // num_heads
        self.scale = qk_scale or head_dim ** -0.5

        self.relative_position_bias_table = nn.Parameter(
            torch.zeros(
                (2 * window_size[0] - 1) *
                (2 * window_size[1] - 1),
                num_heads
            )
        )

        Wh, Ww = self.window_size

        coords_h = torch.arange(Wh)
        coords_w = torch.arange(Ww)
        coords = torch.stack(torch.meshgrid(coords_h, coords_w, indexing="ij"))
        coords_flatten = coords.flatten(1)

        relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :]
        relative_coords = relative_coords.permute(1, 2, 0).contiguous()

        relative_coords[:, :, 0] += Wh - 1
        relative_coords[:, :, 1] += Ww - 1
        relative_coords[:, :, 0] *= 2 * Ww - 1

        relative_position_index = relative_coords.sum(-1)

        self.register_buffer(
            "relative_position_index",
            relative_position_index
        )

        self.qkv = nn.Linear(embed_dims, embed_dims * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop_rate)

        self.proj = nn.Linear(embed_dims, embed_dims)
        self.proj_drop = nn.Dropout(proj_drop_rate)

        self.softmax = nn.Softmax(dim=-1)

        trunc_normal_(self.relative_position_bias_table, std=0.02)

    def forward(self, x, mask=None):
        B_, N, C = x.shape

        qkv = self.qkv(x).reshape(
            B_, N, 3, self.num_heads, C // self.num_heads
        ).permute(2, 0, 3, 1, 4)

        q, k, v = qkv[0], qkv[1], qkv[2]

        q = q * self.scale
        attn = q @ k.transpose(-2, -1)

        relative_position_bias = self.relative_position_bias_table[
            self.relative_position_index.view(-1)
        ].view(N, N, -1)

        relative_position_bias = relative_position_bias.permute(2, 0, 1)
        attn = attn + relative_position_bias.unsqueeze(0)

        if mask is not None:
            nW = mask.shape[0]
            attn = attn.view(
                B_ // nW, nW, self.num_heads, N, N
            ) + mask.unsqueeze(1).unsqueeze(0)

            attn = attn.view(-1, self.num_heads, N, N)

        attn = self.softmax(attn)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B_, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)

        return x


# =====================================================
# ShiftWindowMSA
# =====================================================

class ShiftWindowMSA(nn.Module):
    def __init__(self,
                 embed_dims,
                 num_heads,
                 window_size,
                 shift_size=0,
                 qkv_bias=True,
                 qk_scale=None,
                 attn_drop_rate=0.,
                 proj_drop_rate=0.,
                 dropout_layer=dict(type="DropPath", drop_prob=0.),):
        super().__init__()

        self.window_size = window_size
        self.shift_size = shift_size

        self.w_msa = WindowMSA(
            embed_dims=embed_dims,
            num_heads=num_heads,
            window_size=to_2tuple(window_size),
            qkv_bias=qkv_bias,
            qk_scale=qk_scale,
            attn_drop_rate=attn_drop_rate,
            proj_drop_rate=proj_drop_rate
        )

        self.drop = build_dropout(dropout_layer)

    def window_partition(self, x):
        B, H, W, C = x.shape
        ws = self.window_size

        x = x.reshape(B, H // ws, ws, W // ws, ws, C)
        windows = x.permute(0, 1, 3, 2, 4, 5).contiguous()
        windows = windows.reshape(-1, ws, ws, C)
        return windows

    def window_reverse(self, windows, H, W):
        ws = self.window_size
        B = int(windows.shape[0] / (H * W / ws / ws))

        x = windows.reshape(B, H // ws, W // ws, ws, ws, -1)
        x = x.permute(0, 1, 3, 2, 4, 5).contiguous()
        x = x.reshape(B, H, W, -1)

        return x

    def forward(self, query, hw_shape):
        B, L, C = query.shape
        H, W = hw_shape

        query = query.reshape(B, H, W, C)

        pad_r = (self.window_size - W % self.window_size) % self.window_size
        pad_b = (self.window_size - H % self.window_size) % self.window_size

        query = F.pad(query, (0, 0, 0, pad_r, 0, pad_b))
        H_pad, W_pad = query.shape[1], query.shape[2]

        if self.shift_size > 0:
            shifted = torch.roll(
                query,
                shifts=(-self.shift_size, -self.shift_size),
                dims=(1, 2)
            )
        else:
            shifted = query

        query_windows = self.window_partition(shifted)
        query_windows = query_windows.reshape(
            -1, self.window_size * self.window_size, C
        )

        attn_windows = self.w_msa(query_windows)

        attn_windows = attn_windows.reshape(
            -1, self.window_size, self.window_size, C
        )

        shifted_x = self.window_reverse(attn_windows, H_pad, W_pad)

        if self.shift_size > 0:
            x = torch.roll(
                shifted_x,
                shifts=(self.shift_size, self.shift_size),
                dims=(1, 2)
            )
        else:
            x = shifted_x

        if pad_r > 0 or pad_b > 0:
            x = x[:, :H, :W, :]

        x = x.reshape(B, H * W, C)
        x = self.drop(x)

        return x


# =====================================================
# SwinBlock
# =====================================================

class SwinBlock(nn.Module):
    def __init__(self,
                 embed_dims,
                 num_heads,
                 feedforward_channels,
                 window_size=7,
                 shift=False,
                 qkv_bias=True,
                 qk_scale=None,
                 drop_rate=0.,
                 attn_drop_rate=0.,
                 drop_path_rate=0.,
                 act_cfg=dict(type="GELU"),
                 norm_cfg=dict(type="LN"),
                 with_cp=False):
        super().__init__()

        self.with_cp = with_cp

        self.norm1 = build_norm_layer(norm_cfg, embed_dims)[1]

        self.attn = ShiftWindowMSA(
            embed_dims=embed_dims,
            num_heads=num_heads,
            window_size=window_size,
            shift_size=window_size // 2 if shift else 0,
            qkv_bias=qkv_bias,
            qk_scale=qk_scale,
            attn_drop_rate=attn_drop_rate,
            proj_drop_rate=drop_rate,
            dropout_layer=dict(
                type="DropPath",
                drop_prob=drop_path_rate
            )
        )

        self.norm2 = build_norm_layer(norm_cfg, embed_dims)[1]

        self.ffn = FFN(
            embed_dims=embed_dims,
            feedforward_channels=feedforward_channels,
            ffn_drop=drop_rate,
            dropout_layer=dict(
                type="DropPath",
                drop_prob=drop_path_rate
            ),
            act_cfg=act_cfg
        )

    def forward(self, x, hw_shape):

        def _inner(x):
            identity = x
            x = self.norm1(x)
            x = self.attn(x, hw_shape)
            x = x + identity

            identity = x
            x = self.norm2(x)
            x = self.ffn(x, identity)

            return x

        if self.with_cp and x.requires_grad:
            x = cp.checkpoint(_inner, x)
        else:
            x = _inner(x)

        return x


# =====================================================
# SwinTransformer
# =====================================================

class SwinTransformer(nn.Module):
    def __init__(self,
                 pretrain_img_size=224,
                 in_channels=3,
                 embed_dims=96,
                 patch_size=4,
                 window_size=7,
                 mlp_ratio=4,
                 depths=(2, 2, 6, 2),
                 num_heads=(3, 6, 12, 24),
                 strides=(4, 2, 2, 2),
                 out_indices=(0, 1, 2, 3),
                 qkv_bias=True,
                 qk_scale=None,
                 patch_norm=True,
                 drop_rate=0.,
                 attn_drop_rate=0.,
                 drop_path_rate=0.1,
                 use_abs_pos_embed=False,
                 norm_cfg=dict(type="LN"),
                 with_cp=False):
        super().__init__()

        self.out_indices = out_indices
        self.use_abs_pos_embed = use_abs_pos_embed

        self.patch_embed = PatchEmbed(
            in_channels=in_channels,
            embed_dims=embed_dims,
            kernel_size=patch_size,
            stride=strides[0],
            norm_cfg=norm_cfg if patch_norm else None
        )

        self.drop_after_pos = nn.Dropout(drop_rate)

        total_depth = sum(depths)
        dpr = torch.linspace(0, drop_path_rate, total_depth).tolist()

        self.stages = ModuleList()

        in_ch = embed_dims
        cur = 0

        for i in range(len(depths)):
            blocks = []

            for j in range(depths[i]):
                blocks.append(
                    SwinBlock(
                        embed_dims=in_ch,
                        num_heads=num_heads[i],
                        feedforward_channels=int(mlp_ratio * in_ch),
                        window_size=window_size,
                        shift=(j % 2 == 1),
                        qkv_bias=qkv_bias,
                        qk_scale=qk_scale,
                        drop_rate=drop_rate,
                        attn_drop_rate=attn_drop_rate,
                        drop_path_rate=dpr[cur + j],
                        with_cp=with_cp
                    )
                )

            cur += depths[i]

            downsample = None
            if i < len(depths) - 1:
                downsample = PatchMerging(
                    in_channels=in_ch,
                    out_channels=in_ch * 2
                )

            self.stages.append(
                nn.ModuleDict({
                    "blocks": nn.ModuleList(blocks),
                    "downsample": downsample
                })
            )

            if downsample is not None:
                in_ch *= 2

        self.num_features = [
            int(embed_dims * 2 ** i)
            for i in range(len(depths))
        ]

        for i in out_indices:
            self.add_module(
                f"norm{i}",
                nn.LayerNorm(self.num_features[i])
            )

        self.init_weights()

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                trunc_normal_init(m, std=0.02)

            elif isinstance(m, nn.LayerNorm):
                constant_init(m, 1.0)

    def load_pretrained_weights(self, checkpoint_path):
        checkpoint = load_checkpoint(checkpoint_path, map_location="cpu")
        state_dict = checkpoint["state_dict"] if "state_dict" in checkpoint else checkpoint

        new_state_dict = OrderedDict()

        for k, v in state_dict.items():
            if k.startswith("backbone."):
                new_k = k[len("backbone."):]
                new_state_dict[new_k] = v

        self.load_state_dict(new_state_dict, strict=False)

    def forward(self, x):
        x, hw_shape = self.patch_embed(x)
        x = self.drop_after_pos(x)

        outs = []

        for i, stage in enumerate(self.stages):
            for blk in stage["blocks"]:
                x = blk(x, hw_shape)

            out = x

            if i in self.out_indices:
                norm = getattr(self, f"norm{i}")
                out = norm(out)

                B = out.shape[0]
                H, W = hw_shape
                C = self.num_features[i]

                out = out.reshape(B, H, W, C).permute(0, 3, 1, 2).contiguous()
                outs.append(out)

            if stage["downsample"] is not None:
                x, hw_shape = stage["downsample"](x, hw_shape)

        return outs
