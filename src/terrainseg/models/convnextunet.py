from typing import Sequence, Optional, Dict, Any, Union, Tuple, Literal

import torch
import torch.nn as nn
import torch.nn.functional as F


# -------------------------
# Utility norms / helpers
# -------------------------

class LayerNorm2d(nn.Module):
    """LayerNorm over channel dimension for NCHW tensors."""
    def __init__(self, num_channels: int, eps: float = 1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(num_channels))
        self.bias = nn.Parameter(torch.zeros(num_channels))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        mean = x.mean(dim=1, keepdim=True)
        var = (x - mean).pow(2).mean(dim=1, keepdim=True)
        x = (x - mean) / torch.sqrt(var + self.eps)
        return x * self.weight[:, None, None] + self.bias[:, None, None]


def _valid_gn_groups(num_channels: int, requested_groups: int) -> int:
    """Choose a GroupNorm group count that divides num_channels (fallback safely)."""
    g = min(requested_groups, num_channels)
    while g > 1 and (num_channels % g) != 0:
        g -= 1
    return g


# -------------------------
# ConvNeXt blocks (encoder + bottleneck)
# -------------------------

class ConvNeXtBlock(nn.Module):
    """
    ConvNeXt-style block:
      DWConv 7x7 -> LN -> 1x1 conv expand -> GELU -> 1x1 conv -> residual
    """
    def __init__(self, dim: int, mlp_ratio: int = 4, drop_path: float = 0.0):
        super().__init__()
        self.dwconv = nn.Conv2d(dim, dim, kernel_size=7, padding=3, groups=dim)
        self.norm = LayerNorm2d(dim)
        hidden = int(dim * mlp_ratio)
        self.pwconv1 = nn.Conv2d(dim, hidden, kernel_size=1)
        self.act = nn.GELU()
        self.pwconv2 = nn.Conv2d(hidden, dim, kernel_size=1)
        self.drop_path = float(drop_path)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = x
        x = self.dwconv(x)
        x = self.norm(x)
        x = self.pwconv1(x)
        x = self.act(x)
        x = self.pwconv2(x)

        if self.training and self.drop_path > 0:
            keep_prob = 1.0 - self.drop_path
            shape = (x.shape[0],) + (1,) * (x.ndim - 1)
            mask = x.new_empty(shape).bernoulli_(keep_prob)
            x = x / keep_prob * mask

        return identity + x


# -------------------------
# Decoder blocks (GN + residual)
# -------------------------

class DoubleConvGN(nn.Module):
    """UNet double conv using GroupNorm: (conv -> GN -> ReLU) x2."""
    def __init__(self, in_ch: int, out_ch: int, gn_groups: int = 8):
        super().__init__()
        g = _valid_gn_groups(out_ch, gn_groups)
        self.net = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(num_groups=g, num_channels=out_ch),
            nn.GELU(),
            nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(num_groups=g, num_channels=out_ch),
            nn.GELU(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class ResidualDoubleConvGN(nn.Module):
    """
    DoubleConvGN + residual:
      out = F(x) + proj(x)
    """
    def __init__(self, in_ch: int, out_ch: int, gn_groups: int = 8):
        super().__init__()
        self.f = DoubleConvGN(in_ch, out_ch, gn_groups=gn_groups)

        if in_ch == out_ch:
            self.proj = nn.Identity()
        else:
            g = _valid_gn_groups(out_ch, gn_groups)
            self.proj = nn.Sequential(
                nn.Conv2d(in_ch, out_ch, kernel_size=1, bias=False),
                nn.GroupNorm(num_groups=g, num_channels=out_ch),
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.f(x) + self.proj(x)


# -------------------------
# Skip projection (GELU along skip connections)
# -------------------------

class SkipProjection(nn.Module):
    """
    Skip projection before attention/merge.
    Uses GELU by default for consistency with ConvNeXt-style features.
    """
    def __init__(self, in_ch: int, proj_ch: int, gn_groups: int = 8, act: str = "gelu"):
        super().__init__()
        g = _valid_gn_groups(proj_ch, gn_groups)

        if act == "gelu":
            activation = nn.GELU()
        elif act == "relu":
            activation = nn.ReLU(inplace=True)
        elif act == "silu":
            activation = nn.SiLU(inplace=True)
        else:
            raise ValueError("act must be 'gelu', 'relu', or 'silu'")

        self.net = nn.Sequential(
            nn.Conv2d(in_ch, proj_ch, kernel_size=1, bias=False),
            nn.GroupNorm(num_groups=g, num_channels=proj_ch),
            activation,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


# -------------------------
# Attention Gate (GELU inside gate for smoother masks)
# -------------------------

class AttentionGate(nn.Module):
    """
    Additive attention gate for skip connections.

    x: encoder skip (N, Cx, H, W)
    g: decoder gating (N, Cg, H, W)
    returns gated skip: x * alpha
    """
    def __init__(self, x_channels: int, g_channels: int, inter_channels: int = None):
        super().__init__()
        if inter_channels is None:
            inter_channels = max(1, x_channels // 2)

        self.theta_x = nn.Conv2d(x_channels, inter_channels, kernel_size=1, bias=False)
        self.phi_g = nn.Conv2d(g_channels, inter_channels, kernel_size=1, bias=False)

        self.psi = nn.Sequential(
            nn.ReLU(),
            nn.Conv2d(inter_channels, 1, kernel_size=1, bias=True),
            nn.Sigmoid()
        )

    def forward(self, x: torch.Tensor, g: torch.Tensor) -> torch.Tensor:
        att = self.theta_x(x) + self.phi_g(g)
        alpha = self.psi(att)  # (N,1,H,W)
        return x * alpha


# -------------------------
# Encoder / Decoder stages
# -------------------------

class DownStage(nn.Module):
    """
    Encoder stage:
      - ConvNeXt blocks at fixed width
      - stride-2 downsample conv to next width
    """
    def __init__(self, in_dim: int, out_dim: int, depth: int = 2, mlp_ratio: int = 4):
        super().__init__()
        self.blocks = nn.Sequential(*[ConvNeXtBlock(in_dim, mlp_ratio=mlp_ratio) for _ in range(depth)])
        self.down = nn.Conv2d(in_dim, out_dim, kernel_size=2, stride=2)

    def forward(self, x: torch.Tensor):
        x = self.blocks(x)
        skip = x
        x = self.down(x)
        return skip, x


class Upsampler(nn.Module):
    """
    Upsampling module:
      - 'interp': non-trainable interpolate (bilinear/nearest) + optional 1x1 projection
      - 'transpose': trainable ConvTranspose2d
    Always aims for 2x upsampling.
    """
    def __init__(self, in_ch: int, out_ch: int, mode: str = "interp", interp_mode: str = "bilinear"):
        super().__init__()
        assert mode in ("interp", "transpose"), "mode must be 'interp' or 'transpose'"
        assert interp_mode in ("bilinear", "nearest"), "interp_mode must be 'bilinear' or 'nearest'"

        self.mode = mode
        self.interp_mode = interp_mode

        if self.mode == "transpose":
            self.up = nn.ConvTranspose2d(in_ch, out_ch, kernel_size=2, stride=2)
            self.proj = None
        else:
            self.up = None
            self.proj = nn.Identity() if in_ch == out_ch else nn.Conv2d(in_ch, out_ch, kernel_size=1, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.mode == "transpose":
            return self.up(x)
        x = F.interpolate(
            x, scale_factor=2, mode=self.interp_mode,
            align_corners=False if self.interp_mode == "bilinear" else None
        )
        return self.proj(x)


class UpStageAttnRes(nn.Module):
    """
    Decoder stage:
      - upsample by 2 (interp or transpose)
      - match shape to skip (pad if needed)
      - (optional) project skip via 1x1 + GN + GELU
      - attention-gate the (projected) skip using gating signal (upsampled x)
      - concat gated skip + x
      - ResidualDoubleConvGN to produce out_dim
    """
    def __init__(
        self,
        in_dim: int,
        skip_dim: int,
        out_dim: int,
        gn_groups: int = 8,
        upsample_mode: str = "interp",      # "interp" or "transpose"
        interp_mode: str = "bilinear",      # used when upsample_mode="interp"
        attn_inter_channels: int = None,
        skip_proj_ch: int = None,           # if None -> no projection; else project skip_dim -> skip_proj_ch
        skip_proj_act: str = "gelu",        # GELU along skip connections
    ):
        super().__init__()

        # Up produces out_dim channels (keeps decoder widths consistent)
        self.up = Upsampler(in_dim, out_dim, mode=upsample_mode, interp_mode=interp_mode)

        # Skip projection (optional)
        if skip_proj_ch is None:
            self.skip_proj = nn.Identity()
            gated_skip_ch = skip_dim
        else:
            self.skip_proj = SkipProjection(skip_dim, skip_proj_ch, gn_groups=gn_groups, act=skip_proj_act)
            gated_skip_ch = skip_proj_ch

        self.attn = AttentionGate(gated_skip_ch, out_dim, inter_channels=attn_inter_channels)
        self.conv = ResidualDoubleConvGN(out_dim + gated_skip_ch, out_dim, gn_groups=gn_groups)

    def forward(self, x: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        x = self.up(x)  # (N,out_dim, H*2, W*2)

        # Align x to skip spatial dims (handles odd sizes safely)
        diff_y = skip.size(2) - x.size(2)
        diff_x = skip.size(3) - x.size(3)
        if diff_y != 0 or diff_x != 0:
            x = F.pad(x, [diff_x // 2, diff_x - diff_x // 2, diff_y // 2, diff_y - diff_y // 2])

        skip = self.skip_proj(skip)
        skip_gated = self.attn(skip, x)

        x = torch.cat([skip_gated, x], dim=1)
        x = self.conv(x)
        return x


# -------------------------
# Full model
# -------------------------

class defineCNXTUNet(nn.Module):
    """
    ConvNeXt-style UNet with attention-gated skip connections for semantic segmentation.

    This model uses a ConvNeXt-inspired encoder (stacked ConvNeXt blocks within downsampling
    stages), a bottleneck block stack, and a U-Net decoder with attention-gated skip
    connections and residual decoder blocks. Skip features can optionally be projected
    to match decoder widths (or to a user-specified channel count) prior to fusion.

    Parameters
    ----------
    in_channels : int, default=6
        Number of input channels in the input tensor (e.g., multispectral bands or
        terrain derivatives).
    num_classes : int, default=2
        Number of output segmentation classes (i.e., number of logits per pixel).
    features : tuple[int, int, int, int, int], default=(64, 128, 256, 512, 1024)
        Channel widths for each resolution level: ``(c1, c2, c3, c4, c5)``, where ``c1``
        is the highest-resolution stage width (stem/decoder output) and ``c5`` is the
        deepest stage width (encoder output / bottleneck input).
    enc_depths : tuple[int, int, int, int], default=(2, 2, 2, 2)
        Number of ConvNeXt blocks in each encoder downsampling stage: ``(d1, d2, d3, d4)``.
    bottleneck_depth : int, default=2
        Number of ConvNeXt blocks in the bottleneck (deepest) stage.
    mlp_ratio : int, default=4
        Expansion ratio for the ConvNeXt block MLP (channel mixing) sub-layer.
    gn_groups : int, default=8
        Target number of groups for GroupNorm. The effective group count may be adjusted
        per stage to ensure it divides the channel count (see `_valid_gn_groups`).
    upsample_mode : {"interp", "transpose"}, default="interp"
        Upsampling strategy in decoder stages:
        - ``"interp"`` uses non-trainable interpolation followed by convolution.
        - ``"transpose"`` uses a learnable transposed convolution.
    interp_mode : {"bilinear", "nearest"}, default="bilinear"
        Interpolation mode used when ``upsample_mode="interp"``.
    attn_inter_channels : str or None, default=None
        Intermediate channel setting for attention gates used on skip connections.
        Interpretation depends on `UpStageAttnRes` implementation (e.g., may accept
        ``"auto"`` or explicit sizing). If None, `UpStageAttnRes` chooses its default.
    skip_proj : {"none", "match"} or int or None, default="match"
        Skip projection strategy before fusing encoder skip features into the decoder:
        - ``"none"`` or None: no projection; skip features are used as-is.
        - ``"match"``: project skip features to match the decoder stage width.
        - ``int``: project skip features to this fixed number of channels.
    skip_proj_act : str, default="gelu"
        Activation used in the skip projection path (if enabled). Passed through to
        `UpStageAttnRes` (e.g., ``"gelu"``, ``"relu"``, etc., depending on implementation).

    Attributes
    ----------
    stem : nn.Module
        Initial convolutional stem (Conv2d + GroupNorm + GELU) that maps inputs to `c1`.
    down1, down2, down3, down4 : nn.Module
        Encoder downsampling stages producing multi-scale features.
    bottleneck : nn.Module
        Sequence of ConvNeXt blocks operating at the deepest resolution (`c5` channels).
    up4, up3, up2, up1 : nn.Module
        Decoder upsampling stages with attention-gated skip connections and residual blocks.
    head : nn.Module
        Final 1x1 convolution mapping `c1` channels to `num_classes` logits.

    Notes
    -----
    - Output logits have shape ``(N, num_classes, H, W)`` and are typically passed to a loss
      such as CrossEntropyLoss (multi-class) or BCEWithLogitsLoss (binary/multi-label),
      depending on your label encoding.
    - GroupNorm is used in the stem/decoder for stability across batch sizes.

    See Also
    --------
    DownStage : Encoder stage module used for downsampling and feature extraction.
    ConvNeXtBlock : Core ConvNeXt-style block used in encoder and bottleneck.
    UpStageAttnRes : Decoder stage with attention-gated skip fusion and residual blocks.
    """
    def __init__(
        self,
        in_channels: int = 6,
        num_classes: int = 2,
        features: Tuple[int, int, int, int, int]  = (64, 128, 256, 512, 1024),
        enc_depths: Tuple[int, int, int, int] = (2, 2, 2, 2),
        bottleneck_depth: int = 2,
        mlp_ratio: int = 4,
        gn_groups: int = 8,
        upsample_mode: Literal["interp", "transpose"] = "interp", 
        interp_mode: Literal["bilinear", "nearest"] = "bilinear",  
        attn_inter_channels: str = None,
        skip_proj: str = "match",            # "none" | "match" | int
        skip_proj_act: str = "gelu",         # GELU along skip connections
    ) -> nn.Module:
        """
        Initialize a ConvNeXt U-Net segmentation model.

        Parameters
        ----------
        in_channels : int, default=6
            Number of channels in the input image/tensor.
        num_classes : int, default=2
            Number of segmentation classes (output channels of the final head).
        features : tuple[int, int, int, int, int], default=(64, 128, 256, 512, 1024)
            Channel widths per resolution level: ``(c1, c2, c3, c4, c5)``.
        enc_depths : tuple[int, int, int, int], default=(2, 2, 2, 2)
            Encoder depths per downsampling stage.
        bottleneck_depth : int, default=2
            Number of ConvNeXt blocks in the bottleneck.
        mlp_ratio : int, default=4
            MLP expansion ratio for ConvNeXt blocks.
        gn_groups : int, default=8
            Target number of groups used for GroupNorm layers.
        upsample_mode : {"interp", "transpose"}, default="interp"
            Decoder upsampling strategy.
        interp_mode : {"bilinear", "nearest"}, default="bilinear"
            Interpolation mode used when ``upsample_mode="interp"``.
        attn_inter_channels : str or None, default=None
            Intermediate channel sizing configuration for attention gates.
        skip_proj : {"none", "match"} or int or None, default="match"
            Skip projection strategy prior to skip fusion.
        skip_proj_act : str, default="gelu"
            Activation used in the skip projection path, if enabled.

        Raises
        ------
        AssertionError
            If `features` is not length 5, `enc_depths` is not length 4, or if an
            invalid upsampling/interpolation option is provided.
        ValueError
            If `skip_proj` is not one of ``{"none", "match"}``, None, or an integer.

        Notes
        -----
        The model uses `GroupNorm` in the stem (and typically in decoder blocks) to
        support training with small batch sizes.
        """
        super().__init__()
        assert len(features) == 5, "features must be (c1,c2,c3,c4,c5)"
        assert len(enc_depths) == 4, "enc_depths must be length 4"
        assert upsample_mode in ("interp", "transpose")
        assert interp_mode in ("bilinear", "nearest")

        c1, c2, c3, c4, c5 = features

        # Stem (unchanged; GN for stability)
        self.stem = nn.Sequential(
            nn.Conv2d(in_channels, c1, kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(num_groups=_valid_gn_groups(c1, gn_groups), num_channels=c1),
            nn.GELU(),
        )

        # Encoder
        self.down1 = DownStage(c1, c2, depth=enc_depths[0], mlp_ratio=mlp_ratio)
        self.down2 = DownStage(c2, c3, depth=enc_depths[1], mlp_ratio=mlp_ratio)
        self.down3 = DownStage(c3, c4, depth=enc_depths[2], mlp_ratio=mlp_ratio)
        self.down4 = DownStage(c4, c5, depth=enc_depths[3], mlp_ratio=mlp_ratio)

        # Bottleneck
        self.bottleneck = nn.Sequential(*[
            ConvNeXtBlock(c5, mlp_ratio=mlp_ratio) for _ in range(bottleneck_depth)
        ])

        # Decide skip projection channels per stage
        def _skip_proj_ch(skip_dim, out_dim):
            if skip_proj == "none" or skip_proj is None:
                return None
            if skip_proj == "match":
                return out_dim  # project skip to match decoder stage width
            if isinstance(skip_proj, int):
                return int(skip_proj)
            raise ValueError("skip_proj must be 'none', 'match', or an int")

        # Decoder (residual blocks everywhere)
        self.up4 = UpStageAttnRes(
            in_dim=c5, skip_dim=c4, out_dim=c4, gn_groups=gn_groups,
            upsample_mode=upsample_mode, interp_mode=interp_mode,
            attn_inter_channels=attn_inter_channels,
            skip_proj_ch=_skip_proj_ch(c4, c4), skip_proj_act=skip_proj_act
        )
        self.up3 = UpStageAttnRes(
            in_dim=c4, skip_dim=c3, out_dim=c3, gn_groups=gn_groups,
            upsample_mode=upsample_mode, interp_mode=interp_mode,
            attn_inter_channels=attn_inter_channels,
            skip_proj_ch=_skip_proj_ch(c3, c3), skip_proj_act=skip_proj_act
        )
        self.up2 = UpStageAttnRes(
            in_dim=c3, skip_dim=c2, out_dim=c2, gn_groups=gn_groups,
            upsample_mode=upsample_mode, interp_mode=interp_mode,
            attn_inter_channels=attn_inter_channels,
            skip_proj_ch=_skip_proj_ch(c2, c2), skip_proj_act=skip_proj_act
        )
        self.up1 = UpStageAttnRes(
            in_dim=c2, skip_dim=c1, out_dim=c1, gn_groups=gn_groups,
            upsample_mode=upsample_mode, interp_mode=interp_mode,
            attn_inter_channels=attn_inter_channels,
            skip_proj_ch=_skip_proj_ch(c1, c1), skip_proj_act=skip_proj_act
        )

        # Heads
        self.head = nn.Conv2d(c1, num_classes, kernel_size=1)


    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor of shape (B, C, H, W).

        Returns
        -------
        torch.Tensor
            Segmentation logits of shape (B, n_classes, H, W).
        """

        x0 = self.stem(x)         # (N,c1,H,W)

        s1, x1 = self.down1(x0)   # s1: (N,c1,H,W),     x1: (N,c2,H/2,W/2)
        s2, x2 = self.down2(x1)   # s2: (N,c2,H/2,W/2), x2: (N,c3,H/4,W/4)
        s3, x3 = self.down3(x2)   # s3: (N,c3,H/4,W/4), x3: (N,c4,H/8,W/8)
        s4, x4 = self.down4(x3)   # s4: (N,c4,H/8,W/8), x4: (N,c5,H/16,W/16)

        x5 = self.bottleneck(x4)  # (N,c5,H/16,W/16)

        y4 = self.up4(x5, s4)     # (N,c4,H/8,W/8)
        y3 = self.up3(y4, s3)     # (N,c3,H/4,W/4)
        y2 = self.up2(y3, s2)     # (N,c2,H/2,W/2)
        y1 = self.up1(y2, s1)     # (N,c1,H,W)

        out = self.head(y1)       # (N,num_classes,H,W)

        return out