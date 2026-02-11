# PyTorch translation of your R torch modules (UNet + building blocks)
# Notes:
# - R torch uses dim=2 for channel concat; PyTorch uses dim=1 (N,C,H,W).
# - Your R code has a couple quirks/bugs (e.g., asppComp forward missing return,
#   asppBlk concat dim=2). This translation uses the correct PyTorch behavior.

from typing import Sequence, Optional, Dict, Any, Union, Tuple, Literal

import torch
import torch.nn as nn
import torch.nn.functional as F


# -------------------------
# small helpers
# -------------------------
def _make_act(act_func: str = "relu", negative_slope: float = 0.01) -> nn.Module:
    act_func = (act_func or "relu").lower()
    if act_func == "lrelu":
        return nn.LeakyReLU(negative_slope=negative_slope, inplace=True)
    if act_func == "swish":
        return nn.SiLU(inplace=True)
    return nn.ReLU(inplace=True)


# -------------------------
# Squeeze-and-Excitation
# -------------------------
class SEModule(nn.Module):
    def __init__(self, in_chn: int, ratio: int = 8):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        hidden = max(1, in_chn // ratio)
        self.se = nn.Sequential(
            nn.Linear(in_chn, hidden, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(hidden, in_chn, bias=True),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, c, _, _ = x.shape
        s = self.avg_pool(x).view(b, c)
        s = self.se(s).view(b, c, 1, 1)
        return x * s


# -------------------------
# Conv blocks
# -------------------------
class SimpleConvBlk(nn.Module):
    def __init__(self, in_chn: int, out_chn: int, act_func: str = "relu", negative_slope: float = 0.01):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_chn, out_chn, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(out_chn),
            _make_act(act_func, negative_slope),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(x)


class FeatReduce(nn.Module):
    # 1x1 conv optionally followed by BN+Act
    def __init__(
        self,
        in_chn: int,
        out_chn: int,
        act_func: str = "relu",
        do_bn_act: bool = True,
        negative_slope: float = 0.01,
    ):
        super().__init__()
        self.do_bn_act = bool(do_bn_act)
        self.conv = nn.Conv2d(in_chn, out_chn, kernel_size=1, stride=1, padding=0)
        if self.do_bn_act:
            self.bn_act = nn.Sequential(
                nn.BatchNorm2d(out_chn),
                _make_act(act_func, negative_slope),
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.conv(x)
        return self.bn_act(y) if self.do_bn_act else y


class UpConvBlk(nn.Module):
    def __init__(self, in_chn: int, out_chn: int, act_func: str = "relu", negative_slope: float = 0.01):
        super().__init__()
        self.up = nn.Sequential(
            nn.ConvTranspose2d(in_chn, out_chn, kernel_size=2, stride=2),
            nn.BatchNorm2d(out_chn),
            _make_act(act_func, negative_slope),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.up(x)


class DoubleConvBlk(nn.Module):
    def __init__(self, in_chn: int, out_chn: int, act_func: str = "relu", negative_slope: float = 0.01):
        super().__init__()
        act1 = _make_act(act_func, negative_slope)
        act2 = _make_act(act_func, negative_slope)
        self.block = nn.Sequential(
            nn.Conv2d(in_chn, out_chn, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(out_chn),
            act1,
            nn.Conv2d(out_chn, out_chn, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(out_chn),
            act2,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class DoubleConvBlkR(nn.Module):
    # residual: (conv-conv-bn) + 1x1 skip, then final activation
    def __init__(self, in_chn: int, out_chn: int, act_func: str = "relu", negative_slope: float = 0.01):
        super().__init__()
        self.res = nn.Sequential(
            nn.Conv2d(in_chn, out_chn, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(out_chn),
            _make_act(act_func, negative_slope),
            nn.Conv2d(out_chn, out_chn, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(out_chn),
        )
        self.skip = FeatReduce(in_chn, out_chn, act_func=act_func, do_bn_act=False, negative_slope=negative_slope)
        self.final_act = _make_act(act_func, negative_slope)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.res(x) + self.skip(x)
        return self.final_act(out)


# -------------------------
# Upsampling utilities
# -------------------------
class UpSamp(nn.Module):
    def __init__(self, scale_factor: float, mode: str = "bilinear", align_corners: bool = False):
        super().__init__()
        self.scale_factor = scale_factor
        self.mode = mode
        self.align_corners = align_corners

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.interpolate(
            x,
            scale_factor=self.scale_factor,
            mode=self.mode,
            align_corners=self.align_corners if self.mode in ("linear", "bilinear", "bicubic", "trilinear") else None,
        )


class InterpUp(nn.Module):
    def __init__(self, s_factor: float = 2):
        super().__init__()
        self.s_factor = s_factor

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.interpolate(x, scale_factor=self.s_factor, mode="bilinear", align_corners=True)


# -------------------------
# Attention gate
# -------------------------
class AttnBlk(nn.Module):
    # based on your R implementation (gating + downsample skip + psi + upsample)
    def __init__(self, x_chn: int, g_chn: int):
        super().__init__()
        self.W_gate = nn.Sequential(
            nn.Conv2d(g_chn, x_chn, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(x_chn),
        )
        self.W_x = nn.Sequential(
            nn.Conv2d(x_chn, x_chn, kernel_size=1, stride=2, padding=0, bias=True),
            nn.BatchNorm2d(x_chn),
        )
        self.psi = nn.Sequential(
            nn.Conv2d(x_chn, 1, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(1),
            nn.Sigmoid(),
            UpSamp(scale_factor=2, mode="bilinear", align_corners=False),
        )

    def forward(self, sc_in: torch.Tensor, gate_in: torch.Tensor) -> torch.Tensor:
        g1 = self.W_gate(gate_in)
        x1 = self.W_x(sc_in)
        psi = F.relu(g1 + x1, inplace=False)
        psi = self.psi(psi)
        return sc_in * psi


# -------------------------
# Classifier head
# -------------------------
class ClassifierBlk(nn.Module):
    def __init__(self, in_chn: int, n_cls: int):
        super().__init__()
        self.classifier = nn.Conv2d(in_chn, n_cls, kernel_size=1, stride=1, padding=0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.classifier(x)


# -------------------------
# Bottleneck / ASPP
# -------------------------
class Bottleneck(nn.Module):
    def __init__(self, in_chn: int, out_chn: int = 256, act_func: str = "relu", negative_slope: float = 0.01):
        super().__init__()
        self.btnk = DoubleConvBlk(in_chn, out_chn, act_func=act_func, negative_slope=negative_slope)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.btnk(x)


class BottleneckR(nn.Module):
    def __init__(self, in_chn: int, out_chn: int = 256, act_func: str = "relu", negative_slope: float = 0.01):
        super().__init__()
        self.btnk = DoubleConvBlkR(in_chn, out_chn, act_func=act_func, negative_slope=negative_slope)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.btnk(x)


class AsppComp(nn.Module):
    def __init__(
        self,
        in_chn: int,
        out_chn: int,
        kernel_size,
        stride,
        padding,
        dilation,
        act_func: str = "relu",
        negative_slope: float = 0.01,
    ):
        super().__init__()
        self.aspp = nn.Sequential(
            nn.Conv2d(
                in_channels=in_chn,
                out_channels=out_chn,
                kernel_size=kernel_size,
                stride=stride,
                padding=padding,
                dilation=dilation,
            ),
            nn.BatchNorm2d(out_chn),
            _make_act(act_func, negative_slope),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.aspp(x)


class GlobalAvgPool2d(nn.Module):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # output size should match H,W (assumes square in your R code; here do general H,W)
        n, c, h, w = x.shape
        out = F.adaptive_avg_pool2d(x, output_size=(1, 1))
        out = F.interpolate(out, size=(h, w), mode="bilinear", align_corners=False)
        return out


class AsppBlk(nn.Module):
    def __init__(
        self,
        in_chn: int,
        out_chn: int,
        dil_chn=(256, 256, 256, 256),
        dil_rates=(6, 12, 18),
        act_func: str = "relu",
        negative_slope: float = 0.01,
    ):
        super().__init__()
        d1, d2, d3, d4 = list(dil_chn)

        self.a1 = FeatReduce(in_chn, d1, act_func=act_func, negative_slope=negative_slope)
        self.a2 = AsppComp(in_chn, d2, kernel_size=3, stride=1, padding=dil_rates[0], dilation=dil_rates[0],
                           act_func=act_func, negative_slope=negative_slope)
        self.a3 = AsppComp(in_chn, d3, kernel_size=3, stride=1, padding=dil_rates[1], dilation=dil_rates[1],
                           act_func=act_func, negative_slope=negative_slope)
        self.a4 = AsppComp(in_chn, d4, kernel_size=3, stride=1, padding=dil_rates[2], dilation=dil_rates[2],
                           act_func=act_func, negative_slope=negative_slope)
        self.a5 = GlobalAvgPool2d()

        self.conv1_1 = FeatReduce(d1 + d2 + d3 + d4 + in_chn, out_chn, act_func=act_func, do_bn_act=True,
                                  negative_slope=negative_slope)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x1 = self.a1(x)
        x2 = self.a2(x)
        x3 = self.a3(x)
        x4 = self.a4(x)
        x5 = self.a5(x)
        xx = torch.cat([x1, x2, x3, x4, x5], dim=1)  # concat channels
        return self.conv1_1(xx)


class AsppBlkR(nn.Module):
    def __init__(
        self,
        in_chn: int,
        out_chn: int,
        dil_chn=(256, 256, 256, 256),
        dil_rates=(6, 12, 18),
        act_func: str = "relu",
        negative_slope: float = 0.01,
    ):
        super().__init__()
        d1, d2, d3, d4 = list(dil_chn)

        self.a1 = FeatReduce(in_chn, d1, act_func=act_func, negative_slope=negative_slope)
        self.a2 = AsppComp(in_chn, d2, kernel_size=3, stride=1, padding=dil_rates[0], dilation=dil_rates[0],
                           act_func=act_func, negative_slope=negative_slope)
        self.a3 = AsppComp(in_chn, d3, kernel_size=3, stride=1, padding=dil_rates[1], dilation=dil_rates[1],
                           act_func=act_func, negative_slope=negative_slope)
        self.a4 = AsppComp(in_chn, d4, kernel_size=3, stride=1, padding=dil_rates[2], dilation=dil_rates[2],
                           act_func=act_func, negative_slope=negative_slope)
        self.a5 = GlobalAvgPool2d()

        self.conv1_1 = FeatReduce(d1 + d2 + d3 + d4 + in_chn, out_chn, act_func=act_func, do_bn_act=True,
                                  negative_slope=negative_slope)
        self.skip = FeatReduce(in_chn, out_chn, act_func=act_func, do_bn_act=False, negative_slope=negative_slope)
        self.final_act = _make_act(act_func, negative_slope)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        xx = torch.cat([self.a1(x), self.a2(x), self.a3(x), self.a4(x), self.a5(x)], dim=1)
        xx = self.conv1_1(xx)
        xx = xx + self.skip(x)
        return self.final_act(xx)


class UpSampConv(nn.Module):
    def __init__(
        self,
        in_chn: int,
        out_chn: int,
        scale_factor: float,
        mode: str = "bilinear",
        align_corners: bool = False,
        act_func: str = "relu",
        negative_slope: float = 0.01,
    ):
        super().__init__()
        self.scale_factor = scale_factor
        self.mode = mode
        self.align_corners = align_corners
        self.conv1x1 = nn.Sequential(
            nn.Conv2d(in_chn, out_chn, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(out_chn),
            _make_act(act_func, negative_slope),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.interpolate(
            x,
            scale_factor=self.scale_factor,
            mode=self.mode,
            align_corners=self.align_corners if self.mode in ("linear", "bilinear", "bicubic", "trilinear") else None,
        )
        return self.conv1x1(x)

# -------------------------
# UNet
# -------------------------
class defineCUNet(nn.Module):
    """
    Customizable UNet for terrain semantic segmentation.

    Parameters
    ----------
    in_chn : int, default=1
        Number of input channels.
    n_cls : int, default=2
        Number of predicted classes.
    act_func :  Literal["relu", "lrelu", "swish"], default="relu"
        Activation function to use throughout architecture. "relu" = Rectified Linear Unit (ReLU); "lrelu" = Leaky ReLU; "swish" = Swish.
    use_attn : bool, default=False
        Whether or not to incorporate attention gates along the skip connections. 
    use_se : bool, default=False
        Whether or not to incorporate squeeze and excitation modules into the encoder.
    use_res : bool, default=False
        Whether or not to include residual connections with all double-convolution blocks in the encoder and decoder. 
    use_aspp : bool, default=False
        Whether or not to use an atrous spatial pyramid pooling (ASPP) module as the bottleneck block. 
    en_chn : Tuple[int, int, int, int], default=(16,32,64,128)
        Number of output feature maps from each encoder block. Must be tuple of 4 integers.
    dc_chn : Tuple[int, int, int, int], default=(128,64,32,16)
        Number of output feature maps from each decoder block. Must be tuple of 4 integers. 
    btn_chn : int, default=256
        Number of output feature maps from the bottleneck. 
    dil_rates : Tuple[int, int, int], default=(6,12,18)
        If use_aspp = True, dilation rates to use in the ASPP module. Must be a tuple of 3 integers.
    dil_chn : Tuple[int, int, int, int], default=(256,256,256,256)
        If use_aspp = True, number of feature maps to be generated by each component of the ASPP module. Must be a tuple of 4 integers.
    negative_slope : float, default=0.01
        If act_func = "lrelu", negative slope term. 
    se_ratio : int, default=8
        If use_se = True, reduction ratio to use.
    
    Returns
    -------
        Model as nn.Module subclass.
    """
    def __init__(
        self,
        in_chn: int = 1,
        n_cls: int = 2,
        act_func: Literal["relu", "lrelu", "swish"] = "relu",
        use_attn: bool = False,
        use_se: bool = False,
        use_res: bool = False,
        use_aspp: bool = False,
        en_chn: Tuple[int, int, int, int] = (16, 32, 64, 128),
        dc_chn: Tuple[int, int, int, int] = (128, 64, 32, 16),
        btn_chn: int = 256,
        dil_rates: Tuple[int, int, int] = (6, 12, 18),
        dil_chn: Tuple[int, int, int, int] = (256, 256, 256, 256),
        negative_slope: float = 0.01,
        se_ratio: int = 8,
    ) -> nn.Module:
        
        """
        Initialize a configurable U-Net for terrain semantic segmentation.

        The architecture consists of a 4-stage encoder with max-pooling downsampling,
        a bottleneck block (standard or ASPP; residual or non-residual), and a 4-stage
        decoder that upsamples and fuses encoder skip features via concatenation.
        Optional squeeze-and-excitation (SE) modules can be applied to encoder outputs,
        and optional attention gates can be applied along skip connections prior to
        concatenation in the decoder.

        Parameters
        ----------
        in_chn : int, default=1
            Number of input channels in the input tensor (C dimension).
        n_cls : int, default=2
            Number of segmentation classes (output logits per pixel).
        act_func : {"relu", "lrelu", "swish"}, default="relu"
            Activation function used throughout convolutional blocks.
            - ``"relu"``: ReLU
            - ``"lrelu"``: LeakyReLU (uses `negative_slope`)
            - ``"swish"``: Swish / SiLU (depending on your block implementation)
        use_attn : bool, default=False
            If True, apply attention gates to skip features (encoder outputs) conditioned
            on decoder/gating features before skip fusion.
        use_se : bool, default=False
            If True, apply SE modules to encoder stage outputs (`e1`..`e4`) before pooling.
        use_res : bool, default=False
            If True, use residual variants of double-convolution blocks in encoder and
            decoder, and residual variants of the bottleneck blocks when available.
        use_aspp : bool, default=False
            If True, use an atrous spatial pyramid pooling (ASPP) bottleneck instead of a
            standard bottleneck.
        en_chn : tuple[int, int, int, int], default=(16, 32, 64, 128)
            Output channel widths for encoder stages (`e1`..`e4`). Must be length 4.
        dc_chn : tuple[int, int, int, int], default=(128, 64, 32, 16)
            Output channel widths for decoder stages (`d1`..`d4`). Must be length 4.
            Note that decoder blocks receive concatenated tensors; their *input* channel
            counts are determined by the concatenation of upsampled decoder features and
            corresponding encoder skip features.
        btn_chn : int, default=256
            Number of output channels produced by the bottleneck block and used as the
            initial decoder width.
        dil_rates : tuple[int, int, int], default=(6, 12, 18)
            Dilation rates used by the ASPP bottleneck when `use_aspp=True`.
            Must be length 3.
        dil_chn : tuple[int, int, int, int], default=(256, 256, 256, 256)
            Channel widths for ASPP sub-branches when `use_aspp=True` (e.g., 1x1 branch,
            dilated conv branches, and pooling/image-level branch). Interpretation depends
            on `AsppBlk`/`AsppBlkR`. Must be length 4.
        negative_slope : float, default=0.01
            Negative slope used when `act_func="lrelu"`.
        se_ratio : int, default=8
            Reduction ratio used by SE modules when `use_se=True`.

        Raises
        ------
        ValueError
            If `act_func` is not one of ``{"relu", "lrelu", "swish"}``.
        AssertionError
            If `en_chn` or `dc_chn` is not length 4, or if ASPP tuples are invalid
            (implementation-dependent; recommended to validate if you add checks).

        Notes
        -----
        Module selection is conditional on flags:
        - Encoder/decoder blocks: `DoubleConvBlk` vs `DoubleConvBlkR` (if `use_res`).
        - Bottleneck: `Bottleneck`/`BottleneckR` vs `AsppBlk`/`AsppBlkR` (if `use_aspp` and/or `use_res`).
        - SE modules (`SEModule`) are created only when `use_se=True`.
        - Attention gates (`AttnBlk`) are created only when `use_attn=True`.

        The final prediction head (`ClassifierBlk`) outputs logits with shape
        ``(B, n_cls, H, W)`` (assuming input spatial dimensions are divisible by 16).
        """


        super().__init__()
        self.use_attn = bool(use_attn)
        self.use_se = bool(use_se)
        self.use_res = bool(use_res)
        self.use_aspp = bool(use_aspp)

        enc_block = DoubleConvBlkR if self.use_res else DoubleConvBlk
        self.e1 = enc_block(in_chn, en_chn[0], act_func=act_func, negative_slope=negative_slope)
        self.e2 = enc_block(en_chn[0], en_chn[1], act_func=act_func, negative_slope=negative_slope)
        self.e3 = enc_block(en_chn[1], en_chn[2], act_func=act_func, negative_slope=negative_slope)
        self.e4 = enc_block(en_chn[2], en_chn[3], act_func=act_func, negative_slope=negative_slope)

        # bottleneck
        if (not self.use_aspp) and (not self.use_res):
            self.btn = Bottleneck(en_chn[3], out_chn=btn_chn, act_func=act_func, negative_slope=negative_slope)
        elif (not self.use_aspp) and self.use_res:
            self.btn = BottleneckR(en_chn[3], out_chn=btn_chn, act_func=act_func, negative_slope=negative_slope)
        elif self.use_aspp and (not self.use_res):
            self.btn = AsppBlk(en_chn[3], out_chn=btn_chn, dil_chn=dil_chn, dil_rates=dil_rates,
                               act_func=act_func, negative_slope=negative_slope)
        else:
            self.btn = AsppBlkR(en_chn[3], out_chn=btn_chn, dil_chn=dil_chn, dil_rates=dil_rates,
                                act_func=act_func, negative_slope=negative_slope)

        # up + decoder
        self.dUp1 = UpConvBlk(btn_chn, btn_chn, act_func=act_func, negative_slope=negative_slope)
        self.dUp2 = UpConvBlk(dc_chn[0], dc_chn[0], act_func=act_func, negative_slope=negative_slope)
        self.dUp3 = UpConvBlk(dc_chn[1], dc_chn[1], act_func=act_func, negative_slope=negative_slope)
        self.dUp4 = UpConvBlk(dc_chn[2], dc_chn[2], act_func=act_func, negative_slope=negative_slope)

        self.d1 = enc_block(btn_chn + en_chn[3], dc_chn[0], act_func=act_func, negative_slope=negative_slope)
        self.d2 = enc_block(dc_chn[0] + en_chn[2], dc_chn[1], act_func=act_func, negative_slope=negative_slope)
        self.d3 = enc_block(dc_chn[1] + en_chn[1], dc_chn[2], act_func=act_func, negative_slope=negative_slope)
        self.d4 = enc_block(dc_chn[2] + en_chn[0], dc_chn[3], act_func=act_func, negative_slope=negative_slope)

        # SE
        if self.use_se:
            self.se1 = SEModule(en_chn[0], ratio=se_ratio)
            self.se2 = SEModule(en_chn[1], ratio=se_ratio)
            self.se3 = SEModule(en_chn[2], ratio=se_ratio)
            self.se4 = SEModule(en_chn[3], ratio=se_ratio)

        # Attention gates (xChn=en, gChn=decoder/gate channels)
        if self.use_attn:
            self.ag1 = AttnBlk(en_chn[0], dc_chn[2])
            self.ag2 = AttnBlk(en_chn[1], dc_chn[1])
            self.ag3 = AttnBlk(en_chn[2], dc_chn[0])
            self.ag4 = AttnBlk(en_chn[3], btn_chn)

        # heads
        self.c4 = ClassifierBlk(dc_chn[3], n_cls)


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

        e1 = self.e1(x)
        if self.use_se:
            e1 = self.se1(e1)
        e1_mp = F.max_pool2d(e1, kernel_size=2, stride=2)

        e2 = self.e2(e1_mp)
        if self.use_se:
            e2 = self.se2(e2)
        e2_mp = F.max_pool2d(e2, kernel_size=2, stride=2)

        e3 = self.e3(e2_mp)
        if self.use_se:
            e3 = self.se3(e3)
        e3_mp = F.max_pool2d(e3, kernel_size=2, stride=2)

        e4 = self.e4(e3_mp)
        if self.use_se:
            e4 = self.se4(e4)
        e4_mp = F.max_pool2d(e4, kernel_size=2, stride=2)

        btn = self.btn(e4_mp)

        if self.use_attn:
            e4 = self.ag4(e4, btn)
        d1_up = self.dUp1(btn)
        d1 = self.d1(torch.cat([d1_up, e4], dim=1))

        if self.use_attn:
            e3 = self.ag3(e3, d1)
        d2_up = self.dUp2(d1)
        d2 = self.d2(torch.cat([d2_up, e3], dim=1))

        if self.use_attn:
            e2 = self.ag2(e2, d2)
        d3_up = self.dUp3(d2)
        d3 = self.d3(torch.cat([d3_up, e2], dim=1))

        if self.use_attn:
            e1 = self.ag1(e1, d3)
        d4_up = self.dUp4(d3)
        d4 = self.d4(torch.cat([d4_up, e1], dim=1))

        c4 = self.c4(d4)

        return c4
