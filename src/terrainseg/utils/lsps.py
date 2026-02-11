import os
import random
import math
from dataclasses import dataclass
from typing import Sequence, Optional, Dict, Any, Union, Tuple, Literal
import torch.nn.functional as F
import torch
import torch.nn as nn


from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import torchmetrics as tm

import numpy as np
import pandas as pd
import torch.nn as nn

import geopandas as gpd
from shapely.geometry import Point, box

import rasterio as rio
import pandas as pd
from rasterio.windows import Window
from rasterio.features import rasterize
from rasterio.transform import Affine
import albumentations as A


class gaussPyramids(nn.Module):
    def __init__(self, 
                 inChn: int, 
                 spatDims: int, 
                 device=None, 
                 dtype=torch.float32) ->  nn.Module:
        super().__init__()
        self.inChn = inChn
        self.spatDims = spatDims

        if device is None:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # 5x5 Gaussian kernel (fixed)
        gauss = torch.tensor(
            [1, 4, 6, 4, 1,
             4, 16, 24, 16, 4,
             6, 24, 36, 24, 6,
             4, 16, 24, 16, 4,
             1, 4, 6, 4, 1],
            device=device, dtype=dtype
        ).view(5, 5) / 256.0

        # Depthwise kernel so it works for arbitrary inChn:
        # shape: [C, 1, 5, 5], used with groups=C
        gauss_kernel = gauss.view(1, 1, 5, 5).repeat(inChn, 1, 1, 1)
        self.register_buffer("gauss_kernel", gauss_kernel, persistent=True)

        # Build checkerboard-like maskGrid (same intent as your gridCol * gridRow)
        # This produces 1 at positions where BOTH row and col are "odd" in that pattern.
        # Equivalent and simpler: mask[i,j] = (i%2==0) & (j%2==0) in 0-based indexing.
        idx = torch.arange(spatDims, device=device)
        mask = ((idx[:, None] % 2 == 0) & (idx[None, :] % 2 == 0)).to(dtype)

        # shape: [1,1,H,W] so it broadcasts over [N,C,H,W]
        self.register_buffer("maskGridT", mask.view(1, 1, spatDims, spatDims), persistent=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: [N, C, H, W]  (H=W should match spatDims for the mask as written)
        returns: [N, 5*C, H, W]  (concatenation of 5 pyramid levels along channel dim)
        """
        # Depthwise conv: groups=C
        def blur(z):
            return F.conv2d(z, self.gauss_kernel, stride=1, padding=2, groups=z.shape[1])

        l1_1 = blur(x)

        l1_2 = blur(l1_1 * self.maskGridT) * 4.0
        l1_3 = blur(l1_2 * self.maskGridT) * 4.0
        l1_4 = blur(l1_3 * self.maskGridT) * 4.0
        l1_5 = blur(l1_4 * self.maskGridT) * 4.0

        # concat along channel dimension
        return torch.cat([l1_1, l1_2, l1_3, l1_4, l1_5], dim=1)


class lspCalc(nn.Module):
    def __init__(
        self,
        cellSize: float = 1.0,
        innerRadius: int = 2,
        outerRadius: int = 5,
        hsRadius: int = 50,
        smoothRadius: int = 11,
        doTPIHS: bool = True,
        device=None,
        dtype=torch.float32,
    ) -> nn.Module:
        super().__init__()
        self.cellSize = float(cellSize)
        self.innerRadius = int(innerRadius)
        self.outerRadius = int(outerRadius)
        self.hsRadius = int(hsRadius)
        self.smoothRadius = int(smoothRadius)
        self.doTPIHS = bool(doTPIHS)

        if device is None:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Sun geometry buffers (radians)
        self.register_buffer("sunAltitudeT", torch.tensor((90.0 - 45.0) * (3.141592653589793 / 180.0), device=device, dtype=dtype))
        self.register_buffer("sunAzimuthNT",  torch.tensor(((360.0 - 360.0) + 90.0) * (3.141592653589793/ 180.0), device=device, dtype=dtype))
        self.register_buffer("sunAzimuthWT",  torch.tensor(((360.0 - 270.0) + 90.0) * (3.141592653589793 / 180.0), device=device, dtype=dtype))
        self.register_buffer("sunAzimuthET",  torch.tensor(((360.0 -  90.0) + 90.0) * (3.141592653589793 / 180.0), device=device, dtype=dtype))
        self.register_buffer("sunAzimuthST",  torch.tensor(((360.0 - 180.0) + 90.0) * (3.141592653589793 / 180.0), device=device, dtype=dtype))

        # 1) Slope kernels (Sobel-like)
        kx_init = torch.tensor(
            [[-1, 0, 1],
             [-2, 0, 2],
             [-1, 0, 1]],
            device=device, dtype=dtype
        ).view(1, 1, 3, 3)

        ky_init = torch.tensor(
            [[-1, -2, -1],
             [ 0,  0,  0],
             [ 1,  2,  1]],
            device=device, dtype=dtype
        ).view(1, 1, 3, 3)

        # Curvature kernels (normalized versions)
        kx_curv = kx_init / 8.0
        ky_curv = ky_init / 8.0

        kxx_curv = torch.tensor(
            [[ 1, -2,  1],
             [ 1, -2,  1],
             [ 1, -2,  1]],
            device=device, dtype=dtype
        ).view(1, 1, 3, 3) / 3.0

        kyy_curv = torch.tensor(
            [[ 1,  1,  1],
             [-2, -2, -2],
             [ 1,  1,  1]],
            device=device, dtype=dtype
        ).view(1, 1, 3, 3) / 3.0

        kxy_curv = torch.tensor(
            [[ 1,  0, -1],
             [ 0,  0,  0],
             [-1,  0,  1]],
            device=device, dtype=dtype
        ).view(1, 1, 3, 3) / 4.0

        self.register_buffer("kx_slope", kx_init)
        self.register_buffer("ky_slope", ky_init)

        self.register_buffer("kx_curv", kx_curv)
        self.register_buffer("ky_curv", ky_curv)
        self.register_buffer("kxx_curv", kxx_curv)
        self.register_buffer("kyy_curv", kyy_curv)
        self.register_buffer("kxy_curv", kxy_curv)

        # 2) Annulus kernel
        annulus_size = 2 * self.outerRadius + 1
        annulus = torch.zeros((1, 1, annulus_size, annulus_size), device=device, dtype=dtype)
        c = self.outerRadius
        for i in range(annulus_size):
            for j in range(annulus_size):
                dist = math.sqrt((i - c) ** 2 + (j - c) ** 2)
                if (dist >= self.innerRadius) and (dist <= self.outerRadius):
                    annulus[0, 0, i, j] = 1.0
        self.register_buffer("annulus_kernel", annulus)
        self.register_buffer("annulus_area", annulus.sum())

        # 3) Hillslope kernel (disk)
        hs_size = 2 * self.hsRadius + 1
        hs = torch.zeros((1, 1, hs_size, hs_size), device=device, dtype=dtype)
        c = self.hsRadius
        for i in range(hs_size):
            for j in range(hs_size):
                dist = math.sqrt((i - c) ** 2 + (j - c) ** 2)
                if dist <= self.hsRadius:
                    hs[0, 0, i, j] = 1.0
        self.register_buffer("hs_kernel", hs)
        self.register_buffer("hs_area", hs.sum())

        # 4) Smoothness kernel (disk)
        sm_size = 2 * self.smoothRadius + 1
        sm = torch.zeros((1, 1, sm_size, sm_size), device=device, dtype=dtype)
        c = self.smoothRadius
        for i in range(sm_size):
            for j in range(sm_size):
                dist = math.sqrt((i - c) ** 2 + (j - c) ** 2)
                if dist <= self.smoothRadius:
                    sm[0, 0, i, j] = 1.0
        self.register_buffer("smth_kernel", sm)
        self.register_buffer("smth_area", sm.sum())

    def forward(self, inDTM: torch.Tensor) -> torch.Tensor:
        """
        inDTM: [N, 1, H, W]
        returns:
          if doTPIHS: [N, 6, H, W]  (tpiHS, slp, tpiL, hillshade, crvPro, crvPln)
          else:       [N, 5, H, W]  (slp, tpiL, hillshade, crvPro, crvPln)
        """

        # 1) Slope
        dx = F.conv2d(inDTM, self.kx_slope, padding=1) / (8.0 * self.cellSize)
        dy = F.conv2d(inDTM, self.ky_slope, padding=1) / (8.0 * self.cellSize)

        gradMag = torch.sqrt(dx * dx + dy * dy)
        slpR = torch.atan(gradMag)                 # radians
        slp = slpR * 57.2958                       # degrees
        slp = torch.sqrt(slp)
        slp = torch.clamp(slp, 0.0, 10.0) / 10.0   # [0,1]

        aspect = 3.141592653589793 / 2.0 - torch.atan2(-dy, dx)

        # 2) Hillshade (average of N/E/W/S)
        hillshadeN = (torch.cos(self.sunAltitudeT) * torch.cos(slpR) +
                      torch.sin(self.sunAltitudeT) * torch.sin(slpR) *
                      torch.cos(self.sunAzimuthNT - aspect)) * 255.0

        hillshadeE = (torch.cos(self.sunAltitudeT) * torch.cos(slpR) +
                      torch.sin(self.sunAltitudeT) * torch.sin(slpR) *
                      torch.cos(self.sunAzimuthET - aspect)) * 255.0

        hillshadeW = (torch.cos(self.sunAltitudeT) * torch.cos(slpR) +
                      torch.sin(self.sunAltitudeT) * torch.sin(slpR) *
                      torch.cos(self.sunAzimuthWT - aspect)) * 255.0

        hillshadeS = (torch.cos(self.sunAltitudeT) * torch.cos(slpR) +
                      torch.sin(self.sunAltitudeT) * torch.sin(slpR) *
                      torch.cos(self.sunAzimuthST - aspect)) * 255.0

        hillshade = (hillshadeN + hillshadeE + hillshadeW + hillshadeS) / 4.0
        hillshade = torch.clamp(hillshade, 0.0, 255.0) / 255.0

        # 3) Local TPI (annulus)
        neighborhood_sum = F.conv2d(inDTM, self.annulus_kernel, padding=self.outerRadius)
        neighborhood_mean = neighborhood_sum / (self.annulus_area + 1e-12)
        tpiL = inDTM - neighborhood_mean
        tpiL = torch.clamp(tpiL, -10.0, 10.0)
        tpiL = (tpiL + 10.0) / 20.0

        # 4) Hillslope TPI (disk) optional
        if self.doTPIHS:
            hs_sum = F.conv2d(inDTM, self.hs_kernel, padding=self.hsRadius)
            hs_mean = hs_sum / (self.hs_area + 1e-12)
            tpiHS = inDTM - hs_mean
            tpiHS = torch.clamp(tpiHS, -10.0, 10.0)
            tpiHS = (tpiHS + 10.0) / 20.0

        # 5) Curvatures (smooth then derivatives)
        sum_elev = F.conv2d(inDTM, self.smth_kernel, padding=self.smoothRadius)
        mean_elev = sum_elev / (self.smth_area + 1e-12)

        p  = F.conv2d(mean_elev, self.kx_curv,  padding=1)
        q  = F.conv2d(mean_elev, self.ky_curv,  padding=1)
        r_ = F.conv2d(mean_elev, self.kxx_curv, padding=1)
        t_ = F.conv2d(mean_elev, self.kyy_curv, padding=1)
        s_ = F.conv2d(mean_elev, self.kxy_curv, padding=1)

        # remove channel dim: [N, H, W]
        p_ = p.squeeze(1)
        q_ = q.squeeze(1)
        r_ = r_.squeeze(1)
        s_ = s_.squeeze(1)
        t_ = t_.squeeze(1)

        slope_sq = p_*p_ + q_*q_
        denom = slope_sq**1.5 + 1e-12

        crvPln = (q_*q_ * r_ - 2.0 * p_ * q_ * s_ + p_*p_ * t_) / denom
        crvPro = (p_*p_ * r_ + 2.0 * p_ * q_ * s_ + q_*p_ * t_) / denom

        crvPln = torch.clamp(crvPln, -0.1, 0.1)
        crvPln = (crvPln + 0.1) / 0.2

        crvPro = torch.clamp(crvPro, -0.1, 0.1)
        crvPro = (crvPro + 0.1) / 0.2

        crvPln = crvPln.unsqueeze(1)  # [N,1,H,W]
        crvPro = crvPro.unsqueeze(1)

        # 6) Concatenate outputs along channel dim
        if self.doTPIHS:
            out = torch.cat([tpiHS, slp, tpiL, hillshade, crvPro, crvPln], dim=1)
        else:
            out = torch.cat([slp, tpiL, hillshade, crvPro, crvPln], dim=1)

        return out


class lspModule(nn.Module):
    def __init__(
        self,
        cell_size: float = 1.0,
        spat_dim: int = 640,
        do_gp: bool = False,
        inner_radius: float = 2.0,
        outer_radius: float = 10.0,
        hs_radius: float = 50.0,
        smooth_radius: float = 11.0
    ):
        """
        PyTorch translation of your R torch module.

        Notes
        -----
        - Expects input x shaped [B, 1, H, W] if you are computing LSPs from a DEM-like single channel.
        - If do_gp=True, gaussian pyramid is computed and LSPs are computed for each pyramid level.
        - `seg_module` is the actual segmentation network (UNet/DeepLab/etc.) that consumes tIn.
          If you don't pass it, you must set `self.segMod` later.
        """
        super().__init__()

        self.cell_size = cell_size
        self.spat_dim = spat_dim
        self.do_gp = do_gp
        self.inner_radius = inner_radius
        self.outer_radius = outer_radius
        self.hs_radius = hs_radius
        self.smooth_radius = smooth_radius

        # Mirrors your R logic (used mainly for bookkeeping)
        self.in_channels = 31 if self.do_gp else 6

        # Modules (assumed available)
        self.gaussPyramid = gaussPyramids(1, self.spat_dim)

        self.lspOrig = lspCalc(
            cellSize=self.cell_size,
            innerRadius=self.inner_radius,
            outerRadius=self.outer_radius,
            hsRadius=self.hs_radius,
            smoothRadius=self.smooth_radius,
            doTPIHS=True,
        )

        self.lspGP = lspCalc(
            cellSize=self.cell_size,
            innerRadius=self.inner_radius,
            outerRadius=self.outer_radius,
            hsRadius=self.hs_radius,
            smoothRadius=self.smooth_radius,
            doTPIHS=False,
        )


    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: torch.Tensor [B, 1, H, W] (typical) or whatever lspOrig expects.
        """
        if self.do_gp:
            # In your R code, xGP is indexed like xGP[,1,,] etc.
            # That implies xGP is [B, 5, H, W] (5 pyramid levels).
            x_gp = self.gaussPyramid(x)  # expected [B, 5, H, W]

            # LSPs from original resolution
            x_lsp = self.lspOrig(x)      # expected [B, C0, H, W]

            # Split pyramid levels, add channel dim back: [B, 1, H, W]
            x_gp1 = x_gp[:, 0, :, :].unsqueeze(1)
            x_gp2 = x_gp[:, 1, :, :].unsqueeze(1)
            x_gp3 = x_gp[:, 2, :, :].unsqueeze(1)
            x_gp4 = x_gp[:, 3, :, :].unsqueeze(1)
            x_gp5 = x_gp[:, 4, :, :].unsqueeze(1)

            # LSPs per pyramid level
            x_gp_lsp1 = self.lspGP(x_gp1)
            x_gp_lsp2 = self.lspGP(x_gp2)
            x_gp_lsp3 = self.lspGP(x_gp3)
            x_gp_lsp4 = self.lspGP(x_gp4)
            x_gp_lsp5 = self.lspGP(x_gp5)

            # Concatenate along channel dim (R used dim=2 for NCHW -> channel)
            t_in = torch.cat(
                [x_lsp, x_gp_lsp1, x_gp_lsp2, x_gp_lsp3, x_gp_lsp4, x_gp_lsp5],
                dim=1,
            )
        else:
            t_in = self.lspOrig(x)

        return t_in