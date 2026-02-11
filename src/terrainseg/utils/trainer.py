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

from terrainseg.utils import lsps
from terrainseg.utils import dynamicchips


def crop_tensor(inT: torch.Tensor, crpFactor: int) -> torch.Tensor:
    startCR = crpFactor
    endCR = inT.size(2) - crpFactor

    outT = inT[:, :, startCR:endCR, startCR:endCR]
    return outT


class terrainDataset(Dataset):
    """
    PyTorch dataset for raster-based terrain semantic segmentation.

    Each sample consists of a single-band raster image and a corresponding
    segmentation mask read from disk using rasterio. The dataset expects a
    pandas DataFrame containing file paths to images and masks. Optional
    transforms (e.g., Albumentations) can be applied jointly to image and mask.

    Parameters
    ----------
    df : pandas.DataFrame
        Table containing file paths to input rasters and masks. The dataset
        expects:
        - column index 1 : image path
        - column index 2 : mask path
    transform : callable or None, default=None
        Joint transform applied to both image and mask. Must accept
        ``transform(image=img, mask=mask)`` and return a dictionary with
        keys ``"image"`` and ``"mask"`` (e.g., Albumentations Compose).

    Notes
    -----
    - Images are read as single-band rasters and returned as float tensors
      of shape ``(1, H, W)``.
    - Masks are converted to integer class labels using rounding and returned
      as ``torch.long`` tensors of shape ``(H, W)``.
    - Label value ``0`` is preserved and can be treated as an ignore index
      during loss/metric computation.
    """
    def __init__(self, 
                 df: pd.DataFrame, 
                 transform: None):
        
        """
        Initialize the terrain dataset.

        Parameters
        ----------
        df : pandas.DataFrame
            DataFrame containing image and mask file paths.
        transform : callable or None, default=None
            Optional joint transform applied to image and mask.
        """

        super().__init__()
        self.df = df
        self.transform = transform

    def __getitem__(self, idx):
        """
        Load a single sample from the dataset.

        Parameters
        ----------
        idx : int
            Index of the sample.

        Returns
        -------
        tuple[torch.Tensor, torch.Tensor]
            image : torch.Tensor
                Float tensor of shape ``(1, H, W)``.
            mask : torch.Tensor
                Long tensor of shape ``(H, W)`` containing integer class labels.

        Notes
        -----
        - Images are read using rasterio and converted to float tensors.
        - Masks are rounded to the nearest integer and cast to int64.
        - If a transform is provided, it is applied before conversion to tensors.
        """

        image_name = self.df.iloc[idx, 1]
        mask_name  = self.df.iloc[idx, 2]

        # ---- read multi-band image (H, W, C) ----
        with rio.open(image_name) as src:
            img = src.read(1)  # (H, W)

        # ---- read mask (single band) ----
        with rio.open(mask_name) as src:
            mask = src.read(1)  # (H, W)

        # masks are float in your case -> make integer labels
        mask = np.rint(mask).astype(np.int64)

        # Option B: ignore unlabeled where mask==0 (do NOT remap)
        # keep {0,1,2,3} as-is, loss/metrics will ignore 0

        if self.transform is not None:

            transformed = self.transform(image=img, mask=mask)
            img  = transformed["image"]
            mask = transformed["mask"]
        img  = torch.from_numpy(img).unsqueeze(0).float()  # (C,H,W)
        mask = torch.from_numpy(mask).long()

        return img, mask

    def __len__(self):
        """
        Return dataset size.

        Returns
        -------
        int
            Number of samples in the dataset.
        """
        return len(self.df)






class terrainDatasetDynamic(Dataset):
    """
    PyTorch dataset that dynamically generates raster chips for terrain
    semantic segmentation.

    Instead of reading pre-saved rasters from disk, each sample is created
    on-the-fly using `dynamicchips.makeDynamicChip`, typically from vector
    geometry (e.g., polygons, points, or bounding boxes stored in a
    GeoDataFrame). This allows stochastic sampling, jittering, and extremely
    large training datasets without storing image tiles on disk.

    Parameters
    ----------
    df : geopandas.GeoDataFrame
        Table describing chip locations and metadata required by
        `dynamicchips.makeDynamicChip`. Each row must contain all fields
        required by that function (e.g., geometry, raster paths, attributes).
    chip_size : int
        Output chip width/height in pixels.
    cell_size : int
        Pixel resolution used when rasterizing the chip (map units per pixel).
    background_value : int, default=0
        Value assigned to pixels outside valid data regions. Typically used
        as an ignore label in loss functions.
    transforms : callable or None, default=None
        Joint transform applied to both image and mask. Must accept
        ``transform(image=img, mask=mask)`` and return a dictionary with
        keys ``"image"`` and ``"mask"`` (e.g., Albumentations Compose).

    Notes
    -----
    - Chips are generated dynamically at each access; repeated epochs may
      yield slightly different samples if the chip generator is stochastic.
    - Masks are rounded and converted to integer labels.
    - Label value ``0`` is preserved and can be treated as an ignore index.
    - Returned image tensor has shape ``(1, H, W)`` and dtype ``float32``.
    """

    def __init__(self, 
                 df: gpd.GeoDataFrame, 
                 chip_size: int, 
                 cell_size: int, 
                 background_value: int = 0, 
                 transforms=None):
        
        """
        Initialize the dynamic terrain dataset.

        Parameters
        ----------
        df : geopandas.GeoDataFrame
            Source table describing chip sampling locations.
        chip_size : int
            Spatial size of generated raster chips in pixels.
        cell_size : int
            Pixel resolution of the generated chip.
        background_value : int, default=0
            Fill value assigned to empty pixels.
        transforms : callable or None, default=None
            Optional joint transform applied to image and mask.
        """
        super().__init__()
        self.df = df
        self.chip_size = chip_size
        self.cell_size = cell_size
        self.background_value = background_value
        self.transforms=transforms

    def __getitem__(self, i):
        """
        Generate and return a single training sample.

        Parameters
        ----------
        i : int
            Index of the sample.

        Returns
        -------
        tuple[torch.Tensor, torch.Tensor]
            image : torch.Tensor
                Float tensor of shape ``(1, H, W)``.
            mask : torch.Tensor
                Long tensor of shape ``(H, W)`` containing integer class labels.

        Notes
        -----
        - Chips are created using `dynamicchips.makeDynamicChip`.
        - Mask values are rounded to integers.
        - If transforms are provided, they are applied before tensor conversion.
        """

        row = self.df.iloc[i]

        out = dynamicchips.makeDynamicChip(
            chip_row=row,
            chip_size=self.chip_size,
            cell_size=self.cell_size,
            background_value=self.background_value,
        )

        img = out["image"].astype(np.float32)  # (C,H,W)
        mask = out["mask"].astype(np.float32)


        # masks are float in your case -> make integer labels
        mask = np.rint(mask).astype(np.int64)

        # Option B: ignore unlabeled where mask==0 (do NOT remap)
        # keep {0,1,2,3} as-is, loss/metrics will ignore 0

        if self.transforms is not None:
            transformed = self.transforms(image=img, mask=mask)
            img  = transformed["image"]
            mask = transformed["mask"]
        img  = torch.from_numpy(img).unsqueeze(0).float() 
        mask = torch.from_numpy(mask).long()

        return img, mask

    def __len__(self):
        """
        Return dataset size.

        Returns
        -------
        int
            Number of sampling locations available.
        """
        return len(self.df)


class terrainSegModel(nn.Module):
    """
    Initialize terrain segmentation pipeline model.

    This class wraps a segmentation neural network and augments its input with
    locally derived spatial predictors (LSPs) computed from the input raster.
    The forward pass performs:

    1. LSP feature generation (`lspModule`)
    2. Edge cropping to remove convolution padding artifacts
    3. Segmentation model inference

    Parameters
    ----------
    modelIn : nn.Module
        Segmentation model receiving the processed tensor as input
        (e.g., UNet, ConvNeXt-UNet, etc.). The model must accept an input
        channel count matching the LSP output.

    cell_size : float, default=1.0
        Spatial resolution of input raster (map units per pixel).
        Used for scale-aware terrain derivatives.

    spat_dim : int, default=640
        Spatial size expected by the LSP module before cropping.

    t_crop : int, default=64
        Number of pixels trimmed from each border after LSP generation.
        Removes convolution boundary artifacts before inference.

    do_gp : bool, default=False
        If True, include Gaussian pyramid features in the LSP stack.
        Changes input channel count:
        - False → 6 channels
        - True  → 31 channels

    inner_radius : float, default=2.0
        Inner radius used in local spatial pattern calculations.

    outer_radius : float, default=10.0
        Outer radius used in annular terrain derivatives.

    hs_radius : float, default=50.0
        Search radius used for hillshade-style or horizon-based features.

    smooth_radius : float, default=11.0
        Smoothing radius applied to terrain derivatives.

    Notes
    -----
    The wrapped segmentation model (`modelIn`) must be configured to accept
    the channel count produced by the LSP module:

    - 6 channels when `do_gp=False`
    - 31 channels when `do_gp=True`

    The cropping step ensures predictions are spatially aligned and not
    influenced by padding artifacts introduced during feature computation.
    """

    def __init__(
            self,
            modelIn: nn.Module,
            cell_size: float = 1.0,
            spat_dim: int = 640,
            t_crop: int = 64,
            do_gp: bool = False,
            inner_radius: float = 2.0,
            outer_radius: float = 10.0,
            hs_radius: float = 50.0,
            smooth_radius: float = 11.0,
    ):
        super().__init__()  # <-- REQUIRED

        self.do_gp = do_gp
        self.cell_size = cell_size
        self.spat_dim = spat_dim
        self.t_crop = t_crop
        self.inner_radius = inner_radius
        self.outer_radius = outer_radius
        self.hs_radius = hs_radius
        self.smooth_radius = smooth_radius

        self.inChn = 31 if self.do_gp else 6

        # These are nn.Modules, so they must be assigned after super().__init__()
        self.trnM = modelIn

        self.lspM = lsps.lspModule(
            cell_size=self.cell_size,
            spat_dim=self.spat_dim,
            do_gp=self.do_gp,
            inner_radius=self.inner_radius,
            outer_radius=self.outer_radius,
            hs_radius=self.hs_radius,
            smooth_radius=self.smooth_radius
        )

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
        x = self.lspM(x)
        x = crop_tensor(x, crpFactor=self.t_crop)
        x = self.trnM(x)
        return x


class unifiedFocalLoss(nn.Module):
    def __init__(
        self,
        nCls: int = 3,
        lambda_: float = 0.5,
        gamma: float = 0.5,
        delta: float = 0.6,
        smooth: float = 1e-8,
        zeroStart: bool = True,
        clsWghtsDist: float = 1.0,
        clsWghtsReg: float = 1.0,
        useLogCosH: bool = False,
        device: str = "cuda"
    ):
        """
        Initialize the Unified Focal Loss.

        This loss combines a focal cross-entropy distribution term and a
        Tversky-style region overlap term into a single objective:

            loss = λ * focal_CE + (1-λ) * focal_Tversky

        The distribution component focuses on per-pixel classification accuracy,
        while the region component emphasizes spatial overlap and class imbalance
        robustness. Class-specific weights can be applied independently to each
        component.

        Parameters
        ----------
        nCls : int, default=3
            Number of segmentation classes (channel dimension of model output).
        lambda_ : float, default=0.5
            Weighting between distribution and region terms.
            - 1.0 → pure focal cross-entropy
            - 0.0 → pure focal Tversky loss
        gamma : float, default=0.5
            Focusing parameter controlling hard-example emphasis in both loss terms.
            Higher values increase emphasis on misclassified pixels.
        delta : float or sequence[float], default=0.6
            Tversky false-negative weighting parameter.
            Can be scalar (applied to all classes) or class-specific sequence.
        smooth : float, default=1e-8
            Numerical stability constant added to region overlap denominator.
        zeroStart : bool, default=True
            Placeholder flag retained for compatibility with prior label indexing
            conventions. Currently does not remap labels but preserved for parity.
        clsWghtsDist : float or sequence[float], default=1.0
            Class weights applied to the focal cross-entropy component.
            May be scalar or per-class weights.
        clsWghtsReg : float or sequence[float], default=1.0
            Class weights applied to the region (Tversky) component.
            May be scalar or per-class weights.
        useLogCosH : bool, default=False
            If True, applies log-cosh stabilization to the region loss to reduce
            gradient explosion for difficult samples.
        device : str, default="cuda"
            Target device for internal tensors. Registered buffers automatically
            move with ``model.to(device)``.

        Notes
        -----
        - Targets must be integer class labels, not one-hot encoded.
        - Prediction tensor should contain raw logits (no softmax).
        - Class weights and delta parameters are stored as registered buffers so
        they automatically follow device transfers.
        - Particularly suited for imbalanced segmentation tasks (e.g., terrain
        features, rare geomorphic classes, or boundary-dominated targets).
        """
        super().__init__()
        self.nCls = int(nCls)
        self.lambda_ = float(lambda_)
        self.gamma = float(gamma)
        self.delta = delta
        self.smooth = float(smooth)
        self.zeroStart = bool(zeroStart)
        self.useLogCosH = bool(useLogCosH)
        self.device=device

        # store as buffers so they move with .to(device)
        self.register_buffer("gammaRep", torch.full((self.nCls,), self.gamma, dtype=torch.float32))

        # delta can be scalar or list/tuple/tensor
        if isinstance(delta, (list, tuple, torch.Tensor)):
            d = torch.as_tensor(delta, dtype=torch.float32)
        else:
            d = torch.full((self.nCls,), float(delta), dtype=torch.float32)
        self.register_buffer("delta2", d)

        if isinstance(clsWghtsDist, (list, tuple, torch.Tensor)):
            wdist = torch.as_tensor(clsWghtsDist, dtype=torch.float32)
        else:
            wdist = torch.full((self.nCls,), float(clsWghtsDist), dtype=torch.float32)
        self.register_buffer("clsWghtsDist2", wdist)

        if isinstance(clsWghtsReg, (list, tuple, torch.Tensor)):
            wreg = torch.as_tensor(clsWghtsReg, dtype=torch.float32)
        else:
            wreg = torch.full((self.nCls,), float(clsWghtsReg), dtype=torch.float32)
        self.register_buffer("clsWghtsReg2", wreg)

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        pred:   [B,C,H,W] logits
        target: [B,1,H,W] or [B,H,W] integer labels
        """
        device = pred.device
        dtype = pred.dtype

        # ensure buffers are on same device/dtype as needed (device handled by .to()).
        gammaRep = self.gammaRep.to(device=device, dtype=torch.float32)
        delta2   = self.delta2.to(device=device, dtype=torch.float32)
        wDist    = self.clsWghtsDist2.to(device=device, dtype=torch.float32)
        wReg     = self.clsWghtsReg2.to(device=device, dtype=torch.float32)

        # target -> [B,H,W] long
        if target.ndim == 4:
            target1 = target.squeeze(1)
        else:
            target1 = target
        target1 = target1.long().to(device)

        if self.zeroStart:
            target1 = target1  # no-op kept for parity

        # ================= Distribution loss (focal CE) =================
        distMetric = None
        if self.lambda_ > 0.0:
            # log_probs: [B,C,H,W]
            logp = F.log_softmax(pred, dim=1)
            # NLL per pixel with class weights: [B,H,W]
            ce = F.nll_loss(logp, target1, weight=wDist, reduction="none")

            # pt = exp(-ce) but compute from logp directly to avoid extra exp of ce:
            # log_pt = logp.gather(1, target1.unsqueeze(1)).squeeze(1)
            log_pt = logp.gather(1, target1.unsqueeze(1)).squeeze(1)  # [B,H,W]
            pt = log_pt.exp()

            # your version: (1-pt)^(1-gamma) * CE
            # keep in float32 for stability
            gammaT = torch.tensor(self.gamma, device=device, dtype=torch.float32)
            focal = (1.0 - pt).clamp_min(0).pow(1.0 - gammaT) * ce

            # denominator: sum of weights for true class at each pixel
            # wDist[target] -> [B,H,W]
            denom = wDist.gather(0, target1.view(-1)).view_as(target1).sum().clamp_min(1e-12)
            distMetric = focal.sum() / denom

        # ================= Region loss (Tversky-ish) =================
        regMetric = None
        if self.lambda_ < 1.0:
            pred_soft = F.softmax(pred, dim=1)  # [B,C,H,W]

            # one_hot target (still used, but keep it efficient)
            # [B,H,W] -> [B,C,H,W]
            target_oh = F.one_hot(target1, num_classes=self.nCls).permute(0,3,1,2).to(pred_soft.dtype)

            # sums over batch + spatial
            dims = (0, 2, 3)
            tps = (pred_soft * target_oh).sum(dim=dims)
            fps = (pred_soft * (1.0 - target_oh)).sum(dim=dims)
            fns = ((1.0 - pred_soft) * target_oh).sum(dim=dims)

            smooth = torch.tensor(self.smooth, device=device, dtype=torch.float32)
            mTI = (tps + smooth) / (tps + (1.0 - delta2) * fps + delta2 * fns + smooth)

            reg = (1.0 - mTI).pow(gammaRep) * wReg
            regMetric = reg.sum() / wReg.sum().clamp_min(1e-12)

            if self.useLogCosH:
                regMetric = torch.log(torch.cosh(regMetric))

        # ================= Combine =================
        if self.lambda_ >= 1.0:
            return distMetric
        if self.lambda_ <= 0.0:
            return regMetric
        return (self.lambda_ * distMetric) + ((1.0 - self.lambda_) * regMetric)


def terrainTrainer(saveFolder,
                   trainDF: Union[pd.DataFrame, gpd.GeoDataFrame], 
                   valDF: Union[pd.DataFrame, gpd.GeoDataFrame],
                   trainableModel: nn.Module,
                   lossFnc: nn.Module,
                   useDynamicTrain: bool = False,
                   useDynamicVal: bool = False,
                   nCls: int = 2,
                   background_value: int = 0,
                   do_gp: bool = True,
                   cropFactor: int = 64,
                   epochs: int = 50,
                   batchSize: int = 10,
                   lr: float = 0.0001,
                   cell_size: int = 1,
                   spat_dim: int = 640,
                   inner_radius: float = 2.0,
                   outer_radius: float = 10.0,
                   hs_radius: float =  50.0,
                   smooth_radius: float = 11.0,
                   device:str = torch.device("cuda:0" if torch.cuda.is_available() else "cpu"),
                   doMultiGPU: bool = False,
                   doRotate90: bool = True,
                   doFlips: bool = True,
                   augProb: float = 0.3):
        """
        Train a semantic segmentation model to extract geomorphic or terrain features from a DTM.

        Parameters
        ----------
        trainDF : Union[pd.DataFrame, gpd.GeoDataFrame]
            Input dataframe created with defineChipsDF() or defineDynamicChipsGDF().
        valDF : Union[pd.DataFrame, gpd.GeoDataFrame]
            Input dataframe created with defineChipsDF() or defineDynamicChipsGDF().
        trainableModel : nn.Module
            Instantiated trainable model.
        lossFnc : nn.Module
            Instantiated loss metric.
        useDynamicTrain : bool, default=False
            Whether to use a dynamic chip generator for the training set.
        useDynamicVal : bool, default=False
            Whether to use a dynamic chip generator for the validation set. 
        nCls : int, default=2
            Number of output classes.
        background_value : int, default=0
            Integer value to assign to the background class.
        do_gp : bool, default=False
            Whethe or not to incorporate Gaussian Pyramids. If Gaussian Pyramids are used, the number of input predictor variable to the trainable model will be 31. If they are not used, the number of inputs will be 6. 
        cropFactor : int, default=64
            Number of rows and columns to crop from each side of the LSPs tensors prior to passing them to the trainable component of the model. 
        epochs : int, default=50
            Number of training epochs or iterations for the entire training set. 
        batchSize : int, default=10
            Mini-batch size. Note that the last mini-batch is dropped.
        lr : float, default=0.0001
            Default learning rate for optimization algorithm. Adamw is used. 
        cell_size : int, default=1
            Cell size of input images and masks
        spat_dim : int, default=640
            Size of each chip and associated mask in the spatial dimensions. 
        inner_radius : float, default=2.0
            Inner radius for local TPI calculation using an annulus window. 
        outer_radius : float, default=10.0
            Outer radius for local TPI calculation using an annulus window. 
        hs_radius : float, default= 50.0
            Outer radius for hillslope-scale TPI calculation using circular moving window. 
        smooth_radius : float, default=11.0
            Radius for circular smoothing kernel applied prior to curvature calculations. 
        device:str = default=torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
            Device used to generate LSPs and train the model.
        doMultiGPU : bool, default=False
            Whether or not to use multiple GPUs to train model. Must have computer with multiple GPUs.
        doRotate90 : bool, default=True
            Whether or not to include random rotations of 90-, 180, or 270-degrees in the data augmentations. 
        doFlips : bool, default=True
            Whether or not to include random horizontal or vertical flips in the data augmentations. 
        augProb : float, default=0.3
            Default probability for applying each random augmentation. 
        """
        
        if doRotate90 == True and doFlips == True:
            myTransforms = A.Compose([A.HorizontalFlip(p=augProb),
                                    A.VerticalFlip(p=augProb),
                                    A.RandomRotate90(p=augProb),])
        elif doFlips == True and doRotate90 == False:
            myTransforms = A.Compose([A.HorizontalFlip(p=augProb),
                                    A.VerticalFlip(p=augProb),])
        elif doRotate90 == True and doFlips == False:
            myTransforms = A.Compose([A.RandomRotate90(p=augProb),])
        else:
            myTransforms = None
        
        modelT = terrainSegModel(modelIn = trainableModel.to(device),
                                 cell_size = cell_size,
                                 spat_dim = spat_dim,
                                 t_crop = cropFactor,
                                 do_gp = do_gp,
                                 inner_radius = inner_radius,
                                 outer_radius = outer_radius,
                                 hs_radius = hs_radius,
                                 smooth_radius = smooth_radius).to(device)
        
        if doMultiGPU == True:
          modelT = nn.DataParallel(modelT)
    
        criterion = lossFnc
        optimizer = torch.optim.AdamW(modelT.parameters(), lr=lr)
    

        if useDynamicTrain == True:
            trainDS = terrainDatasetDynamic(df=trainDF, 
                                            chip_size=spat_dim, 
                                            cell_size=cell_size, 
                                            background_value=background_value, 
                                            transforms=myTransforms)
        else:
            trainDS = terrainDataset(trainDF, transform=myTransforms)

        if useDynamicVal == True:
            valDS = terrainDatasetDynamic(df=valDF, 
                                          chip_size=spat_dim, 
                                          cell_size=cell_size, 
                                          background_value=background_value, 
                                          transforms=None)
        else:
            valDS = terrainDataset(valDF, transform=None)

        trainDL = DataLoader(trainDS, batch_size=batchSize, shuffle=True, sampler=None,
            batch_sampler=None, num_workers=12, collate_fn=None,
            pin_memory=True, drop_last=True, timeout=0,
            worker_init_fn=None)
        
        valDL =  DataLoader(valDS, batch_size=batchSize, shuffle=False, sampler=None,
            batch_sampler=None, num_workers=12, collate_fn=None,
            pin_memory=True, drop_last=True, timeout=0,
            worker_init_fn=None)
    
        acc = tm.Accuracy(task="multiclass", num_classes=nCls, average="micro").to(device)
        f1 = tm.F1Score(task="multiclass", num_classes=nCls, average="macro").to(device)

        eNum = []
        t_loss = []
        t_acc = []
        t_f1 = []
        v_loss = []
        v_acc = []
        v_f1 = []

        f1VMax = 0.0

        # Loop over epochs
        for epoch in range(1, epochs+1):
            #Initiate running loss for epoch
            running_loss = 0.0
            #Make sure model is in training mode
            modelT.train()
            # Loop over training batches
            for batch_idx, (inputs, targets) in enumerate(trainDL):
                # Get data and move to device
                inputs, targets = inputs.to(device), targets.to(device)

                # Clear gradients
                optimizer.zero_grad()
                # Predict data

                outputs = modelT(inputs)
                # Calculate loss
                targets2 = targets.unsqueeze(1)
                targets2 = crop_tensor(targets2, cropFactor)
                loss = criterion(outputs, targets2)

                # Calculate metrics
                targets3 = targets2.squeeze()
                accT = acc(outputs, targets3)
                f1T = f1(outputs, targets3)
                
                # Backpropagate
                loss.backward()

                # Update parameters
                optimizer.step()
                #scheduler.step()

                #Update running with batch results
                running_loss += loss.item()

            # Accumulate loss and metrics at end of training epoch
            epoch_loss = running_loss/len(trainDL)
            accT = acc.compute()
            f1T = f1.compute()

            # Print Losses and metrics at end of each training epoch   
            print(f'Epoch: {epoch}, Training Loss: {epoch_loss:.4f}, Training Accuracy: {accT:.4f}, Training F1: {f1T:.4f}')

            # Append results
            eNum.append(epoch)
            t_loss.append(epoch_loss)
            t_acc.append(accT.detach().cpu().numpy())
            t_f1.append(f1T.detach().cpu().numpy())

            # Reset metrics
            acc.reset()
            f1.reset()

            #Make sure model is in eval mode
            modelT.eval()
            # loop over validation batches
            with torch.no_grad():
                #Initialize running validation loss
                running_loss_v = 0.0
                for batch_idx, (inputs, targets) in enumerate(valDL):
                    # Get data and move to device
                    inputs, targets = inputs.to(device), targets.to(device)


                    outputs = modelT(inputs)
                    # Calculate loss
                    targets2 = targets.unsqueeze(1)
                    targets2 = crop_tensor(targets2, cropFactor)
                    loss_v = criterion(outputs, targets2)

                    # Calculate metrics
                    targets3 = targets2.squeeze()
                    accV = acc(outputs, targets3)
                    f1V = f1(outputs, targets3)

                    #Update running with batch results
                    running_loss_v += loss_v.item()
                    
            #Accumulate loss and metrics at end of validation epoch
            epoch_loss_v = running_loss_v/len(valDL)
            accV = acc.compute()
            f1V = f1.compute()

            # Print validation loss and metrics
            print(f'Validation Loss: {epoch_loss_v:.4f}, Validation Accuracy: {accV:.4f}, Validation F1: {f1V:.4f}')

            # Append results
            v_loss.append(epoch_loss_v)
            v_acc.append(accV.detach().cpu().numpy())
            v_f1.append(f1V.detach().cpu().numpy())

            # Reset metrics
            acc.reset()
            f1.reset()
            
            # Save model if validation F1-score improves
            f1V2 = f1V.detach().cpu().numpy()
            if f1V2 > f1VMax:
                f1VMax = f1V2
                torch.save(modelT.state_dict(), saveFolder + 'model.pt')
                print(f'Model saved for epoch {epoch}.')
            
        SeNum = pd.Series(eNum, name="epoch")
        St_loss = pd.Series(t_loss, name="training_loss")
        St_acc = pd.Series(t_acc, name="training_accuracy")
        St_f1 = pd.Series(t_f1, name="training_f1")
        Sv_loss = pd.Series(v_loss, name="val_loss")
        Sv_acc = pd.Series(v_acc, name="val_accuracy")
        Sv_f1 = pd.Series(v_f1, name="val_f1")
        resultsDF = pd.concat([SeNum, St_loss, St_acc, St_f1, Sv_loss, Sv_acc, Sv_f1], axis=1)

        resultsDF.to_csv(saveFolder+"trainLog.csv")