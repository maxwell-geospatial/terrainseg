from typing import Any, Dict, Sequence, Literal, Tuple
import numpy as np
import torch
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from matplotlib.patches import Patch
from torchvision.utils import make_grid

def pd_like_class_table(cCodes, cNames, cColors):
    import pandas as pd
    if not (len(cCodes) == len(cNames) == len(cColors)):
        raise ValueError("cCodes, cNames, cColors must have the same length.")
    return pd.DataFrame(
        {"cCodes": list(cCodes), "cNames": list(cNames), "cColors": list(cColors)}
    )

def viewBatch(
    dataloader,
    ncols: int = 3,
    cCodes: Sequence[int] = (),
    cNames: Sequence[str] = (),
    cColors: Sequence[str] = (),
    padding: int = 10,
    figsize: Tuple[int, int] =(10, 10),
    mask_mode: Literal["auto", "index", "onehote"] = "auto", 
    stretch: str = "percentile",         
    p_low: float = 2.0,              
    p_high: float = 98.0,                 
) -> Dict[str, Any]:
    """
    View a mini-batch of DTM chips and associated masks to visually validate that chips are being generated as expected. 

    Parameters
    ----------
    dataloader 
        Instantiated torch dataloader created using `terrainDataset()` or `terrainDatasetDynamic()`.
    ncols : int, default=3
        Number of columns in image grids.
    cCode : Sequence[int]
        Integer codes assigned to each class. Must be the same length as the number of classes.
    cNames : Sequence[int]
        Class names associated with each class code and in the same order as the class codes.
    cColors: Sequence[int]
        Named Python colors or hex codes as strings that define colors used to differentiate the classes.
    padding : int, default=10
        Number of cells or pixels to add between each chip in the image array.
    figsize : Tuple[int, int], default=(10, 10)
        Size of image as a tuple of height and width values. 
    mask_mode : Literal["auto", "index", "onehot"], default="auto", 
        Defines how masks are represented (either "auto", "index", or "onehot"). terrainseg uses "index". Recommend leaving this set to the default value of "auto".
    stretch : str, default="percentile
        Type of stretch to apply (either "percentile", "minmax", or "max"). Recommend using the default. 
    p_low : float, default=2.0
        If stretch = "percentile", defines the lower percentile. 
    p_high : float, default=98.0
        If stretch = "percentile", defines the upper percentile to use. 

    Returns
    -------
        Matplotlib plot.

    """

    # ------------------------------------------------------------------
    # Fetch batch
    # ------------------------------------------------------------------
   
    r = 1
    g = 1
    b = 1
    eps = 1e-8
    per_channel = False

    batch = next(iter(dataloader))

    if isinstance(batch, dict):
        images = batch["image"]
        masks  = batch["mask"]
    else:
        images, masks = batch

    if not torch.is_tensor(images) or images.ndim != 4:
        raise ValueError("images must be torch.Tensor [B,C,H,W]")
    if not torch.is_tensor(masks):
        raise ValueError("masks must be torch.Tensor")

    # Ensure float images for NaN padding + scaling
    images = images.float()

    # ------------------------------------------------------------------
    # Normalize masks to class-index [B,H,W]
    # ------------------------------------------------------------------
    if masks.ndim == 3:
        masks_idx = masks.long()
    elif masks.ndim == 4:
        if mask_mode == "index":
            masks_idx = masks[:, 0].long()
        elif mask_mode == "onehot":
            masks_idx = masks.argmax(dim=1).long()
        else:  # auto
            masks_idx = masks.argmax(dim=1).long() if masks.shape[1] > 1 else masks[:, 0].long()
    else:
        raise ValueError("masks must be [B,H,W], [B,1,H,W], or [B,K,H,W]")

    masks_for_grid = masks_idx.unsqueeze(1).float()  # [B,1,H,W]

    # ------------------------------------------------------------------
    # Image grid (NaN padding)
    # ------------------------------------------------------------------
    img_grid_t = make_grid(
        images,
        nrow=ncols,
        padding=padding,
        pad_value=float("nan")
    )  # [C,H,W]

    img_grid = (
        img_grid_t
        .permute(1, 2, 0)
        .detach()
        .cpu()
        .numpy()
        .astype(np.float32)
    )  # [H,W,C]

    # ------------------------------------------------------------------
    # Contrast stretch -> [0,1]
    # ------------------------------------------------------------------
    def _stretch_minmax(arr: np.ndarray) -> Dict[str, Any]:
        finite = arr[np.isfinite(arr)]
        if finite.size == 0:
            raise ValueError("No finite image values found for stretching.")
        lo = float(finite.min())
        hi = float(finite.max())
        return {"lo": lo, "hi": hi}

    def _stretch_max(arr: np.ndarray) -> Dict[str, Any]:
        finite = arr[np.isfinite(arr)]
        if finite.size == 0:
            raise ValueError("No finite image values found for stretching.")
        lo = 0.0
        hi = float(finite.max())
        return {"lo": lo, "hi": hi}

    def _stretch_percentile(arr: np.ndarray, pl: float, ph: float) -> Dict[str, Any]:
        finite = arr[np.isfinite(arr)]
        if finite.size == 0:
            raise ValueError("No finite image values found for stretching.")
        lo = float(np.percentile(finite, pl))
        hi = float(np.percentile(finite, ph))
        return {"lo": lo, "hi": hi}

    stretch_info: Dict[str, Any] = {"mode": stretch, "per_channel": per_channel}

    if per_channel:
        # stretch each channel independently
        out = img_grid.copy()
        per_ch = []
        for c in range(out.shape[2]):
            ch = out[..., c]
            if stretch == "minmax":
                s = _stretch_minmax(ch)
            elif stretch == "max":
                s = _stretch_max(ch)
            elif stretch == "percentile":
                s = _stretch_percentile(ch, p_low, p_high)
            else:
                raise ValueError("stretch must be 'percentile', 'minmax', or 'max'.")

            lo, hi = s["lo"], s["hi"]
            # guard
            if not np.isfinite(lo) or not np.isfinite(hi) or (hi - lo) < eps:
                # if degenerate, leave channel as zeros (but keep NaNs)
                ch2 = ch.copy()
                ch2[np.isfinite(ch2)] = 0.0
            else:
                ch2 = np.clip(ch, lo, hi)
                ch2 = (ch2 - lo) / (hi - lo + eps)

            out[..., c] = ch2
            per_ch.append({"channel": c, **s})
        img_grid = out
        stretch_info["channels"] = per_ch

    else:
        # stretch using all channels together (keeps cross-channel comparability)
        if stretch == "minmax":
            s = _stretch_minmax(img_grid)
        elif stretch == "max":
            s = _stretch_max(img_grid)
        elif stretch == "percentile":
            s = _stretch_percentile(img_grid, p_low, p_high)
        else:
            raise ValueError("stretch must be 'percentile', 'minmax', or 'max'.")

        lo, hi = s["lo"], s["hi"]
        stretch_info.update(s)

        if not np.isfinite(lo) or not np.isfinite(hi) or (hi - lo) < eps:
            # degenerate case
            tmp = img_grid.copy()
            tmp[np.isfinite(tmp)] = 0.0
            img_grid = tmp
        else:
            img_grid = np.clip(img_grid, lo, hi)
            img_grid = (img_grid - lo) / (hi - lo + eps)

    # ------------------------------------------------------------------
    # RGB channel selection (1-based)
    # ------------------------------------------------------------------
    rr, gg, bb = r - 1, g - 1, b - 1
    if img_grid.shape[2] <= max(rr, gg, bb):
        raise ValueError(
            f"Requested RGB channels ({r},{g},{b}) but image has {img_grid.shape[2]} channels."
        )

    rgb = img_grid[..., [rr, gg, bb]]
    rgb_disp = np.ma.masked_invalid(rgb)  # NaN padding -> transparent

    # ------------------------------------------------------------------
    # Mask grid (NaN padding)
    # ------------------------------------------------------------------
    msk_grid_t = make_grid(
        masks_for_grid,
        nrow=ncols,
        padding=padding,
        pad_value=float("nan")
    )  # [1,H,W]

    msk_grid = (
        msk_grid_t[0]
        .detach()
        .cpu()
        .numpy()
        .astype(np.float32)
    )  # [H,W]

    # ------------------------------------------------------------------
    # Class table and colormap
    # ------------------------------------------------------------------
    used_codes = sorted({int(v) for v in np.unique(msk_grid[~np.isnan(msk_grid)])})

    if cCodes and cNames and cColors:
        cat = pd_like_class_table(cCodes, cNames, cColors)
        cat = cat[cat["cCodes"].isin(used_codes)].copy()

        code_to_idx = {int(code): i for i, code in enumerate(cat["cCodes"].tolist())}

        mask_idx = np.full(msk_grid.shape, -1, dtype=np.int32)
        valid = ~np.isnan(msk_grid)
        if valid.any():
            flat = msk_grid[valid].astype(np.int64)
            mask_idx[valid] = np.array([code_to_idx.get(int(v), -1) for v in flat], dtype=np.int32)

        mask_plot = np.ma.masked_where(mask_idx < 0, mask_idx)
        cmap = ListedColormap(cat["cColors"].tolist()) if len(cat) else ListedColormap([])
    else:
        mask_plot = np.ma.masked_invalid(msk_grid)
        cmap = "gray"
        cat = None

    # ------------------------------------------------------------------
    # Plot
    # ------------------------------------------------------------------
    fig, axes = plt.subplots(2, 1, figsize=figsize)
    ax_img, ax_msk = axes

    ax_img.imshow(rgb_disp)
    ax_img.set_axis_off()
    ax_img.set_title(f"Images (stretched to [0,1], mode={stretch})")

    ax_msk.imshow(mask_plot, cmap=cmap, interpolation="nearest")
    ax_msk.set_axis_off()
    ax_msk.set_title("Masks")

    if cat is not None and len(cat):
        legend_handles = [
            Patch(facecolor=row.cColors, edgecolor="none", label=row.cNames)
            for row in cat.itertuples(index=False)
        ]
        ax_msk.legend(
            handles=legend_handles,
            loc="upper left",
            bbox_to_anchor=(1.02, 1.0),
            frameon=False,
        )

    plt.tight_layout()
    
