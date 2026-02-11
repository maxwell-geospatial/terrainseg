from __future__ import annotations

import os
from typing import Literal, Union, Optional, Tuple

from pathlib import Path

import numpy as np
import rasterio
from rasterio.windows import Window
from rasterio.transform import Affine
from rasterio.features import rasterize
from rasterio.mask import mask as rio_mask

from pathlib import Path

import pandas as pd
import geopandas as gpd

ModeMask = Literal["Both", "Mask"]

def makeMasks(
    image: str,
    features: str,
    crop: bool = False,
    extent: Optional[str] = None,
    field: Optional[str] = None,
    background: Union[int, float] = 0,
    out_image: Optional[str] = None,
    out_mask: Optional[str] = None,
    mode: ModeMask = "Both",
    all_touched: bool = False,
    dtype: str = "uint8",
) -> None:
    """
    Rasterize vector features to a mask aligned to an existing raster grid, optionally cropping the raster first.

    Parameters
    ----------
    image : str
        Path to reference raster (defines grid, resolution, transform, CRS).
    features : str
        Path to vector data (e.g., .shp, .gpkg).
    crop : bool
        If True, crop the raster to `extent` geometry before rasterizing.
    extent : Optional[str]
        Path to polygon vector defining crop extent (required if crop=True).
    field : Optional[str]
        Attribute field name to burn into raster. If None, burns 1s.
    background : int|float, default=0.
        Background value for mask where no features exist. Default is 0. 
    out_image : Optional[str]
        Output path for (cropped) image. Required if mode="Both".
    out_mask : Optional[str]
        Output path for mask. Required if mode in {"Both","Mask"}.
    mode : {"Both","Mask"}, default="Both"
        Write both raster + mask, or only mask.
    all_touched : bool, default=False
        If True, burn all pixels touched by geometries (rasterize option).
    dtype : str, default="uint8"
        Output dtype for mask (e.g., "uint8", "int16", "float32").

    Notes
    -----
    - Uses rasterio.mask.mask for cropping (masking by polygons).
    - Uses rasterio.features.rasterize aligned to the (possibly cropped) raster grid.
    """

    if mode not in ("Both", "Mask"):
        raise ValueError("Invalid mode. Use mode='Both' or mode='Mask'.")

    if mode == "Both" and not out_image:
        raise ValueError("out_image is required when mode='Both'.")

    if mode in ("Both", "Mask") and not out_mask:
        raise ValueError("out_mask is required when mode is 'Both' or 'Mask'.")

    if crop and not extent:
        raise ValueError("extent is required when crop=True (path to polygon layer).")

    # --- Read vectors
    gdf_feat = gpd.read_file(features)

    with rasterio.open(image) as src:
        raster_crs = src.crs
        if raster_crs is None:
            raise ValueError("Reference raster has no CRS; cannot align/reproject vectors.")

        # Reproject features to raster CRS if needed
        if gdf_feat.crs is None:
            raise ValueError("Features layer has no CRS; cannot reproject.")
        if gdf_feat.crs != raster_crs:
            gdf_feat = gdf_feat.to_crs(raster_crs)

        # Optionally crop raster first (and update transform/shape/profile)
        if crop:
            gdf_ext = gpd.read_file(extent)
            if gdf_ext.crs is None:
                raise ValueError("Extent layer has no CRS; cannot reproject.")
            if gdf_ext.crs != raster_crs:
                gdf_ext = gdf_ext.to_crs(raster_crs)

            # rasterio.mask expects GeoJSON-like mapping geometries
            ext_geoms = [geom.__geo_interface__ for geom in gdf_ext.geometry if geom is not None]
            if not ext_geoms:
                raise ValueError("Extent layer contains no valid geometries.")

            # Crop all bands, keep pixels inside extent (like terra::crop)
            img_data, out_transform = rio_mask(
                src, ext_geoms, crop=True, filled=True
            )
            out_profile = src.profile.copy()
            out_profile.update(
                {
                    "height": img_data.shape[1],
                    "width": img_data.shape[2],
                    "transform": out_transform,
                }
            )
        else:
            img_data = src.read()  # (bands, rows, cols)
            out_transform = src.transform
            out_profile = src.profile.copy()

    # --- Prepare shapes/values for rasterize
    # Burn value for each geometry: either from attribute field or constant 1
    if field is None:
        shapes = [(geom, 1) for geom in gdf_feat.geometry if geom is not None]
    else:
        if field not in gdf_feat.columns:
            raise ValueError(f"Field '{field}' not found in features columns: {list(gdf_feat.columns)}")
        shapes = [
            (geom, val)
            for geom, val in zip(gdf_feat.geometry, gdf_feat[field])
            if geom is not None
        ]

    if not shapes:
        raise ValueError("Features layer contains no valid geometries to rasterize.")

    height = out_profile["height"]
    width = out_profile["width"]

    mask_arr = rasterize(
        shapes=shapes,
        out_shape=(height, width),
        transform=out_transform,
        fill=background,
        all_touched=all_touched,
        dtype=dtype,
    )

    # --- Write outputs
    if mode == "Both":
        os.makedirs(os.path.dirname(out_image) or ".", exist_ok=True)
        with rasterio.open(out_image, "w", **out_profile) as dst:
            dst.write(img_data)

    os.makedirs(os.path.dirname(out_mask) or ".", exist_ok=True)
    mask_profile = out_profile.copy()
    mask_profile.update(
        {
            "count": 1,
            "dtype": np.dtype(dtype).name,
            # Optional but often helpful:
            "nodata": None,
        }
    )
    with rasterio.open(out_mask, "w", **mask_profile) as dst:
        dst.write(mask_arr, 1)


def _ensure_dirs(out_dir: str, paths: Tuple[str, ...], use_existing_dir: bool) -> None:
    if use_existing_dir:
        # still ensure they exist
        for p in paths:
            os.makedirs(p, exist_ok=True)
        return
    for p in paths:
        os.makedirs(p, exist_ok=True)


def _chip_window(col0_1based: int, row0_1based: int, size: int, width: int, height: int) -> Tuple[int, int]:
    """
    Replicate your R edge logic:
    - proposed chip UL at (c1,r1) (1-based)
    - if chip exceeds right edge, shift left so last col aligns
    - if chip exceeds bottom edge, shift up so last row aligns
    Return (col_off_0based, row_off_0based) for rasterio Window.
    """
    c1 = col0_1based
    r1 = row0_1based
    c2 = c1 + (size - 1)
    r2 = r1 + (size - 1)

    if c2 <= width and r2 <= height:
        c1b, r1b = c1, r1
    elif c2 > width and r2 <= height:
        c1b, r1b = width - (size - 1), r1
    elif c2 <= width and r2 > height:
        c1b, r1b = c1, height - (size - 1)
    else:
        c1b, r1b = width - (size - 1), height - (size - 1)

    # Convert 1-based to 0-based offsets
    return (c1b - 1, r1b - 1)


def _window_transform(src_transform: Affine, window: Window) -> Affine:
    # rasterio helper exists, but this avoids extra import
    return src_transform * Affine.translation(window.col_off, window.row_off)


def makeChips(
    image: str,
    mask: str,
    n_channels: int = 1,
    size: int = 256,
    stride_x: int = 256,
    stride_y: int = 256,
    out_dir: str = ".",
    mode: Literal["All", "Positive", "Divided"] = "All",
    use_existing_dir: bool = False,
) -> None:
    """
    This function generates image and mask chips from an input image and associated raster mask. 

    The chips are written into the defined directory. The number of rows and columns of pixels in each chip are equal to the size argument. If a stride_x and/or stride_y is used that is different from the size argument, resulting chips will either overlap or have gaps between them. In order to not have overlap or gaps, the stride_x and stride_y arguments should be the same as the size argument. Both the image chips and associated masks are written to TIFF format (".tif"). Input data are not limited to three band images. This function is specifically for a binary classification where the positive case is indicated with a cell value of 1 and the background or negative case is indicated with a cell value of 0. If an irregular shaped raster grid is provided, only chips and masks that contain no NA or NoDATA cells will be produced. Three modes are available. If "All" is used, all image chips are generated even if they do not contain pixels mapped to the positive case. Within the provided directory, image chips will be written to an "images" folder and masks will be written to a "masks" folder. If "Positive" is used, only chips that have at least 1 pixel mapped to the positive class will be produced. Background-only chips will not be generated. Within the provided directory, image chips will be written to an "images" folder and masks will be written to a "masks" folder. Lastly, if the "Divided" method is used, separate "positive" and "background" folders will be created with "images" and "masks" subfolders. Any chip that has at least 1 pixel mapped to the positive class will be written to the "positive" folder while any chip having only background pixels will be written to the "background" folder.

    Parameters
    ----------
    image : str
        Path to image file or DTM on disk. 
    mask : str
        Path to raster mask on disk. Must align with image and have the same number of rows and columns of cells, cell size, and CRS.
    n_channels : int, default=1
        Number of input channels. Default is 1 since DTMs are single channel.
    size : int, default=256
        Size of image chips in height and width directions. We recommend 128, 256, or 512. 
    stride_x : int, default=256
        Stride in the x or row direction when making chips. If stride_x is smaller than size, generated chips will overlap in x direction. If stride_x is larger than size, there will be gaps between chips in x direction. If stride_x is the same as size, image will be partitioned into non-overlapping chips with no gaps in x direction. 
    stride_y : int, default=256
        Stride in the y or column direction when making chips. If stride_y is smaller than size, generated chips will overlap in y direction. If stride_y is larger than size, there will be gaps between chips in the y direction. If stride_y is the same as size, image will be partitioned into non-overlapping chips with no gaps in the y direction. 
    out_dir : str
        Location at which to store chips. Must include a trailing path separator.
    mode : Literal["All", "Positive", "Divided"], default="All"
        Either "All", "Positive", or "Divided". See notes. 
    use_existing_dir : bool, default=False
        Either True or False. If True, uses an existing directory that already contains chips and has an established folder structure.
    """

    mask_band = 1
    nodata_ok = False

    base = os.path.splitext(os.path.basename(image))[0]

    # output folders
    if mode == "Divided":
        img_pos = os.path.join(out_dir, "images", "positive")
        img_bkg = os.path.join(out_dir, "images", "background")
        msk_pos = os.path.join(out_dir, "masks", "positive")
        msk_bkg = os.path.join(out_dir, "masks", "background")
        _ensure_dirs(out_dir, (img_pos, img_bkg, msk_pos, msk_bkg), use_existing_dir)
    else:
        img_dir = os.path.join(out_dir, "images")
        msk_dir = os.path.join(out_dir, "masks")
        _ensure_dirs(out_dir, (img_dir, msk_dir), use_existing_dir)

    with rasterio.open(image) as src_img, rasterio.open(mask) as src_msk:
        if (src_img.width != src_msk.width) or (src_img.height != src_msk.height):
            raise ValueError("Image and mask dimensions do not match.")
        if src_img.transform != src_msk.transform:
            # You can relax this if you want, but your terra code assumes aligned grids
            raise ValueError("Image and mask transforms differ (not aligned).")
        if src_img.crs != src_msk.crs:
            raise ValueError("Image and mask CRS differ.")

        width, height = src_img.width, src_img.height

        across = int(np.ceil(width / stride_x))
        down = int(np.ceil(height / stride_y))
        across_seq2_1based = [(k * stride_x) + 1 for k in range(across)]
        down_seq2_1based = [(k * stride_y) + 1 for k in range(down)]

        # profiles for writing
        img_profile = src_img.profile.copy()
        msk_profile = src_msk.profile.copy()

        # ensure we write exactly size x size
        img_profile.update(width=size, height=size, count=n_channels)
        msk_profile.update(width=size, height=size, count=1)

        for c1 in across_seq2_1based:
            for r1 in down_seq2_1based:
                col_off, row_off = _chip_window(c1, r1, size, width, height)
                window = Window(col_off=col_off, row_off=row_off, width=size, height=size)

                chip = src_img.read(indexes=list(range(1, n_channels + 1)), window=window)  # (C,H,W)
                msk = src_msk.read(indexes=mask_band, window=window)                         # (H,W)

                # invalid-data checks (similar to NA checks)
                if not nodata_ok:
                    img_nd = src_img.nodata
                    msk_nd = src_msk.nodata
                    if img_nd is not None and np.any(chip == img_nd):
                        continue
                    if msk_nd is not None and np.any(msk == msk_nd):
                        continue

                # NaN checks (for float rasters)
                if np.issubdtype(chip.dtype, np.floating) and np.isnan(chip).any():
                    continue
                if np.issubdtype(msk.dtype, np.floating) and np.isnan(msk).any():
                    continue

                # write logic by mode
                is_positive = float(np.max(msk)) > 0.0

                # Match your filename pattern (uses the ORIGINAL c1,r1, not shifted ones)
                out_name = f"{base}_{c1}_{r1}.tif"

                # Update transform for this window
                chip_transform = _window_transform(src_img.transform, window)
                img_profile.update(transform=chip_transform)
                msk_profile.update(transform=chip_transform)

                if mode == "All":
                    out_img = os.path.join(out_dir, "images", out_name)
                    out_msk = os.path.join(out_dir, "masks", out_name)
                    with rasterio.open(out_img, "w", **img_profile) as dst:
                        dst.write(chip)
                    with rasterio.open(out_msk, "w", **msk_profile) as dst:
                        dst.write(msk, 1)

                elif mode == "Positive":
                    if not is_positive:
                        continue
                    out_img = os.path.join(out_dir, "images", out_name)
                    out_msk = os.path.join(out_dir, "masks", out_name)
                    with rasterio.open(out_img, "w", **img_profile) as dst:
                        dst.write(chip)
                    with rasterio.open(out_msk, "w", **msk_profile) as dst:
                        dst.write(msk, 1)

                elif mode == "Divided":
                    if is_positive:
                        out_img = os.path.join(out_dir, "images", "positive", out_name)
                        out_msk = os.path.join(out_dir, "masks", "positive", out_name)
                    else:
                        out_img = os.path.join(out_dir, "images", "background", out_name)
                        out_msk = os.path.join(out_dir, "masks", "background", out_name)

                    with rasterio.open(out_img, "w", **img_profile) as dst:
                        dst.write(chip)
                    with rasterio.open(out_msk, "w", **msk_profile) as dst:
                        dst.write(msk, 1)

                else:
                    raise ValueError("Invalid mode. Use 'All', 'Positive', or 'Divided'.")


def makeChipsMulticlass(
    image: str,
    mask: str,
    n_channels: int = 3,
    size: int = 256,
    stride_x: int = 256,
    stride_y: int = 256,
    out_dir: str = ".",
    use_existing_dir: bool = False,

) -> None:
    """
    Generates image and mask chips from an input image and associated raster mask. 

    The chips will be written into the defined directory. The number of rows and columns of pixels per chip are equal to the size argument. If a stride_x and/or stride_y is used that is different from the size argument, resulting chips will either overlap or have gaps between them. In order to not have overlap or gaps, the stride_x and stride_y arguments should be the same as the size argument. Both the image chips and associated masks are written to TIFF format (".tif"). Input data are not limited to three band images. This function is specifically for a multiclass classification. For a binary classification or when only two classes are differentiated, use the makeChips() function. If an irregular shaped raster grid is provided, only chips and masks that contain no NA or NoDATA cells will be produced. Within the provided directory, image chips will be written to an "images" folder and masks will be written to a "masks" folder.

    Parameters
    ----------
    image : str
        Path to image file or DTM on disk. 
    mask : str
        Path to raster mask on disk. Must align with image and have the same number of rows and columns of cells, cell size, and CRS.
    n_channels : int, default=1
        Number of input channels. Default is 1 since DTMs are single channel.
    size : int, default=256
        Size of image chips in height and width directions. We recommend 128, 256, or 512. 
    stride_x : int, default=256
        Stride in the x or row direction when making chips. If stride_x is smaller than size, generated chips will overlap in x direction. If stride_x is larger than size, there will be gaps between chips in x direction. If stride_x is the same as size, image will be partitioned into non-overlapping chips with no gaps in x direction. 
    stride_y : int, default=256
        Stride in the y or column direction when making chips. If stride_y is smaller than size, generated chips will overlap in y direction. If stride_y is larger than size, there will be gaps between chips in the y direction. If stride_y is the same as size, image will be partitioned into non-overlapping chips with no gaps in the y direction. 
    out_dir : str
        Location at which to store chips. Must include final slash in folder path. 
    use_existing_dir : bool, default=False
        Either True or False. If True, uses an existing directory that already contains chips and has an established folder structure.
    """

    mask_band = 1
    nodata_ok  = False
    positive_if_max_gt_zero = True

    base = os.path.splitext(os.path.basename(image))[0]

    img_dir = os.path.join(out_dir, "images")
    msk_dir = os.path.join(out_dir, "masks")
    _ensure_dirs(out_dir, (img_dir, msk_dir), use_existing_dir)

    with rasterio.open(image) as src_img, rasterio.open(mask) as src_msk:
        if (src_img.width != src_msk.width) or (src_img.height != src_msk.height):
            raise ValueError("Image and mask dimensions do not match.")
        if src_img.transform != src_msk.transform:
            raise ValueError("Image and mask transforms differ (not aligned).")
        if src_img.crs != src_msk.crs:
            raise ValueError("Image and mask CRS differ.")

        width, height = src_img.width, src_img.height

        across = int(np.ceil(width / stride_x))
        down = int(np.ceil(height / stride_y))
        across_seq2_1based = [(k * stride_x) + 1 for k in range(across)]
        down_seq2_1based = [(k * stride_y) + 1 for k in range(down)]

        img_profile = src_img.profile.copy()
        msk_profile = src_msk.profile.copy()
        img_profile.update(width=size, height=size, count=n_channels)
        msk_profile.update(width=size, height=size, count=1)

        for c1 in across_seq2_1based:
            for r1 in down_seq2_1based:
                col_off, row_off = _chip_window(c1, r1, size, width, height)
                window = Window(col_off=col_off, row_off=row_off, width=size, height=size)

                chip = src_img.read(indexes=list(range(1, n_channels + 1)), window=window)
                msk = src_msk.read(indexes=mask_band, window=window)

                if not nodata_ok:
                    img_nd = src_img.nodata
                    msk_nd = src_msk.nodata
                    if img_nd is not None and np.any(chip == img_nd):
                        continue
                    if msk_nd is not None and np.any(msk == msk_nd):
                        continue

                if np.issubdtype(chip.dtype, np.floating) and np.isnan(chip).any():
                    continue
                if np.issubdtype(msk.dtype, np.floating) and np.isnan(msk).any():
                    continue

                if positive_if_max_gt_zero:
                    if float(np.max(msk)) <= 0.0:
                        continue
                else:
                    # Customize for your labeling scheme if needed.
                    # Example: keep chips that contain ANY class != background_value
                    background_value = 0
                    if not np.any(msk != background_value):
                        continue

                out_name = f"{base}_{c1}_{r1}.tif"
                chip_transform = _window_transform(src_img.transform, window)
                img_profile.update(transform=chip_transform)
                msk_profile.update(transform=chip_transform)

                out_img = os.path.join(out_dir, "images", out_name)
                out_msk = os.path.join(out_dir, "masks", out_name)

                with rasterio.open(out_img, "w", **img_profile) as dst:
                    dst.write(chip)
                with rasterio.open(out_msk, "w", **msk_profile) as dst:
                    dst.write(msk, 1)

def makeChipsDF(
    folder: Union[str, Path],
    out_csv: Optional[Union[str, Path]] = None,
    extension: str = ".tif",
    mode: Literal["All", "Positive", "Divided"] = "All",
    shuffle: bool = False,
    save_csv: bool = False,
    seed: Optional[int] = None,
) -> pd.DataFrame:
    """
    This function creates a data frame and, optionally, a CSV file that lists all of the image chips and associated masks in a directory. 
    
    Three columns are produced. The chpN column provides the name of the chip, the chpPth column provides the path to the chip, and the chpMsk column provides the path to the associated mask. All paths are relative to the input folder as opposed to the full file path so that the results can still be used if the data are copied to a new location on disk or to a new computer.

    Parameters
    ----------
    folder : Union[str, Path]
        Folder location of chips. Must include final slash in folder path.
    out_csv : Optional[Union[str, Path]]
        If a CSV will be saved on disk, path and name of csv file with .csv extension. 
    extension : str 
        Extension used for chips and mask data. Default is ".tif".
    mode : Literal["All", "Positive", "Divided"], default="All" 
        Mode used to generate chips. Must match the setting used in makeChips. If makeChipsMulticlass was used, mode should be "All".
    shuffle : bool, default=False
        Whether or not to shuffle the chips. 
    save_csv : bool, Default=False
        Whether or not to save a CSV file. If True, out_csv must be defined. 
    seed : int
        Random seed to make shuffling reproducible. 

    Returns
    -------
    pd.DataFrame
        Output Pandas data frame where one row provides information for one chip.
    
    """
    folder = Path(folder).resolve()

    # normalize extension handling
    if not extension.startswith("."):
        extension = "." + extension
    ext = extension.lower()

    def list_files(rel_dir: str):
        p = folder / rel_dir
        if not p.exists():
            return []
        return sorted(
            [f.name for f in p.iterdir() if f.is_file() and f.suffix.lower() == ext]
        )

    if mode in ("All", "Positive"):
        chips = list_files("images")
        df = pd.DataFrame(
            {
                "chpN": chips,
                "chpPth": [
                    str((folder / "images" / fn).resolve()) for fn in chips
                ],
                "mskPth": [
                    str((folder / "masks" / fn).resolve()) for fn in chips
                ],
            }
        )

    elif mode == "Divided":
        chips_b = list_files("images/background")
        chips_p = list_files("images/positive")

        df_b = pd.DataFrame(
            {
                "chpN": chips_b,
                "chpPth": [
                    str((folder / "images" / "background" / fn).resolve())
                    for fn in chips_b
                ],
                "mskPth": [
                    str((folder / "masks" / "background" / fn).resolve())
                    for fn in chips_b
                ],
                "division": ["Background"] * len(chips_b),
            }
        )

        df_p = pd.DataFrame(
            {
                "chpN": chips_p,
                "chpPth": [
                    str((folder / "images" / "positive" / fn).resolve())
                    for fn in chips_p
                ],
                "mskPth": [
                    str((folder / "masks" / "positive" / fn).resolve())
                    for fn in chips_p
                ],
                "division": ["Positive"] * len(chips_p),
            }
        )

        df = pd.concat([df_b, df_p], ignore_index=True)

    else:
        raise ValueError("Invalid mode. Use 'All', 'Positive', or 'Divided'.")

    if shuffle and len(df) > 0:
        df = df.sample(frac=1.0, replace=False, random_state=seed).reset_index(drop=True)

    if save_csv:
        if out_csv is None:
            raise ValueError("out_csv must be provided when save_csv=True.")
        out_csv = Path(out_csv)
        out_csv.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(out_csv, index=False)

    return df
