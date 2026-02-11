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

def _chip_bbox_from_point(
    pt: Point,
    chip_size: int,
    cell_size: float,
) -> box:
    """
    Create a square bbox centered on pt with side length chip_size*cell_size (map units).
    Equivalent to buffering point by (chip_size*cell_size)/2 then making a 1-cell grid box.
    """
    half = (chip_size * cell_size) / 2.0
    return box(pt.x - half, pt.y - half, pt.x + half, pt.y + half)


def _window_from_bounds_aligned(
    src: rio.io.DatasetReader,
    bounds: Tuple[float, float, float, float],
    chip_size: int,
) -> Tuple[Window, Affine]:
    """
    Convert bounds -> a rasterio Window.
    Ensures the output window is exactly (chip_size x chip_size) by shifting at edges.
    Returns (window, window_transform).
    """
    # Get float window covering bounds
    win = rio.windows.from_bounds(*bounds, transform=src.transform)

    # Use the upper-left of that window as a starting integer offset
    col_off = int(np.floor(win.col_off))
    row_off = int(np.floor(win.row_off))

    # Shift to guarantee chip_size and stay inside image
    col_off = max(0, min(col_off, src.width - chip_size))
    row_off = max(0, min(row_off, src.height - chip_size))

    window = Window(col_off=col_off, row_off=row_off, width=chip_size, height=chip_size)
    w_transform = src.transform * Affine.translation(window.col_off, window.row_off)
    return window, w_transform


def makeDynamicChip(
    chip_row: Union[pd.Series, gpd.GeoSeries, Dict[str, Any]],
    chip_size: int,
    cell_size: float,
    code_field: str = "code",
    background_value: int = 0,
    mask_dtype: str = "int32",
) -> Dict[str, Any]:
    """
    Generate a single dynamic image chip and rasterized mask on demand.

    Chip center geometry and all dataset paths (predictor raster and mask
    polygon features) are read directly from `chip_row`, which must be a row
    from the GeoDataFrame produced by `makeDynamicChipsGDF`.

    If the sampling geometry is a Polygon or MultiPolygon, its centroid is
    used as the chip center.

    Parameters
    ----------
    chip_row : pandas.Series or geopandas.GeoSeries or dict
        One row from a chip-center GeoDataFrame produced by
        `makeDynamicChipsGDF`. Must contain:

        - ``geometry`` : Point or Polygon defining the chip center
        - ``imgPth`` / ``imgName`` : predictor raster path
        - ``mask_featPth`` / ``mask_featName`` : mask polygon dataset path

    chip_size : int
        Width and height of the output chip in pixels.

    cell_size : float
        Spatial resolution of output chip in map units per pixel.

    code_field : str, default="code"
        Attribute field in the mask polygon dataset containing integer class
        codes to burn into the mask raster.

    background_value : int, default=0
        Pixel value assigned to areas not covered by any mask polygons.

    mask_dtype : str, default="int32"
        NumPy dtype used for the output mask array.

    Returns
    -------
    dict[str, Any]
        Dictionary containing:

        - ``"image"`` : ndarray
            Raster chip extracted from the source image.
        - ``"mask"`` : ndarray
            Rasterized segmentation mask aligned to the image chip.
        - ``"transform"`` : affine.Affine
            Affine transform for the chip.
        - ``"crs"`` : rasterio.crs.CRS
            Coordinate reference system of the chip.
        - ``"img_profile"`` : dict
            Rasterio profile copied from the source raster.
        - ``"window"`` : rasterio.windows.Window
            Window describing the chip’s location in the source raster.

    Raises
    ------
    ValueError
        If required fields are missing or CRS information is unavailable.
    """

    # --------------------
    # Access helpers
    # --------------------
    def getv(k: str):
        return chip_row[k]

    # --------------------
    # Chip center geometry
    # --------------------
    geom = getv("geometry")
    if geom is None or geom.is_empty:
        raise ValueError("chip_row geometry is missing or empty.")

    if geom.geom_type in ("Polygon", "MultiPolygon"):
        geom = geom.centroid
    elif geom.geom_type == "MultiPoint":
        geom = list(geom.geoms)[0]
    elif geom.geom_type != "Point":
        raise ValueError("chip center geometry must be Point or Polygon-like.")

    # --------------------
    # Paths
    # --------------------
    img_path = os.path.join(str(getv("imgPth")), str(getv("imgName")))
    mask_feat_path = os.path.join(str(getv("mask_featPth")), str(getv("mask_featName")))

    # --------------------
    # Read mask features
    # --------------------
    mask_gdf = gpd.read_file(mask_feat_path)

    with rio.open(img_path) as src:
        # CRS alignment
        if mask_gdf.crs is None:
            raise ValueError("Mask feature layer has no CRS; cannot align.")
        if src.crs is None:
            raise ValueError("Raster has no CRS; cannot align.")
        if mask_gdf.crs != src.crs:
            mask_gdf = mask_gdf.to_crs(src.crs)

        # --------------------
        # Chip window
        # --------------------
        chip_bounds = _chip_bbox_from_point(geom, chip_size, cell_size).bounds
        window, w_transform = _window_from_bounds_aligned(src, chip_bounds, chip_size)

        img = src.read(1,window=window)

        # --------------------
        # Intersect mask polygons
        # --------------------
        chip_poly = box(*chip_bounds)
        chip_poly_gdf = gpd.GeoDataFrame({"geometry": [chip_poly]}, crs=src.crs)

        f1 = mask_gdf[mask_gdf.intersects(chip_poly)]
        if len(f1) > 0:
            f1 = gpd.overlay(f1, chip_poly_gdf, how="intersection")

        # --------------------
        # Rasterize mask
        # --------------------
        out_shape = (chip_size, chip_size)
        if len(f1) == 0:
            msk = np.full(out_shape, background_value, dtype=np.dtype(mask_dtype))
        else:
            if code_field not in f1.columns:
                raise ValueError(f"'{code_field}' not found in mask feature attributes.")
            shapes = [
                (geom, int(val))
                for geom, val in zip(f1.geometry, f1[code_field])
                if geom is not None and not geom.is_empty
            ]
            msk = rasterize(
                shapes=shapes,
                out_shape=out_shape,
                transform=w_transform,
                fill=background_value,
                dtype=mask_dtype,
                all_touched=False,
            )

        return {
            "image": img,
            "mask": msk,
            "transform": w_transform,
            "crs": src.crs,
            "img_profile": src.profile,
            "window": window,
        }



        
def makeDynamicChipsGDF(
    center_featPth: str,
    center_featName: str,
    mask_featPth: str,
    mask_featName: str,
    extentPth: str,
    extentName: str,
    imgPth: str,
    imgName: str,
    extent_crop: float = 50.0,
    do_background: bool = False,
    background_cnt: int = 0,
    background_dist: float = 0.0,
    use_seed: bool = False,
    seed: int = 42,
    do_shuffle: bool = False,
) -> gpd.GeoDataFrame:
    """
    Generate a GeoDataFrame of dynamic chip sampling locations.

    This function builds a table of chip-center locations from a point or
    polygon feature dataset, associates them with predictor raster grids,
    and links them to a separate polygon dataset used for mask generation.

    Chip centers and mask polygons are explicitly separated:

      * Chip centers → define where chips are sampled
      * Mask polygons → define semantic labels rasterized into masks

    If chip centers are provided as polygons, centroids are computed and used
    as sampling locations.

    Parameters
    ----------
    center_featPth : str
        Directory containing chip-center feature dataset (Point or Polygon).

    center_featName : str
        Filename of chip-center feature dataset.

    mask_featPth : str
        Directory containing polygon features used to generate raster masks.

    mask_featName : str
        Filename of polygon feature dataset used for mask generation.

    extentPth : str
        Directory containing study-area extent polygon dataset.

    extentName : str
        Filename of extent polygon dataset.

    imgPth : str
        Directory containing predictor raster grid(s).

    imgName : str
        Filename of predictor raster grid used for chip extraction.

    extent_crop : float, default=50.0
        Buffer distance applied to the extent geometry (map units). Positive
        values expand the sampling area; negative values shrink it.

    do_background : bool, default=False
        If True, randomly generate background chip centers within the extent.

    background_cnt : int, default=0
        Number of background chip centers to generate.

    background_dist : float, default=0.0
        Minimum distance from the extent boundary for background points.

    use_seed : bool, default=False
        If True, use deterministic random sampling.

    seed : int, default=42
        Random seed used when `use_seed=True`.

    do_shuffle : bool, default=False
        If True, randomly shuffle output rows before assigning chip IDs.

    Returns
    -------
    geopandas.GeoDataFrame
        GeoDataFrame containing:

        - ``chipID`` : unique integer identifier
        - ``geometry`` : Point geometries defining chip centers
        - ``imgPth`` / ``imgName`` : predictor raster metadata
        - ``center_featPth`` / ``center_featName`` : chip-center dataset metadata
        - ``mask_featPth`` / ``mask_featName`` : mask polygon dataset metadata
        - ``extentPth`` / ``extentName`` : extent dataset metadata

    Raises
    ------
    ValueError
        If any dataset lacks a CRS.
    ValueError
        If chip-center geometries are not Point/MultiPoint or
        Polygon/MultiPolygon.

    Notes
    -----
    This function produces a sampling table only — raster chips and masks
    are generated dynamically by `makeDynamicChip`.

    This design allows:
      * Multiple label schemas per sampling grid
      * Easy swapping of predictor rasters
      * Clean separation of sampling geometry and supervision geometry
    """

    # --------------------
    # Read inputs
    # --------------------
    centers = gpd.read_file(os.path.join(center_featPth, center_featName))
    masks = gpd.read_file(os.path.join(mask_featPth, mask_featName))
    extent = gpd.read_file(os.path.join(extentPth, extentName))

    # --------------------
    # CRS validation
    # --------------------
    if centers.crs is None:
        raise ValueError("Chip-center feature dataset is missing a CRS.")
    if masks.crs is None:
        raise ValueError("Mask feature dataset is missing a CRS.")
    if extent.crs is None:
        raise ValueError("Extent dataset is missing a CRS.")

    if masks.crs != centers.crs:
        masks = masks.to_crs(centers.crs)
    if extent.crs != centers.crs:
        extent = extent.to_crs(centers.crs)

    # --------------------
    # Geometry type checks
    # --------------------
    geom_types = set(centers.geometry.geom_type.dropna().unique())
    is_pointlike = geom_types.issubset({"Point", "MultiPoint"})
    is_polygonlike = geom_types.issubset({"Polygon", "MultiPolygon"})

    if not (is_pointlike or is_polygonlike):
        raise ValueError(
            f"Unsupported/mixed chip-center geometry types: {sorted(geom_types)}. "
            "Expected Point/MultiPoint or Polygon/MultiPolygon."
        )

    cent = centers[["geometry"]].copy()

    # Polygons → centroids
    if is_polygonlike:
        cent["geometry"] = cent.geometry.centroid

    # --------------------
    # Extent crop
    # --------------------
    bnd = extent.buffer(extent_crop) if extent_crop != 0 else extent.geometry
    bnd_gdf = gpd.GeoDataFrame({"geometry": bnd}, crs=cent.crs)

    cent = (
        gpd.sjoin(cent, bnd_gdf, predicate="within", how="inner")
        .drop(columns=["index_right"], errors="ignore")
        .reset_index(drop=True)
    )

    # --------------------
    # Background sampling
    # --------------------
    if do_background and background_cnt > 0:
        rng = np.random.default_rng(seed if use_seed else None)

        poly = bnd_gdf.unary_union
        safe_poly = poly.buffer(-background_dist) if background_dist != 0 else poly
        if safe_poly.is_empty:
            safe_poly = poly

        pts = []
        tries = max(background_cnt * 20, 200)
        minx, miny, maxx, maxy = safe_poly.bounds
        for _ in range(tries):
            x = rng.uniform(minx, maxx)
            y = rng.uniform(miny, maxy)
            p = Point(x, y)
            if safe_poly.contains(p):
                pts.append(p)
            if len(pts) >= background_cnt:
                break

        bg = gpd.GeoDataFrame({"geometry": pts}, crs=cent.crs)
        cent = pd.concat([cent, bg], ignore_index=True)
        cent = gpd.GeoDataFrame(cent, crs=cent.crs)

    # --------------------
    # Attach metadata
    # --------------------
    cent["imgPth"] = imgPth
    cent["imgName"] = imgName
    cent["center_featPth"] = center_featPth
    cent["center_featName"] = center_featName
    cent["mask_featPth"] = mask_featPth
    cent["mask_featName"] = mask_featName
    cent["extentPth"] = extentPth
    cent["extentName"] = extentName

    # --------------------
    # chipID
    # --------------------
    if do_shuffle and len(cent) > 0:
        cent = cent.sample(frac=1.0, replace=False,
                           random_state=(seed if use_seed else None)).reset_index(drop=True)
    cent.insert(0, "chipID", np.arange(1, len(cent) + 1))

    return cent
  

def checkDynamicChips(
    chips_gdf: gpd.GeoDataFrame,
    chip_size: int = 512,
    cell_size: float = 1.0,
    nodata_ok: bool = False,
) -> gpd.GeoDataFrame:
    """
    Validate dynamically generated raster chips and append quality metrics.

    For each sampling location in `chips_gdf`, this function generates a
    dynamic image chip and segmentation mask using `makeDynamicChip`, then
    inspects them for missing values, nodata pixels, and dimensional
    consistency. The resulting quality metrics are appended to the input
    GeoDataFrame and returned.

    All dataset paths (predictor raster and mask polygon layers) are read
    directly from `chips_gdf`, which must be produced by
    `makeDynamicChipsGDF`.

    Parameters
    ----------
    chips_gdf : geopandas.GeoDataFrame
        Table of sampling locations produced by `makeDynamicChipsGDF`.
        Must contain ``imgPth``, ``imgName``, ``mask_featPth``,
        and ``mask_featName``.

    chip_size : int, default=512
        Width and height of generated chips in pixels.

    cell_size : float, default=1.0
        Spatial resolution of generated chips (map units per pixel).

    nodata_ok : bool, default=False
        If False, nodata values from the source raster are counted as invalid.
        If True, nodata values are ignored in validation.

    Returns
    -------
    geopandas.GeoDataFrame
        Copy of `chips_gdf` with additional columns:

        - ``NoData`` : "Yes"/"No" flag indicating presence of nodata pixels
        - ``cCntImg`` : image width in pixels
        - ``rCntImg`` : image height in pixels
        - ``naCntImg`` : number of NaN or nodata pixels in image
        - ``cCntMsk`` : mask width in pixels
        - ``rCntMsk`` : mask height in pixels
        - ``naCntMsk`` : number of NaN pixels in mask

    Notes
    -----
    This function does not modify chip geometry — it only evaluates the
    raster content generated dynamically.
    """
    rows = []

    for idx in range(len(chips_gdf)):
        out = makeDynamicChip(
            chip_row=chips_gdf.iloc[idx],
            chip_size=chip_size,
            cell_size=cell_size,
        )

        img = out["image"]
        msk = out["mask"]

        # NaN checks
        na_img = int(np.isnan(img).sum()) if np.issubdtype(img.dtype, np.floating) else 0
        na_msk = int(np.isnan(msk).sum()) if np.issubdtype(msk.dtype, np.floating) else 0

        minimum_value = img.min()
        hasNoData = "Yes" if minimum_value < 0 else "No"

        # nodata checks
        nd_img_cnt = 0
        nd_msk_cnt = 0
        if not nodata_ok:
            row = chips_gdf.iloc[idx]
            with rio.open(os.path.join(str(row["imgPth"]), str(row["imgName"]))) as src:
                if src.nodata is not None:
                    nd_img_cnt = int((img == src.nodata).sum())
            # dynamic masks use fill value → leave nd_msk_cnt = 0

        rows.append(
            {
                "NoData": str(hasNoData),
                "cCntImg": int(img.shape[-1]),
                "rCntImg": int(img.shape[-2]),
                "naCntImg": int(na_img + nd_img_cnt),
                "cCntMsk": int(msk.shape[1]),
                "rCntMsk": int(msk.shape[0]),
                "naCntMsk": int(na_msk + nd_msk_cnt),
            }
        )

    check_df = pd.DataFrame(rows)
    out_gdf = chips_gdf.reset_index(drop=True).join(check_df)
    out_gdf = out_gdf.query("NoData == 'No' and naCntImg == 0 and naCntMsk == 0 and cCntImg == 640 and rCntImg == 640 and cCntMsk == 640 and rCntMsk == 640")
    return out_gdf







def saveDynamicChips(
    out: Dict[str, Any],
    out_image_path: str,
    out_mask_path: str,
    n_channels: Optional[int] = None,
    mask_dtype: Optional[str] = None,
) -> None:
    """
    Save a dynamically generated chip (image and mask) to GeoTIFF files.

    The function writes the raster image and corresponding segmentation mask
    returned by `makeDynamicChip` (or similar functions) to disk while
    preserving spatial metadata (transform and CRS). The image is written as
    a multi-band GeoTIFF and the mask as a single-band GeoTIFF.

    Parameters
    ----------
    out : dict[str, Any]
        Dictionary produced by a dynamic chip generator containing:

        - ``"image"`` : ndarray of shape (C, H, W)
        - ``"mask"`` : ndarray of shape (H, W)
        - ``"transform"`` : affine transform
        - ``"crs"`` : coordinate reference system
        - ``"img_profile"`` : rasterio profile template

    out_image_path : str
        Output file path for the image GeoTIFF.
    out_mask_path : str
        Output file path for the mask GeoTIFF.
    n_channels : int or None, default=None
        Number of image bands to write. If None, all channels are written.
        Useful when the chip contains derived layers but only a subset
        should be exported.
    mask_dtype : str or None, default=None
        Data type used when writing the mask. If None, the mask's existing
        dtype is used (commonly ``uint8`` or ``int16``).

    Returns
    -------
    None
        Files are written to disk.

    Notes
    -----
    - Spatial metadata (CRS and affine transform) is preserved.
    - The mask is always written as a single-band raster.
    - Output directories are created automatically if they do not exist.
    - Intended for debugging, dataset export, or visualization of
      dynamically generated training samples.
    """

    img = out["image"]
    msk = out["mask"]
    transform = out["transform"]
    crs = out["crs"]

    profile = out["img_profile"].copy()
    if n_channels is None:
        n_channels = img.shape[0]
    profile.update(
        driver="GTiff",
        height=img.shape[1],
        width=img.shape[2],
        count=n_channels,
        transform=transform,
        crs=crs,
    )

    os.makedirs(os.path.dirname(out_image_path) or ".", exist_ok=True)
    with rio.open(out_image_path, "w", **profile) as dst:
        dst.write(img[:n_channels])

    m_profile = profile.copy()
    m_profile.update(count=1, dtype=(mask_dtype or str(msk.dtype)))
    os.makedirs(os.path.dirname(out_mask_path) or ".", exist_ok=True)
    with rio.open(out_mask_path, "w", **m_profile) as dst:
        dst.write(msk.astype(m_profile["dtype"]), 1)

