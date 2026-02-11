import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import rasterio as rio


def terrainPredict(image_in: str,
                   pred_out: str,
                   model: nn.Module,
                   chip_size: int = 640,
                   stride_x: int = 256,
                   stride_y: int = 256,
                   crop: int = 128,
                   device: str = "cuda"):
    """
    Use train model to make predictions over a spatial extent.

    Parameters
    ----------
    image_in : str
        Path and name of input DTM over which to make prediction. Must include the file extension. 
    pred_out : str
        Path and name of output raster grid. Must include the file extension.
    model : nn.Module 
        Instantiated model with weights loaded. 
    chip_size : int, default=640
        Size of window used to make predictions.
    stride_x : int, default=256
        Stride in the x direction. Must be smaller than chip_size. 
    stride_y : int, default=256
        Stride in the y direction. Must be smaller than chip_size. 
    crop : int, default=128
        Number of rows and columns of cells or pixels to crop from each window. Avoids including margin cells in final predictions. 
    device : str
        Device on which to perform predictions. Default is "cuda".
    """

# ==============================================================
# Read topo map using rasterio and convert to torch tensor
# ==============================================================

    n_channels = 1

    model = model.to(device)

    with rio.open(image_in) as src:
        img = src.read()

    # Convert to torch tensor
    image1 = torch.from_numpy(img)

    # Ensure float32
    image1 = image1.float()

    t_arr = image1  # shape: (C, H, W)

# ==============================================================
# Make blank grid for predictions (same H, W)
# ==============================================================

    # Use first band as template
    p_arr = torch.zeros(
        (t_arr.shape[1], t_arr.shape[2]),
        dtype=torch.float32,
    ).to(device)

    # Predict to entire topo using overlapping chips, merge back to original extent=============
    size = chip_size
    stride_x = stride_x
    stride_y = stride_y
    crop = crop
    n_channels = n_channels

    across_cnt = t_arr.shape[2]
    down_cnt = t_arr.shape[1]
    tile_size_across = size
    tile_size_down = size
    overlap_across = stride_x
    overlap_down = stride_y
    across = math.ceil(across_cnt/overlap_across)
    down = math.ceil(down_cnt/overlap_down)

    print("Processing " + str(across) + " by " + str(down) + " tiles.")

    across_seq = list(range(0, across, 1))
    down_seq = list(range(0, down, 1))
    across_seq2 = [(x*overlap_across) for x in across_seq]
    down_seq2 = [(x*overlap_down) for x in down_seq]
    # Loop through row/column combinations to make predictions for entire image
    for c in across_seq2:
        for r in down_seq2:
            # print("processing tile " + str(c) + " " + str(r) + ".")
            c1 = c
            r1 = r
            c2 = c + size
            r2 = r + size
            # Default
            if c2 <= across_cnt and r2 <= down_cnt:
                r1b = r1
                r2b = r2
                c1b = c1
                c2b = c2
            # Last column
            elif c2 > across_cnt and r2 <= down_cnt:
                r1b = r1
                r2b = r2
                c1b = across_cnt - size
                c2b = across_cnt + 1
            # Last row
            elif c2 <= across_cnt and r2 > down_cnt:
                r1b = down_cnt - size
                r2b = down_cnt + 1
                c1b = c1
                c2b = c2
            # Last row, last column
            else:
                c1b = across_cnt - size
                c2b = across_cnt + 1
                r1b = down_cnt - size
                r2b = down_cnt + 1
            ten1 = t_arr[0:n_channels, r1b:r2b, c1b:c2b]
            ten1 = ten1.to(device).unsqueeze(0)
            model.eval()
            with torch.no_grad():
                ten2 = model(ten1)
            m = nn.Softmax(dim=1)
            pr_probs = m(ten2)
            ten_p = torch.argmax(pr_probs, dim=1).squeeze(1)
            ten_p = ten_p.squeeze()
            # print("executed for " + str(r1) + ", " + str(c1))
            if (r1b == 0 and c1b == 0):  # Write first row, first column
                p_arr[r1b:r2b-crop, c1b:c2b -
                      crop] = ten_p[0:size-crop, 0:size-crop]
            elif (r1b == 0 and c2b == across_cnt+1):  # Write first row, last column
                p_arr[r1b:r2b-crop, c1b+crop:c2b] = ten_p[0:size-crop, 0+crop:size]
            elif (r2b == down_cnt+1 and c1b == 0):  # Write last row, first column
                p_arr[r1b+crop:r2b, c1b:c2b -
                      crop] = ten_p[crop:size+1, 0:size-crop]
            elif (r2b == down_cnt+1 and c2b == across_cnt+1):  # Write last row, last column
                p_arr[r1b+crop:r2b, c1b+crop:c2b] = ten_p[crop:size, 0+crop:size+1]
            elif ((r1b == 0 and c1b != 0) or (r1b == 0 and c2b != across_cnt+1)):  # Write first row
                p_arr[r1b:r2b-crop, c1b+crop:c2b -
                      crop] = ten_p[0:size-crop, 0+crop:size-crop]
            elif ((r2b == down_cnt+1 and c1b != 0) or (r2b == down_cnt+1 and c2b != across_cnt+1)):  # Write last row
                p_arr[r1b+crop:r2b, c1b+crop:c2b -
                      crop] = ten_p[crop:size, 0+crop:size-crop]
            elif ((c1b == 0 and r1b != 0) or (c1b == 0 and r2b != down_cnt+1)):  # Write first column
                p_arr[r1b+crop:r2b-crop, c1b:c2b -
                      crop] = ten_p[crop:size-crop, 0:size-crop]
            elif (c2b == across_cnt+1 and r1b != 0) or (c2b == across_cnt+1 and r2b != down_cnt+1):  # write last column
                p_arr[r1b+crop:r2b-crop, c1b +
                      crop:c2b] = ten_p[crop:size-crop, 0+crop:size]
            else:  # Write middle chips
                p_arr[r1b+crop:r2b-crop, c1b+crop:c2b -
                      crop] = ten_p[crop:size-crop, crop:size-crop]

    # Read in a GeoTIFF to get CRS info=======================================
    image3 = rio.open(image_in)
    profile1 = image3.profile.copy()
    image3.close()
    profile1.update(dtype="uint8", count=1)
    profile1["nodata"] = 255
    profile1["driver"] = "GTiff"
    profile1["dtype"] = "uint8"
    profile1["count"] = 1
    profile1["PHOTOMETRIC"] = "MINISBLACK"
    profile1["COMPRESS"] = "NONE"

    pr_out = p_arr.cpu().numpy().round().astype('uint8')

    # Write out result========================================================
    with rio.open(pred_out, "w", **profile1) as f:
        f.write(pr_out, 1)

    torch.cuda.empty_cache()
