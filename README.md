# terrainseg

<img src="terrainsegHex.png" alt="Alt text" width="250">

Geomorphic semantic segmentation using DTMs and LSPs. 

Supports a workflow for performing geomorphic mapping and feature extraction using land surface parameters (LSPs) derived from high spatial resolution digital terrain model (DTM) data. 

Model combines an LSP module, which allows for the calculation of LSPs within the model architecture and using tensor-/GPU-based computation, and a trainable semantic segmentation model. Three models are implemented in this package: traditional UNet, a customizable UNet, and a UNet with a ConvNeXt-style decoder and attention gates along the skip connections. However, users can implement their own trainable models or use those provided by another package, such as Segmentation Models. 

Models can be trained using DTM chips and associated masks that have been pre-generated and saved to disk or chips and masks that are dynamically generated during the training loop. 

<img src="architecture.png" alt="Alt text" width="500">

## Installation

We recommend:

1. Creating a virtual Python environment using Python 3.11 or 3.12
2. Installing PyTorch and TorchVision as instructed on the [PyTorch webpage](https://pytorch.org/)
3. Installing the other dendencies using pip: rasterio, pandas, geopandas, matplotlib, torchmetrics, segmentation-models-pytorch, and albumentations
4. Installing terrainseg

```bash
$ pip install terrainseg
$ pip install git+https://github.com/maxwell-geospatial/terrainseg.git
```

**Note**: If you want to use GPU acceleration, you will need to install and set up CUDA and cuDNN. We used CUDA verion 12.6. 

## Usage

- Please see the provided [documentation](https://www.wvview.org/terrainseg/index.html)

## License

`terrainseg` was created by Aaron Maxwell, Sarah Farhadpour, and Miles Reed. It is licensed under the terms of the MIT license.

## Credits

`terrainseg` was created with [`cookiecutter`](https://cookiecutter.readthedocs.io/en/latest/) and the `py-pkgs-cookiecutter` [template](https://github.com/py-pkgs/py-pkgs-cookiecutter).