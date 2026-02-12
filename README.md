# terrainseg

<img src="terrainsegHex.png" alt="Alt text" width="250">

Geomorphic semantic segmentation using DTMs and LSPs. 

Supports a workflow for performing geomorphic mapping and feature extraction using land surface parameters (LSPs) derived from high spatial resolution digital terrain model (DTM) data. 

Model combines an LSP module, which allows for the calculation of LSPs within the model architecture and using tensor-/GPU-based computation, and a trainable semantic segmentation model. Three models are implemented in this package: traditional UNet, a customizable UNet, and a UNet with a ConvNeXt-style decoder and attention gates along the skip connections. However, users can implement their own trainable models or use those provided by another package, such as Segmentation Models. 

Models can be trained using DTM chips and associated masks that have been pre-generated and saved to disk or chips and masks that are dynamically generated during the training loop. 

<img src="architecture.png" alt="Alt text" width="500">

## Installation

```bash
$ pip install terrainseg
$ pip install git+https://github.com/maxwell-geospatial/terrainseg.git
```

## Usage

- Please see the provided [documentation](https://www.wvview.org/terrainseg/index.html)

## License

`terrainseg` was created by Aaron Maxwell, Sarah Farhadpour, and Miles Reed. It is licensed under the terms of the MIT license.

## Credits

`terrainseg` was created with [`cookiecutter`](https://cookiecutter.readthedocs.io/en/latest/) and the `py-pkgs-cookiecutter` [template](https://github.com/py-pkgs/py-pkgs-cookiecutter).
