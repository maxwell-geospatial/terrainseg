# terrainseg

Geomorphic semantic segmentation using DTMs and LSPs. 

Supports a workflow for performing geomporphic mapping and feature extraction using land surface parameters (LSPs) derived from high spatial resoltuion digital terrain model (DTM) data. 

Model combines an LSP module, allows for the calculation of LSPs within the model architecture and using tensor-/GPU-based computation, and a trainble semantic segmentation model. Three models are implemented in this package: traditional UNet, a customizable UNet, and a UNet with a ConvNeXt-style decoder and attention gates along the skip connections. However, users can implement their own trainable models or use those provided by another package, such as Segmentation Models. 

Models can be trained using DTM chips and assocated masks that have been pre-generated and saved to disk or chips and mask that are dynamcially generated during the trianing loop. 

## Installation

```bash
$ pip install terrainseg
$ pip install git+https://github.com/maxwell-geospatial/terrainseg.git
```

## Usage

- Please see the provided [documentation](https://www.wvview.org/terrainseg/index.html)

## Contributing

Interested in contributing? Check out the contributing guidelines. Please note that this project is released with a Code of Conduct. By contributing to this project, you agree to abide by its terms.

## License

`terrainseg` was created by Aaron Maxwell. It is licensed under the terms of the MIT license.

## Credits

`terrainseg` was created with [`cookiecutter`](https://cookiecutter.readthedocs.io/en/latest/) and the `py-pkgs-cookiecutter` [template](https://github.com/py-pkgs/py-pkgs-cookiecutter).
