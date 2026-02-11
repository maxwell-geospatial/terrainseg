# read version from installed package (fallback when running from source)
from importlib.metadata import PackageNotFoundError, version

try:
    __version__ = version("terrainseg")
except PackageNotFoundError:
    __version__ = "0.0.0+local"

from .models.unet import defineUNet
from .models.customunet import defineCUNet
from .models.convnextunet import defineCNXTUNet

from .utils.checks import viewBatch
from .utils.chips import (makeMasks,
                          makeChips,
                          makeChipsDF,
                          makeChipsMulticlass)
from .utils.dynamicchips import (
    makeDynamicChipsGDF,
    checkDynamicChips,
    saveDynamicChips
)
from .utils.lsps import (gaussPyramids,
                         lspCalc,
                         lspModule)
from .utils.spatialpredict import terrainPredict
from .utils.trainer import (terrainDataset,
                            terrainDatasetDynamic,
                            terrainSegModel,
                            unifiedFocalLoss,
                            terrainTrainer)

__all__ = [
    "defineUNet",
    "defineCUNet",
    "defineCNXTUNet",
    "defineSMPModel",
    "makeMasks",
    "makeChips",
    "makeChipsDF",
    "makeChipsMulticlass",
    "makeDynamicChipsGDF",
    "checkDynamicChips",
    "unifiedFocalLoss",
    "terrainPredict",
    "terrainDataset",
    "terrainDatasetDynamic",
    "terrainSegModel"
    "terrainTrainer",
    "viewBatch",
    "gaussPyramids",
    "lspCalc",
    "lspModule"
]
