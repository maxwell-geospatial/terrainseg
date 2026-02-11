from typing import Sequence, Optional, Dict, Any, Union, Tuple, Literal
import torch
import torch.nn as nn

def double_conv(inChannels, outChannels):
  return nn.Sequential(
      nn.Conv2d(inChannels, outChannels, kernel_size=(3,3), stride=1, padding=1),
      nn.BatchNorm2d(outChannels),
      nn.ReLU(inplace=True),
      nn.Conv2d(outChannels, outChannels, kernel_size=(3,3), stride=1, padding=1),
      nn.BatchNorm2d(outChannels),
      nn.ReLU(inplace=True)
  )

def up_conv(inChannels, outChannels):
    return nn.Sequential(
      nn.ConvTranspose2d(inChannels, outChannels, kernel_size=(2,2), stride=2),
      nn.BatchNorm2d(outChannels),
      nn.ReLU(inplace=True)
  )

class defineUNet(nn.Module):
  """
  UNet for terrain semantic segmentation.

  Parameters
  ----------
  inChn : int, default=6
    Number of input channels.
  nCls : int, default = 2
    Number of output classes.
  encoderChn : Tuple[int, int, int, int], default=(16, 32, 64, 128).
    Number of output feature maps from each encoder block. Must be a tuple of 4 integers.
  decoderChn : Tuple[int, int, int, int], default=(128,64,32,16).
    Number of output feature maps from each decoder block. Must be a tuple of 4 integers.
  botChn : int, default=256
    Number of output feature maps from the bottleneck block. 

  Returns
  -------
    Model as nn.Module subclass.
  """

  def __init__(self,
               inChn: int = 6, 
               nCls: int = 2,
               encoderChn: Tuple[int, int, int, int] = (16, 32, 64, 128), 
               decoderChn: Tuple[int, int, int, int] = (128, 64, 32, 16), 
               botChn: int = 256):
    
    """
    Initialize a standard U-Net architecture.

    The network consists of a 4-stage encoder with max-pool downsampling,
    a bottleneck block, and a 4-stage decoder with skip-connection
    concatenation. Each stage uses a double convolution block, and spatial
    resolution is reduced by a factor of 16 before reconstruction.

    Parameters
    ----------
    inChn : int, default=6
        Number of channels in the input tensor.
    nCls : int, default=2
        Number of output segmentation classes (logits per pixel).
    encoderChn : tuple[int, int, int, int], default=(16, 32, 64, 128)
        Output channel widths of encoder blocks `(e1, e2, e3, e4)`.
        Each encoder stage halves spatial resolution via max pooling.
    decoderChn : tuple[int, int, int, int], default=(128, 64, 32, 16)
        Output channel widths of decoder blocks `(d1, d2, d3, d4)`.
        Decoder inputs are concatenations of upsampled decoder features
        and encoder skip features, so the *input* channel count to each
        decoder block equals:

        - d1: `botChn + encoderChn[3]`
        - d2: `decoderChn[0] + encoderChn[2]`
        - d3: `decoderChn[1] + encoderChn[1]`
        - d4: `decoderChn[2] + encoderChn[0]`
    botChn : int, default=256
        Number of output channels produced by the bottleneck block and
        the starting width of the decoder.

    Raises
    ------
    AssertionError
        Recommended to enforce that `encoderChn` and `decoderChn` contain
        exactly four elements each (not enforced automatically).

    Notes
    -----
    - Input spatial dimensions should be divisible by 16 for exact
      reconstruction without cropping.
    - The final layer outputs raw logits of shape `(B, nCls, H, W)`.
      Apply `softmax` (multi-class) or `sigmoid` (binary/multi-label)
      outside the model depending on the loss function used.
    """


    super().__init__()
    self.inChn = inChn
    self.nCls = nCls
    self.encoderChn = encoderChn
    self.decoderChn = decoderChn 
    self.botChn = botChn

    self.encoder1 = double_conv(inChn, encoderChn[0])
    
    self.encoder2 = nn.Sequential(nn.MaxPool2d(kernel_size=2, stride=2),
                                  double_conv(encoderChn[0], encoderChn[1]))
    
    self.encoder3 = nn.Sequential(nn.MaxPool2d(kernel_size=2, stride=2),
                                  double_conv(encoderChn[1], encoderChn[2]))
    
    self.encoder4 = nn.Sequential(nn.MaxPool2d(kernel_size=2, stride=2),
                                  double_conv(encoderChn[2], encoderChn[3]))
    
    self.bottleneck = nn.Sequential(nn.MaxPool2d(kernel_size=2, stride=2),
                                    double_conv(encoderChn[3], botChn))

    self.decoder1up = up_conv(botChn, botChn)
    self.decoder1 = double_conv(encoderChn[3]+botChn, decoderChn[0])

    self.decoder2up = up_conv(decoderChn[0], decoderChn[0])
    self.decoder2 = double_conv(encoderChn[2]+decoderChn[0], decoderChn[1])

    self.decoder3up = up_conv(decoderChn[1], decoderChn[1])
    self.decoder3 = double_conv(encoderChn[1]+decoderChn[1], decoderChn[2])

    self.decoder4up = up_conv(decoderChn[2], decoderChn[2])
    self.decoder4 = double_conv(encoderChn[0]+decoderChn[2], decoderChn[3])

    self.classifier = nn.Conv2d(decoderChn[3], nCls, kernel_size=(1,1))

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

    #Encoder
    encoder1 = self.encoder1(x)
    encoder2 = self.encoder2(encoder1)
    encoder3 = self.encoder3(encoder2)
    encoder4 = self.encoder4(encoder3)

    #Bottleneck
    x = self.bottleneck(encoder4)

    #Decoder
    x = self.decoder1up(x)
    x = torch.concat([x, encoder4], dim=1)
    x = self.decoder1(x)

    x = self.decoder2up(x)
    x = torch.concat([x, encoder3], dim=1)
    x = self.decoder2(x)

    x = self.decoder3up(x)
    x = torch.concat([x, encoder2], dim=1)
    x = self.decoder3(x)

    x = self.decoder4up(x)
    x = torch.concat([x, encoder1], dim=1)
    x = self.decoder4(x)

    #Classifier head
    x = self.classifier(x)

    return x