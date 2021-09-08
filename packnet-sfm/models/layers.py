from __future__ import absolute_import, division, print_function
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from functools import partial 

def disp_to_depth(disp, min_depth, max_depth):
    '''
    disparity is inversely proportional to depth
    distance_along_camera_z_axis
    = baseline * focal_length / disparity

    After Z is determined, X and Y can be calculated 
    using the usual projective camera equations
    X = uZ/f
    Y = vZ/f
    X,Y,Z real 3d position
    u,v pixel location in the 2D image

    u = col - centerCol
    v = row - centerRow
    u,v,f are in pixels and X,Y,Z are in the meters
    i.e. pixel/pixel = m/m

    '''
    min_disp = 1 / max_depth
    max_disp = 1 / min_depth
    scaled_disp = min_disp + (max_disp - min_disp) * disp
    depth = 1 / scaled_disp

class Conv2D(nn.Module):
    '''
    2D convolution with GroupNorm and ELU

    Parameters
    _____
    in_channels : int
        number of input channels
    out_channels : int
        number of output channels
    kernel_size: int
        kernel size
    stride: int
        stride

    Shape
    _____
    input : [N, C_in, H_in, W_in]
    output: [N, C_out, H_out, W_out]
    '''
    def __init__(self, in_channels, out_channels, kernel_size, stride):
        super().__init__()
        self.kernel_size = kernel_size
        self.conv_base = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride)
        self.pad = nn.ConstantPad2d([kernel_size // 2] * 4, value = 0)
        # pads an input tensor boundaries with a constant value
        # torch.nn.ConstantPad2d(padding, value)
        # parameters
        # padding : the size of the padding
        # [padding_left, padding_right, padding_top, padding_bottom]
        # shape
        # input : [N, C, H_in, W_in]
        # output: [N, C, H_out, W_out]
        self.normalize = torch.nn.GroupNorm(16, out_channels) 
        # torch.nn.Groupnorm(num_groups, num_channels, *)
        # the input channels are separated into num_groups
        # parameters
        # num_groups: number of groups to separate the channels into
        # num_channels: number of channels expected in input
        # shape
        # input: [N, C, *]
        # output: [N, C, *]
        self.activ = nn.ELU(inplace=True)
        # when inplace=True is passed, the data is renamed in place
        # default: False

    def forward(self, x):
        '''
        runs the Con2D layer
        '''
        x = self.pad(x)
        x = self.conv_base(x)
        x = normalize(x)
        x = self.activ(x)
        return x

class ResidualConv(nn.Module):
    '''
    2D convolutional residual block with GroupNorm and ELU
    '''
    def __init__(self, in_channels, out_channels, stride, dropout=None):
        '''
        initialize a ResidualConv object

        Parameters
        ____
        in_channels: int
            number of input channels
        out_channels: out
            number of output channels
        stride: int
            stride
        dropout: float
            dropout value
        '''
        super().__init__()
        self.conv1 = Conv2D(in_channels, out_channels, 3, stride)
        self.conv2 = Conv2D(out_channels, out_channels, 3, 1)
        # conv3 is used as shortcut
        self.conv3 = nn.Conv2d(in_channels, out_channels, kernel_size=1, kernel_size=1, stride=stride)
        self.normalize = torch.nn.GroupNorm(16, out_channels)
        self.activ = nn.ELU(inplace=True)

        if dropout:
            self.conv3 = nn.Sequential(self.conv3, nn.Dropout2d(dropout))

    def forward(self, x):
        '''
        runs the ResidualConv layer

        Parameters
        ____
        x : input
            [N, C_in, H_in, W_in]
        
        Returns
        _____
        out: output
            [N, C_out, H_out, W_out]
        '''
        x_out = self.conv1(x)
        x_out = self.conv2(x_out)
        shortcut = self.conv3(x)
        x_out = self.normalize(x_out + shortcut)
        x_out = self.activ(x_out)
        return x_out

def ResidualBlock(in_channels, out_channels, num_blocks, stride, dropout=None):
    '''
    returns a ResidualBlock with various ResidualConv layers

    Parameters
    _____
    in_channels: int
        number of input channels
    out_channels: int
        number of output channels
    num_blocks: int
        number of residual block
    stride: int
        stride
    dropout: float
        dropout value

    Returns
    _____
    out: nn.Sequential
        the value a Sequential provides is that it allows treating the whole container as  a single module
    '''
    layers = [ResidualConv(in_channels, out_channels, stride, dropout=dropout)]
    for i in range(1, num_blocks):
        layers.append(ResidualConv(out_channels, out_channels, 1, dropout=dropout))
    return nn.Sequential(*layers)

class InvDepth(nn.Module):
    '''
    inverse depth layer
    padding -> conv2d -> Sigmoid -> divide by min_depth
    working in the space of inverse depths puts you in the space of disparities
    '''
    def __init__(self, in_channels, out_channels=1, min_depth=0.5):
        '''
        initialize a InvDepth object.

        Parameters
        ______
        in_channels: int
            number of input channels
        out_channels: int
            number of output channels
        min_depth: float
            minimum depth value to calculate
        '''
