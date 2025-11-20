import torch
from torch import nn
import numpy as np
from typing import Union, Type, List, Tuple
from torch.nn.modules.conv import _ConvNd
from torch.nn.modules.dropout import _DropoutNd
from dynamic_network_architectures.building_blocks.helper import (
    convert_conv_op_to_dim,
    maybe_convert_scalar_to_list,
    get_matching_pool_op,
    get_matching_convtransp,
)
from dynamic_network_architectures.initialization.weight_init import InitWeights_He

##############################################################
# Residual Block Wrapper
##############################################################
class ResidualBlock(nn.Module):
    def __init__(self, main_module: nn.Module, conv_op: Type[_ConvNd], in_ch: int, out_ch: int):
        super().__init__()
        self.main = main_module
        self.need_proj = in_ch != out_ch
        if self.need_proj:
            self.proj = conv_op(in_ch, out_ch, kernel_size=1, stride=1, bias=False)
        else:
            self.proj = nn.Identity()

    def forward(self, x):
        return self.main(x) + self.proj(x)

##############################################################
# StackedConvBlocks with optional Residual
##############################################################
class StackedConvBlocks(nn.Module):
    def __init__(self,
                 num_convs: int,
                 conv_op: Type[_ConvNd],
                 input_channels: int,
                 output_channels: Union[int, List[int], Tuple[int, ...]],
                 kernel_size: Union[int, List[int], Tuple[int, ...]],
                 initial_stride: Union[int, List[int], Tuple[int, ...]],
                 conv_bias: bool = False,
                 norm_op: Union[None, Type[nn.Module]] = None,
                 norm_op_kwargs: dict = None,
                 dropout_op: Union[None, Type[_DropoutNd]] = None,
                 dropout_op_kwargs: dict = None,
                 nonlin: Union[None, Type[torch.nn.Module]] = None,
                 nonlin_kwargs: dict = None,
                 nonlin_first: bool = False,
                 use_residual: bool = True,
                 ):
        super().__init__()

        if not isinstance(output_channels, (tuple, list)):
            output_channels = [output_channels] * num_convs

        blocks = nn.Sequential(
            ConvDropoutNormReLU(
                conv_op, input_channels, output_channels[0], kernel_size, initial_stride, conv_bias,
                norm_op, norm_op_kwargs, dropout_op, dropout_op_kwargs, nonlin, nonlin_kwargs, nonlin_first
            ),
            *[
                ConvDropoutNormReLU(
                    conv_op, output_channels[i - 1], output_channels[i], kernel_size, 1, conv_bias,
                    norm_op, norm_op_kwargs, dropout_op, dropout_op_kwargs, nonlin, nonlin_kwargs, nonlin_first
                ) for i in range(1, num_convs)
            ]
        )

        if use_residual:
            self.convs = ResidualBlock(blocks, conv_op, input_channels, output_channels[-1])
        else:
            self.convs = blocks

        self.output_channels = output_channels[-1]
        self.initial_stride = maybe_convert_scalar_to_list(conv_op, initial_stride)

    def forward(self, x):
        return self.convs(x)

    def compute_conv_feature_map_size(self, input_size):
        size_after_stride = [i // j for i, j in zip(input_size, self.initial_stride)]
        return np.prod([self.output_channels, *size_after_stride], dtype=np.int64)

##############################################################
# ConvDropoutNormReLU (same as original except unchanged)
##############################################################
class ConvDropoutNormReLU(nn.Module):
    def __init__(self,
                 conv_op: Type[_ConvNd],
                 input_channels: int,
                 output_channels: int,
                 kernel_size: Union[int, List[int], Tuple[int, ...]],
                 stride: Union[int, List[int], Tuple[int, ...]],
                 conv_bias: bool = False,
                 norm_op: Union[None, Type[nn.Module]] = None,
                 norm_op_kwargs: dict = None,
                 dropout_op: Union[None, Type[_DropoutNd]] = None,
                 dropout_op_kwargs: dict = None,
                 nonlin: Union[None, Type[torch.nn.Module]] = None,
                 nonlin_kwargs: dict = None,
                 nonlin_first: bool = False):
        super().__init__()
        stride = maybe_convert_scalar_to_list(conv_op, stride)
        kernel_size = maybe_convert_scalar_to_list(conv_op, kernel_size)
        if norm_op_kwargs is None:
            norm_op_kwargs = {}
        if nonlin_kwargs is None:
            nonlin_kwargs = {}

        self.conv = conv_op(
            input_channels,
            output_channels,
            kernel_size,
            stride,
            padding=[(i - 1) // 2 for i in kernel_size],
            dilation=1,
            bias=conv_bias
        )

        ops = [self.conv]
        if dropout_op is not None:
            ops.append(dropout_op(**dropout_op_kwargs))
        if norm_op is not None:
            ops.append(norm_op(output_channels, **norm_op_kwargs))
        if nonlin is not None:
            ops.append(nonlin(**nonlin_kwargs))
        if nonlin_first and (norm_op is not None and nonlin is not None):
            ops[-1], ops[-2] = ops[-2], ops[-1]

        self.all_modules = nn.Sequential(*ops)

    def forward(self, x):
        return self.all_modules(x)

##############################################################
# Encoder with residual integration
##############################################################
class PlainConvEncoder(nn.Module):
    def __init__(self,
                 input_channels: int,
                 n_stages: int,
                 features_per_stage: Union[int, List[int], Tuple[int, ...]],
                 conv_op: Type[_ConvNd],
                 kernel_sizes: Union[int, List[int], Tuple[int, ...]],
                 strides: Union[int, List[int], Tuple[int, ...]],
                 n_conv_per_stage: Union[int, List[int], Tuple[int, ...]],
                 conv_bias: bool = False,
                 norm_op: Union[None, Type[nn.Module]] = None,
                 norm_op_kwargs: dict = None,
                 dropout_op: Union[None, Type[_DropoutNd]] = None,
                 dropout_op_kwargs: dict = None,
                 nonlin: Union[None, Type[torch.nn.Module]] = None,
                 nonlin_kwargs: dict = None,
                 return_skips: bool = False,
                 nonlin_first: bool = False,
                 pool: str = 'conv'):
        super().__init__()

        if isinstance(kernel_sizes, int): kernel_sizes = [kernel_sizes] * n_stages
        if isinstance(features_per_stage, int): features_per_stage = [features_per_stage] * n_stages
        if isinstance(n_conv_per_stage, int): n_conv_per_stage = [n_conv_per_stage] * n_stages
        if isinstance(strides, int): strides = [strides] * n_stages

        stages = []
        for s in range(n_stages):
            stage_mods = []
            if pool == 'max' or pool == 'avg':
                if (isinstance(strides[s], int) and strides[s] != 1) or (
                    isinstance(strides[s], (tuple, list)) and any(i != 1 for i in strides[s])
                ):
                    stage_mods.append(get_matching_pool_op(conv_op, pool_type=pool)(kernel_size=strides[s], stride=strides[s]))
                conv_stride = 1
            else:
                conv_stride = strides[s]

            stage_mods.append(StackedConvBlocks(
                n_conv_per_stage[s], conv_op, input_channels, features_per_stage[s], kernel_sizes[s], conv_stride,
                conv_bias, norm_op, norm_op_kwargs, dropout_op, dropout_op_kwargs,
                nonlin, nonlin_kwargs, nonlin_first, use_residual=True
            ))
            stages.append(nn.Sequential(*stage_mods))
            input_channels = features_per_stage[s]

        self.stages = nn.Sequential(*stages)
        self.output_channels = features_per_stage
        self.strides = [maybe_convert_scalar_to_list(conv_op, i) for i in strides]
        self.return_skips = return_skips
        self.conv_op = conv_op
        self.norm_op = norm_op
        self.norm_op_kwargs = norm_op_kwargs
        self.nonlin = nonlin
        self.nonlin_kwargs = nonlin_kwargs
        self.dropout_op = dropout_op
        self.dropout_op_kwargs = dropout_op_kwargs
        self.conv_bias = conv_bias
        self.kernel_sizes = kernel_sizes

    def forward(self, x):
        outputs = []
        for s in self.stages:
            x = s(x)
            outputs.append(x)
        return outputs if self.return_skips else outputs[-1]

##############################################################
# Decoder with residual integration
##############################################################
class UNetDecoder(nn.Module):
    def __init__(self,
                 encoder: PlainConvEncoder,
                 num_classes: int,
                 n_conv_per_stage: Union[int, Tuple[int, ...], List[int]],
                 deep_supervision,
                 nonlin_first: bool = False):
        super().__init__()
        self.deep_supervision = deep_supervision
        self.encoder = encoder
        n_stages_encoder = len(encoder.output_channels)

        if isinstance(n_conv_per_stage, int):
            n_conv_per_stage = [n_conv_per_stage] * (n_stages_encoder - 1)

        transp_op = get_matching_convtransp(conv_op=encoder.conv_op)
        stages = []
        transpconvs = []
        seg_layers = []

        for s in range(1, n_stages_encoder):
            in_below = encoder.output_channels[-s]
            in_skip = encoder.output_channels[-(s + 1)]
            stride = encoder.strides[-s]

            transpconvs.append(transp_op(in_below, in_skip, stride, stride, bias=True))

            stages.append(StackedConvBlocks(
                n_conv_per_stage[s - 1], encoder.conv_op, 2 * in_skip, in_skip,
                encoder.kernel_sizes[-(s + 1)], 1,
                encoder.conv_bias, encoder.norm_op, encoder.norm_op_kwargs,
                encoder.dropout_op, encoder.dropout_op_kwargs,
                encoder.nonlin, encoder.nonlin_kwargs, nonlin_first,
                use_residual=True
            ))

            seg_layers.append(encoder.conv_op(in_skip, num_classes, 1, 1, 0, bias=True))

        self.stages = nn.ModuleList(stages)
        self.transpconvs = nn.ModuleList(transpconvs)
        self.seg_layers = nn.ModuleList(seg_layers)

    def forward(self, skips):
        x = skips[-1]
        seg_outs = []

        for s in range(len(self.stages)):
            x = self.transpconvs[s](x)
            x = torch.cat((x, skips[-(s + 2)]), dim=1)
            x = self.stages[s](x)

            if self.deep_supervision:
                seg_outs.append(self.seg_layers[s](x))
            elif s == len(self.stages) - 1:
                seg_outs.append(self.seg_layers[-1](x))

        return seg_outs[::-1] if self.deep_supervision else seg_outs[0]

##############################################################
# Final UNet wrapper
##############################################################
class DCPlainConvUNet(nn.Module):
    def __init__(self,
                 input_channels,
                 n_stages,
                 features_per_stage,
                 conv_op,
                 kernel_sizes,
                 strides,
                 n_conv_per_stage,
                 num_classes,
                 n_conv_per_stage_decoder,
                 conv_bias=False,
                 norm_op=None,
                 norm_op_kwargs=None,
                 dropout_op=None,
                 dropout_op_kwargs=None,
                 nonlin=None,
                 nonlin_kwargs=None,
                 deep_supervision=False,
                 nonlin_first=False):
        super().__init__()

        self.encoder = PlainConvEncoder(
            input_channels, n_stages, features_per_stage, conv_op, kernel_sizes, strides,
            n_conv_per_stage, conv_bias, norm_op, norm_op_kwargs, dropout_op, dropout_op_kwargs,
            nonlin, nonlin_kwargs, return_skips=True, nonlin_first=nonlin_first
        )

        self.decoder = UNetDecoder(
            self.encoder, num_classes, n_conv_per_stage_decoder, deep_supervision,
            nonlin_first=nonlin_first
        )

    def forward(self, x):
        skips = self.encoder(x)
        return self.decoder(skips)
