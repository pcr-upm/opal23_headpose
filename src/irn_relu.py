import copy

import numpy as np
import torch.nn as nn


class ConvBNReLU(nn.Sequential):
    def __init__(self, in_planes, out_planes, kernel_size=3, stride=1, groups=1, norm_layer=None, relu='dft'):

        kernel_size_aux = kernel_size
        if isinstance(kernel_size, (list, tuple)):
            kernel_size_aux = np.array(kernel_size_aux)
        padding = (kernel_size_aux - 1) // 2
        if isinstance(kernel_size, (list, tuple)):
            padding = tuple(padding)

        if norm_layer is None:
            norm_layer = nn.BatchNorm2d

        if relu == '6':
            relu_func = nn.ReLU6(inplace=True)
        else:
            relu_func = nn.ReLU(inplace=True)

        super(ConvBNReLU, self).__init__(
            nn.Conv2d(in_planes, out_planes, kernel_size, stride, padding, groups=groups, bias=False),
            norm_layer(out_planes),
            relu_func
        )


class InvertedResidual(nn.Module):
    def __init__(self, inp, oup, stride, expand_ratio, norm_layer=None, dw_kernel=3, groups=1):
        super(InvertedResidual, self).__init__()
        self.stride = stride
        assert stride in [1, 2]

        if norm_layer is None:
            norm_layer = nn.BatchNorm2d

        hidden_dim = int(round(inp * expand_ratio))
        self.use_res_connect = self.stride == 1 and inp == oup

        layers = []
        if expand_ratio != 1:
            # pw
            layers.append(ConvBNReLU(inp, hidden_dim, kernel_size=1, norm_layer=norm_layer, relu='6'))
        layers.extend([
            # dw
            ConvBNReLU(hidden_dim, hidden_dim, stride=stride, kernel_size=dw_kernel,
                       groups=hidden_dim, norm_layer=norm_layer, relu='6'),
            # pw-linear
            nn.Conv2d(hidden_dim, oup, 1, 1, 0, groups=groups, bias=False),
            norm_layer(oup),
        ])
        self.conv = nn.Sequential(*layers)

    def forward(self, x):
        if self.use_res_connect:
            return x + self.conv(x)
        else:
            return self.conv(x)


class IRN(nn.Module):

    def __init__(self,
                 irn_setting,
                 inp_shape=256,
                 in_planes=32,
                 block=None,
                 norm_layer=None,
                 inp_img=True):

        super(IRN, self).__init__()

        if block is None:
            block = InvertedResidual
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d

        # Alignment variables required for heads
        self.in_planes = in_planes
        self.out_cores = 1
        self.channels_core = 0
        self.out_decoders = 0

        # Convert image to input channel tensor
        self.inp_img = inp_img
        if inp_img:
            self.conv_img = ConvBNReLU(3, self.in_planes, kernel_size=1, norm_layer=norm_layer)

        # Prepare backbone
        input_shape = [self.in_planes, inp_shape]
        self.backbone, self.backbone_shape = self._prepare_branch(irn_setting['backbone'], input_shape, block, norm_layer)
        self.channels_core = self.backbone_shape[0]
        # Prepare encoders
        self.encoders = []
        self.encoders, core_shape = self._prepare_branches(irn_setting['encoders'], self.backbone_shape, block, norm_layer)
        self.out_cores = len(self.encoders)
        self.channels_core = core_shape[0]
        self.encoders = nn.ModuleList(self.encoders)

    def _prepare_branch(self, branch, input_shape, block, norm):
        branch_list = []
        out_shape = None
        for t, c, n, s, l in branch:
            stage = []
            block_channel = c * self.in_planes
            stage_head, out_shape = self._encode(input_shape, block_channel, s, norm)
            stage += stage_head

            # Kernel size correction for tensors of shape (c, 1, 1)
            if out_shape[1] == 1:
                dw_kernel = 1
            else:
                dw_kernel = 3

            for block_id in range(n):
                stage.append(block(block_channel, block_channel, 1, t, norm_layer=norm, dw_kernel=dw_kernel))

            branch_list.append(nn.Sequential(*stage))
            input_shape = out_shape

        return nn.ModuleList(branch_list), out_shape

    def _prepare_branches(self, settings, input_shape, block, norm_layer, branch_funct=None):
        branches = []
        input_shape = copy.deepcopy(input_shape)
        if branch_funct is None:
            branch_funct = self._prepare_branch
        for branch_setting in settings:
            branch, last_out_shape = branch_funct(branch_setting, input_shape, block, norm_layer)
            branches.append(branch)
        return branches, last_out_shape

    def _encode(self, input_shape, block_channel, stride, norm):
        layers = []
        [inp_channel, inp_size] = input_shape
        out_shape = input_shape
        same_channel = inp_channel == block_channel
        if stride == 0:
            layers.append(nn.AdaptiveMaxPool2d((1, 1)))
            out_shape[1] = 1
        if not same_channel and stride < 2:
            layers.append(ConvBNReLU(inp_channel, block_channel, kernel_size=1, stride=1, norm_layer=norm))
        elif stride == 2:
            layers.append(ConvBNReLU(inp_channel, block_channel, kernel_size=2, stride=2, norm_layer=norm))
            out_shape[1] //= 2
        out_shape[0] = block_channel
        return layers, out_shape

    def forward(self, x):

        # Image to input tensor
        if self.inp_img:
            x = self.conv_img(x)

        # Backbone
        for stage in self.backbone:
            x = stage(x)
        backbone_out = x

        # Encoders
        core = []
        for idx, encoder in enumerate(self.encoders):
            x = backbone_out
            for i, stage in enumerate(encoder):
                x = stage(x)
            core.append(x)
        output = {'core': core}

        return output
