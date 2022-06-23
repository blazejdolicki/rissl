import torch
from torch import Tensor
from typing import Type, Any, Callable, Union, List, Optional
import logging
import torch.nn.functional as F

from e2cnn import nn
from e2cnn import gspaces
from .utils import *

# Equivariant resnet architecture based on standard ResNet from torchvision
# https://github.com/pytorch/vision/blob/ecbff88a1ad605bf04d6c44862e93dde2fdbfc84/torchvision/models/resnet.py
# and https://github.com/QUVA-Lab/e2cnn_experiments


class E2BasicBlock(nn.EquivariantModule):
    expansion: int = 1

    def __init__(
        self,
        in_fiber: nn.FieldType,
        inner_fiber: nn.FieldType,
        out_fiber: nn.FieldType = None,
        stride: int = 1,
        groups: int = 1,
        base_width: int = 64,
        dilation: int = 1,
        F: float = 1.,
        sigma: float = 0.45,
    ) -> None:
        super(E2BasicBlock, self).__init__()

        if out_fiber is None:
            out_fiber = in_fiber

        self.in_type = in_fiber
        inner_class = inner_fiber
        self.out_type = out_fiber

        if isinstance(in_fiber.gspace, gspaces.FlipRot2dOnR2):
            rotations = in_fiber.gspace.fibergroup.rotation_order
        elif isinstance(in_fiber.gspace, gspaces.Rot2dOnR2):
            rotations = in_fiber.gspace.fibergroup.order()
        else:
            rotations = 0

        if rotations in [0, 2, 4]:
            conv = conv3x3
        else:
            conv = conv5x5


        if groups != 1 or base_width != 64:
            raise ValueError('BasicBlock only supports groups=1 and base_width=64')
        if dilation > 1:
            raise NotImplementedError("Dilation > 1 not supported in BasicBlock")

        # Both self.conv1 and self.downsample layers downsample the input when stride != 1
        self.conv1 = conv(self.in_type, inner_class, stride=stride, sigma=sigma, F=F, initialize=False)
        self.bn1 = nn.InnerBatchNorm(inner_class)
        self.relu = nn.ReLU(inner_class, inplace=True)
        self.conv2 = conv(inner_class, self.out_type, sigma=sigma, F=F, initialize=False)
        self.bn2 = nn.InnerBatchNorm(self.out_type)
        # add another relu because the shape changes
        self.relu2 = nn.ReLU(self.out_type, inplace=True)
        self.stride = stride


        # `downsample` in resnet.py is the equivalent of `shortcut` in e2_wide_resnet.py

        self.downsample = None
        if stride != 1 or self.in_type != self.out_type:
            self.downsample = nn.SequentialModule(
                conv1x1(self.in_type, self.out_type, stride=stride, bias=False, sigma=sigma, F=F, initialize=False),
                nn.InnerBatchNorm(self.out_type),
                )

    def forward(self, x: Tensor) -> Tensor:
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu2(out)

        return out

    # abstract method
    def evaluate_output_shape(self, input_shape):
        raise NotImplementedError

class E2Bottleneck(nn.EquivariantModule):
    # Bottleneck in torchvision places the stride for downsampling at 3x3 convolution(self.conv2)
    # while original implementation places the stride at the first 1x1 convolution(self.conv1)
    # according to "Deep residual learning for image recognition"https://arxiv.org/abs/1512.03385.
    # This variant is also known as ResNet V1.5 and improves accuracy according to
    # https://ngc.nvidia.com/catalog/model-scripts/nvidia:resnet_50_v1_5_for_pytorch.

    expansion: int = 4

    def __init__(
        self,
        in_fiber: nn.FieldType,
        inner_fiber: nn.FieldType,
        out_fiber: nn.FieldType=None,
        stride: int = 1,
        downsample: Optional[nn.EquivariantModule] = None,
        groups: int = 1,
        base_width: int = 64,
        dilation: int = 1,
        F: float = 1.,
        sigma: float = 0.45,
    ) -> None:
        super(E2Bottleneck, self).__init__()

        if out_fiber is None:
            out_fiber = in_fiber

        self.in_type = in_fiber


        if isinstance(in_fiber.gspace, gspaces.FlipRot2dOnR2):
            rotations = in_fiber.gspace.fibergroup.rotation_order
        elif isinstance(in_fiber.gspace, gspaces.Rot2dOnR2):
            rotations = in_fiber.gspace.fibergroup.order()
        else:
            rotations = 0

        if rotations in [0, 2, 4]:
            conv = conv3x3
        else:
            conv = conv5x5

        planes = len(inner_fiber)
        # for ResNext50_32x4d base_width (width_per_group) is 4 and there are 32 groups
        width = int(planes * (base_width / 64.)) * groups

        # now we need to get the same field type but with `width` representations (number of channels)
        first_rep_type = type(in_fiber.representations[0])
        for rep in in_fiber.representations:
            assert first_rep_type == type(rep)

        width_fiber = nn.FieldType(in_fiber.gspace, width * [in_fiber.representations[0]])
        # Both self.conv2 and self.downsample layers downsample the input when stride != 1

        self.conv1 = conv1x1(in_fiber, width_fiber, sigma=sigma, F=F, initialize=False)
        self.bn1 = nn.InnerBatchNorm(width_fiber)
        self.relu1 = nn.ReLU(width_fiber, inplace=True)
        self.conv2 = conv(width_fiber, width_fiber, stride=stride, dilation=dilation, groups=groups, sigma=sigma, F=F, initialize=False)
        self.bn2 = nn.InnerBatchNorm(width_fiber)

        # this might need to be changed to `out_fiber` from `in_fiber` for different value of conv2triv
        exp_out_fiber = nn.FieldType(in_fiber.gspace, planes * self.expansion * [in_fiber.representations[0]])

        self.conv3 = conv1x1(width_fiber, exp_out_fiber, sigma=sigma, F=F, initialize=False)
        self.bn3 = nn.InnerBatchNorm(exp_out_fiber)
        self.relu2 = nn.ReLU(exp_out_fiber, inplace=True)
        self.downsample = downsample
        self.stride = stride

        # `downsample` in resnet.py is the equivalent of `shortcut` in e2_wide_resnet.py
        self.downsample = None
        if stride != 1 or self.in_type != exp_out_fiber:
            self.downsample = nn.SequentialModule(
                conv1x1(self.in_type, exp_out_fiber, stride=stride, bias=False, sigma=sigma, F=F, initialize=False),
                nn.InnerBatchNorm(exp_out_fiber),
            )

        self.out_type = exp_out_fiber

    def forward(self, x: Tensor) -> Tensor:
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu1(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu1(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu2(out)

        return out

    # abstract method
    def evaluate_output_shape(self, input_shape):
        raise NotImplementedError

class E2ResNet(torch.nn.Module):

    def __init__(
        self,
        block: Type[Union[E2BasicBlock, E2Bottleneck]],
        layers: List[int],
        num_classes: int = 1000,
        N: int = 8,
        restrict: int = 1,
        flip: bool = True,
        main_fiber: str = "regular",
        inner_fiber: str = "regular",
        F: float = 1.,
        sigma: float = 0.45,
        deltaorth: bool = False,
        fixparams: bool = True,
        initial_stride: int = 1,
        conv2triv: bool = True,
        zero_init_residual: bool = False,
        groups: int = 1,
        width_per_group: int = 64,
        num_input_channels=3,
        replace_stride_with_dilation: Optional[List[bool]] = None
    ) -> None:
        """

        :param block: Type of block used in the model (E2BasicBlock or E2Bottleneck)
        :param layers: Number of blocks in each layer
        :param num_classes:
        :param N:
        :param restrict:
        :param f: If the model is flip equivariant.
        :param main_fiber:
        :param inner_fiber:
        :param F:
        :param sigma:
        :param deltaorth:
        :param fixparams:
        :param conv2triv:
        :param zero_init_residual:
        :param groups:
        :param width_per_group:
        :param replace_stride_with_dilation:
        """
        super(E2ResNet, self).__init__()

        # Standard initialization of ResNet

        # Number of output channels of the first convolution
        self.inplanes = 64
        self.dilation = 1

        if replace_stride_with_dilation is None:
            # each element in the tuple indicates if we should replace
            # the 2x2 stride with a dilated convolution instead
            replace_stride_with_dilation = [False, False, False]
        if len(replace_stride_with_dilation) != 3:
            raise ValueError("replace_stride_with_dilation should be None "
                             "or a 3-element tuple, got {}".format(replace_stride_with_dilation))
        self.groups = groups
        self.base_width = width_per_group

        # Equivariant part of initialization of ResNet
        self._fixparams = fixparams
        self.conv2triv = conv2triv

        self._layer = 0
        self._N = N

        # if the model is [F]lip equivariant
        self._f = flip

        # level of [R]estriction:
        #   r < 0 : never do restriction, i.e. initial group (either D8 or C8) preserved for the whole network
        #   r = 0 : do restriction before first layer, i.e. initial group doesn't have rotation equivariance (C1 or D1)
        #   r > 0 : restrict after every block, i.e. start with 8 rotations, then restrict to 4 and finally 1
        self._r = restrict

        self._F = F
        self._sigma = sigma

        if self._f:
            self.gspace = gspaces.FlipRot2dOnR2(N)
        else:
            self.gspace = gspaces.Rot2dOnR2(N)

        if self._r == 0:
            id = (0, 1) if self._f else 1
            self.gspace, _, _ = self.gspace.restrict(id)

        # Start building layers

        # field type of layer lifting the Z^2 input to N rotations
        self.in_lifting_type = nn.FieldType(self.gspace, [self.gspace.trivial_repr] * num_input_channels)

        # field type for the first lifted layer
        self.next_in_type = FIBERS[main_fiber](self.gspace, self.inplanes, fixparams=self._fixparams)

        # number of output channels in each outer layer
        num_channels = [64, 128, 256, 512]
        # For this initial cnn, torchvision ResNet uses kernel_size=7, stride=2, padding=3
        # wide_resnet.py uses kernel_size=3, stride=1, padding=1
        # and e2_wideresnet.py uses kernel_size=5. We follow the latter.
        self.conv1 = conv5x5(self.in_lifting_type, self.next_in_type, sigma=sigma, F=F, initialize=False)
        self.bn1 = nn.InnerBatchNorm(self.next_in_type)
        self.relu = nn.ReLU(self.next_in_type, inplace=True)
        self.maxpool = nn.PointwiseMaxPool(self.next_in_type, kernel_size=3, stride=2, padding=1)
        # self.layer_i is equivalent to self.block_i in wide_resnet.py (for ith layer)
        # self._make_layer is equivalent to NetworkBlock (which contains the same method) in wide_resnet.py
        # and to _wide_layer in e2_wide_resnet.py

        self.layer1 = self._make_layer(block, num_channels[0], layers[0], stride=initial_stride,
                                       dilate=replace_stride_with_dilation[0],
                                       main_fiber=main_fiber, inner_fiber=inner_fiber)

        # first restriction layer
        if self._r > 0:
            id = (0, 4) if self._f else 4
            self.restrict1 = self._restrict_layer(id)
        else:
            self.restrict1 = lambda x: x

        self.layer2 = self._make_layer(block, num_channels[1], layers[1], stride=2,
                                       dilate=replace_stride_with_dilation[0],
                                       main_fiber=main_fiber, inner_fiber=inner_fiber)

        # second restriction layer
        if self._r > 1:
            id = (0, 1) if self._f else 1
            self.restrict2 = self._restrict_layer(id)
        else:
            self.restrict2 = lambda x: x

        self.layer3 = self._make_layer(block, num_channels[2], layers[2], stride=2,
                                       dilate=replace_stride_with_dilation[1],
                                       main_fiber=main_fiber, inner_fiber=inner_fiber)

        if self.conv2triv:
            out_fiber = "trivial"
        else:
            out_fiber = None

        self.layer4 = self._make_layer(block, num_channels[3], layers[3], stride=2,
                                       dilate=replace_stride_with_dilation[2],
                                       main_fiber=main_fiber, inner_fiber=inner_fiber, out_fiber=out_fiber)

        if not self.conv2triv:
            self.mp = nn.GroupPooling(self.layer4.out_type)

        self.avgpool = torch.nn.AdaptiveAvgPool2d((1,1))
        linear_input_features = self.mp.out_type.size if not self.conv2triv else self.layer4.out_type.size
        # TODO not sure about the linear input size here
        self.fc = torch.nn.Linear(linear_input_features, num_classes)

        for module in self.modules():
            if isinstance(module, nn.R2Conv):
                if deltaorth:
                    init.deltaorthonormal_init(module.weights.data, module.basisexpansion)
                else:
                    init.generalized_he_init(module.weights.data, module.basisexpansion)
            elif isinstance(module, torch.nn.Linear):
                module.bias.data.zero_()

        # Zero-initialize the last BN in each residual branch,
        # so that the residual branch starts with zeros, and each residual block behaves like an identity.
        # This improves the model by 0.2~0.3% according to https://arxiv.org/abs/1706.02677
        # FIXME doubt this is gonna work
        if zero_init_residual:
            for m in self.modules():
                if isinstance(m, E2Bottleneck):
                    nn.init.constant_(m.bn3.weight, 0)  # type: ignore[arg-type]
                elif isinstance(m, E2BasicBlock):
                    nn.init.constant_(m.bn2.weight, 0)  # type: ignore[arg-type]

        num_params = sum([p.numel() for p in self.parameters() if p.requires_grad])
        print("Total number of learnable parameters:", num_params)

    def _make_layer(self, block: Type[Union[E2BasicBlock, E2Bottleneck]], planes: int, num_blocks: int,
                    stride: int = 1, dilate: bool = False,
                    main_fiber: str = "regular",
                    inner_fiber: str = "regular",
                    out_fiber: str = None,
                    ) -> nn.SequentialModule:
        self._layer += 1
        logging.info(f"Start building layer {self._layer}")

        downsample = None
        previous_dilation = self.dilation
        if dilate:
            self.dilation *= stride
            stride = 1


        layers = []

        main_type = FIBERS[main_fiber](self.gspace, planes, fixparams=self._fixparams)
        inner_class = FIBERS[inner_fiber](self.gspace, planes, fixparams=self._fixparams)

        out_f = main_type

        # add first block that starts with `self.inplanes` channels and ends with `planes` channels
        # use stride=`stride` for the first block and stride=1 for all the rest (default value)
        first_block = block(in_fiber=self.next_in_type,
                            inner_fiber=inner_class,
                            out_fiber=out_f,
                            stride=stride,
                            # downsample=downsample,
                            groups=self.groups,
                            base_width=self.base_width,
                            dilation=previous_dilation,
                            sigma=self._sigma,
                            F=self._F)
        layers.append(first_block)

        # create new field type with `planes * block.expansion` channels
        self.next_in_type = first_block.out_type
        # TODO: not sure if the number of channels here checks out given `expansion`
        out_f = self.next_in_type
        for _ in range(1, num_blocks-1):
            next_block = block(in_fiber=self.next_in_type,
                               inner_fiber=inner_class,
                               out_fiber=out_f,
                               groups=self.groups,
                               base_width=self.base_width,
                               dilation=self.dilation,
                               sigma=self._sigma,
                               F=self._F)
            layers.append(next_block)
            self.next_in_type = out_f

        # add last block
        if out_fiber is None:
            out_fiber = main_fiber
        out_type = FIBERS[out_fiber](self.gspace, planes, fixparams=self._fixparams)

        last_block = block(in_fiber=self.next_in_type,
                           inner_fiber=inner_class,
                           out_fiber=out_type,
                           groups=self.groups,
                           base_width=self.base_width,
                           dilation=self.dilation,
                           sigma=self._sigma,
                           F=self._F)
        layers.append(last_block)
        self.next_in_type = out_f

        logging.info(f"Built layer {self._layer}")

        return nn.SequentialModule(*layers)

    def _restrict_layer(self, subgroup_id):
        layers = list()
        layers.append(nn.RestrictionModule(self.next_in_type, subgroup_id))
        layers.append(nn.DisentangleModule(layers[-1].out_type))
        self.next_in_type = layers[-1].out_type
        self.gspace = self.next_in_type.gspace

        restrict_layer = nn.SequentialModule(*layers)
        return restrict_layer

    def features(self, x):

        x = nn.GeometricTensor(x, self.in_lifting_type)
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        out = self.maxpool(x)

        x1 = self.layer1(out)

        x2 = self.layer2(self.restrict1(x1))

        x3 = self.layer3(self.restrict2(x2))

        x4 = self.layer4(x3)
        # out = self.relu(self.mp(self.bn1(out)))

        return x1, x2, x3, x4

    def _forward_impl(self, x: Tensor) -> Tensor:
        # This exists since TorchScript doesn't support inheritance, so the superclass method
        # (this one) needs to have a name other than `forward` that can be accessed in a subclass

        x = nn.GeometricTensor(x, self.in_lifting_type)
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(self.restrict1(x))
        x = self.layer3(self.restrict2(x))
        x = self.layer4(x)

        if not self.conv2triv:
            x = self.mp(x)
        x = self.avgpool(x.tensor)
        x = torch.flatten(x, 1)
        x = self.fc(x)

        return x

    def forward(self, x: Tensor) -> Tensor:
        return self._forward_impl(x)

# Note: More architectures can be added here as desired, following resnet.py
def e2_resnet18(**model_args):
    return E2ResNet(block=E2BasicBlock, layers=[2, 2, 2, 2], **model_args)


def e2_resnet50(**model_args):
    return E2ResNet(block=E2Bottleneck, layers=[3, 4, 6, 3], **model_args)


def e2_resnext50(**model_args):
    model_args['groups'] = 32
    model_args['width_per_group'] = 4
    return E2ResNet(block=E2Bottleneck, layers=[3, 4, 6, 3], **model_args)