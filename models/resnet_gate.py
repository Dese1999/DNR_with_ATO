import torch
import torch.nn as nn
from .imgnet_utils import load_state_dict_from_url
from .gate_function import soft_gate, virtual_gate

__all__ = [
    'ResNet', 'resnet18', 'resnet34', 'my_resnet50', 'resnet101',
    'resnet152', 'resnext50_32x4d', 'resnext101_32x8d',
    'wide_resnet50_2', 'wide_resnet101_2'
]

model_urls = {
    'resnet18': 'https://download.pytorch.org/models/resnet18-5c106cde.pth',
    'resnet34': 'https://download.pytorch.org/models/resnet34-333f7ec4.pth',
    'resnet50': 'https://download.pytorch.org/models/resnet50-19c8e357.pth',
    'resnet101': 'https://download.pytorch.org/models/resnet101-5d3b4d8f.pth',
    'resnet152': 'https://download.pytorch.org/models/resnet152-b121ed2d.pth',
    'resnext50_32x4d': 'https://download.pytorch.org/models/resnext50_32x4d-7cdf4587.pth',
    'resnext101_32x8d': 'https://download.pytorch.org/models/resnext101_32x8d-8ba56ff5.pth',
    'wide_resnet50_2': 'https://download.pytorch.org/models/wide_resnet50_2-95faca4d.pth',
    'wide_resnet101_2': 'https://download.pytorch.org/models/wide_resnet101_2-32ee1156.pth',
}

def conv3x3(in_planes, out_planes, stride=1, groups=1, dilation=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=dilation, groups=groups, bias=False, dilation=dilation)

def conv1x1(in_planes, out_planes, stride=1):
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)

class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None, groups=1,
                 base_width=64, dilation=1, norm_layer=None, cfg=None, num_gate=0):
        super(BasicBlock, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        if groups != 1 or base_width != 64:
            raise ValueError('BasicBlock only supports groups=1 and base_width=64')
        if dilation > 1:
            raise NotImplementedError("Dilation > 1 not supported in BasicBlock")
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = norm_layer(planes)
        self.relu = nn.ReLU(inplace=True)
        if num_gate == 1:
            self.gate = virtual_gate(planes)
        else:
            self.gate = None
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = norm_layer(planes)
        self.downsample = downsample
        self.stride = stride
        self.num_gate = num_gate

    def forward(self, x, cur_mask_vec=None):
        identity = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        if self.gate is not None and cur_mask_vec is not None:
            out = self.gate(out, cur_mask_vec)
        out = self.conv2(out)
        out = self.bn2(out)
        if self.downsample is not None:
            identity = self.downsample(x)
        out += identity
        out = self.relu(out)
        return out

class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None, groups=1,
                 base_width=64, dilation=1, norm_layer=None, cfg=None, num_gate=2):
        super(Bottleneck, self).__init__()
        if cfg is None:
            if norm_layer is None:
                norm_layer = nn.BatchNorm2d
            width = int(planes * (base_width / 64.)) * groups
            self.conv1 = conv1x1(inplanes, width)
            self.bn1 = norm_layer(width)
            if num_gate > 1:
                self.gate1 = virtual_gate(width)
            else:
                self.gate1 = None
            self.conv2 = conv3x3(width, width, stride, groups, dilation)
            self.bn2 = norm_layer(width)
            if num_gate >= 1:
                self.gate2 = virtual_gate(width)
            else:
                self.gate2 = None
            self.conv3 = conv1x1(width, planes * self.expansion)
            self.bn3 = norm_layer(planes * self.expansion)
            self.relu = nn.ReLU(inplace=True)
            self.downsample = downsample
            self.stride = stride
        else:
            if norm_layer is None:
                norm_layer = nn.BatchNorm2d
            self.conv1 = conv1x1(inplanes, cfg[0])
            self.bn1 = norm_layer(cfg[0])
            if num_gate > 1:
                self.gate1 = virtual_gate(cfg[0])
            else:
                self.gate1 = None
            self.conv2 = conv3x3(cfg[0], cfg[1], stride, groups, dilation)
            self.bn2 = norm_layer(cfg[1])
            if num_gate >= 1:
                self.gate2 = virtual_gate(cfg[1])
            else:
                self.gate2 = None
            self.conv3 = conv1x1(cfg[1], planes * self.expansion)
            self.bn3 = norm_layer(planes * self.expansion)
            self.relu = nn.ReLU(inplace=True)
            self.downsample = downsample
            self.stride = stride

    def forward(self, x, cur_mask_vec=None):
        identity = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        if self.gate1 is not None and cur_mask_vec is not None:
            out = self.gate1(out, cur_mask_vec)
        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)
        if self.gate2 is not None and cur_mask_vec is not None:
            out = self.gate2(out, cur_mask_vec)
        out = self.conv3(out)
        out = self.bn3(out)
        if self.downsample is not None:
            identity = self.downsample(x)
        out += identity
        out = self.relu(out)
        return out

class ResNet(nn.Module):
    def __init__(self, block, layers, num_classes=1000, zero_init_residual=False,
                 groups=1, width_per_group=64, replace_stride_with_dilation=None,
                 norm_layer=None, cfg=None, num_gate=2):
        super(ResNet, self).__init__()
        if block is Bottleneck:
            self.factor = 2
            self.block_string = 'Bottleneck'
        elif block is BasicBlock:
            self.factor = 1
            self.block_string = 'BasicBlock'

        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        self._norm_layer = norm_layer

        self.inplanes = 64
        self.dilation = 1
        if replace_stride_with_dilation is None:
            replace_stride_with_dilation = [False, False, False]
        if len(replace_stride_with_dilation) != 3:
            raise ValueError("replace_stride_with_dilation should be None or a 3-element tuple, got {}".format(replace_stride_with_dilation))
        self.groups = groups
        self.base_width = width_per_group
        self.conv1 = nn.Conv2d(3, self.inplanes, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = norm_layer(self.inplanes)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.num_gate = num_gate

        if cfg is None:
            self.layer1 = self._make_layer(block, 64, layers[0])
            self.layer2 = self._make_layer(block, 128, layers[1], stride=2, dilate=replace_stride_with_dilation[0])
            self.layer3 = self._make_layer(block, 256, layers[2], stride=2, dilate=replace_stride_with_dilation[1])
            self.layer4 = self._make_layer(block, 512, layers[3], stride=2, dilate=replace_stride_with_dilation[2])
        else:
            start = 0
            end = int(self.factor * layers[0])
            self.layer1 = self._make_layer(block, 64, layers[0], cfg=cfg[start:end])
            start = end
            end = end + int(self.factor * layers[1])
            self.layer2 = self._make_layer(block, 128, layers[1], stride=2, dilate=replace_stride_with_dilation[0], cfg=cfg[start:end])
            start = end
            end = end + int(self.factor * layers[2])
            self.layer3 = self._make_layer(block, 256, layers[2], stride=2, dilate=replace_stride_with_dilation[1], cfg=cfg[start:end])
            start = end
            end = end + int(self.factor * layers[3])
            self.layer4 = self._make_layer(block, 512, layers[3], stride=2, dilate=replace_stride_with_dilation[2], cfg=cfg[start:end])

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512 * block.expansion, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        if zero_init_residual:
            for m in self.modules():
                if isinstance(m, Bottleneck):
                    nn.init.constant_(m.bn3.weight, 0)
                elif isinstance(m, BasicBlock):
                    nn.init.constant_(m.bn2.weight, 0)

    def _make_layer(self, block, planes, blocks, stride=1, dilate=False, cfg=None):
        norm_layer = self._norm_layer
        downsample = None
        previous_dilation = self.dilation

        if dilate:
            self.dilation *= stride
            stride = 1
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                conv1x1(self.inplanes, planes * block.expansion, stride),
                norm_layer(planes * block.expansion),
            )
        if cfg is None:
            layers = []
            layers.append(block(self.inplanes, planes, stride, downsample, self.groups,
                                self.base_width, previous_dilation, norm_layer, num_gate=self.num_gate))
            self.inplanes = planes * block.expansion
            for _ in range(1, blocks):
                layers.append(block(self.inplanes, planes, groups=self.groups,
                                    base_width=self.base_width, dilation=self.dilation,
                                    norm_layer=norm_layer, num_gate=self.num_gate))
        else:
            index = 0
            layers = []
            layers.append(block(self.inplanes, planes, stride, downsample, self.groups,
                                self.base_width, previous_dilation, norm_layer, cfg=cfg[int(self.factor * index):int(self.factor * index + self.factor)], num_gate=self.num_gate))
            index += 1
            self.inplanes = planes * block.expansion
            for _ in range(1, blocks):
                layers.append(block(self.inplanes, planes, groups=self.groups,
                                    base_width=self.base_width, dilation=self.dilation,
                                    norm_layer=norm_layer, cfg=cfg[int(self.factor * index):int(self.factor * index + self.factor)], num_gate=self.num_gate))
                index += 1
        return nn.Sequential(*layers)

    def forward(self, x, cur_mask_vec=None):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x, cur_mask_vec)
        x = self.layer2(x, cur_mask_vec)
        x = self.layer3(x, cur_mask_vec)
        x = self.layer4(x, cur_mask_vec)

        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        return x

    def count_structure(self):
        structure = []
        for m in self.modules():
            if isinstance(m, virtual_gate):
                structure.append(m.width)
        self.structure = structure
        return sum(structure), structure

    def set_vritual_gate(self, arch_vector):
        i = 0
        start = 0
        for m in self.modules():
            if isinstance(m, virtual_gate):
                end = start + self.structure[i]
                m.gate_f = arch_vector.squeeze()[start:end].to(m.gate_f.device)
                start = end
                i += 1

    def reset_gates(self):
        for m in self.modules():
            if isinstance(m, virtual_gate):
                m.reset_value()

def _resnet(arch, block, layers, pretrained, progress, **kwargs):
    model = ResNet(block, layers, **kwargs)
    if pretrained:
        state_dict = load_state_dict_from_url(model_urls[arch], progress=progress)
        model.load_state_dict(state_dict)
    return model

def resnet18(pretrained=False, progress=True, **kwargs):
    return _resnet('resnet18', BasicBlock, [2, 2, 2, 2], pretrained, progress, **kwargs)

def my_resnet34(pretrained=False, progress=True, **kwargs):
    return _resnet('resnet34', BasicBlock, [3, 4, 6, 3], pretrained, progress, **kwargs)

def resnet34(pretrained=False, progress=True, **kwargs):
    return _resnet('resnet34', BasicBlock, [3, 4, 6, 3], pretrained, progress, **kwargs)

def my_resnet50(pretrained=False, progress=True, **kwargs):
    return _resnet('resnet50', Bottleneck, [3, 4, 6, 3], pretrained, progress, **kwargs)

def resnet101(pretrained=False, progress=True, **kwargs):
    return _resnet('resnet101', Bottleneck, [3, 4, 23, 3], pretrained, progress, **kwargs)

def my_resnet101(pretrained=False, progress=True, **kwargs):
    return _resnet('resnet101', Bottleneck, [3, 4, 23, 3], pretrained, progress, **kwargs)

def resnet152(pretrained=False, progress=True, **kwargs):
    return _resnet('resnet152', Bottleneck, [3, 8, 36, 3], pretrained, progress, **kwargs)

def resnext50_32x4d(pretrained=False, progress=True, **kwargs):
    kwargs['groups'] = 32
    kwargs['width_per_group'] = 4
    return _resnet('resnext50_32x4d', Bottleneck, [3, 4, 6, 3], pretrained, progress, **kwargs)

def resnext101_32x8d(pretrained=False, progress=True, **kwargs):
    kwargs['groups'] = 32
    kwargs['width_per_group'] = 8
    return _resnet('resnext101_32x8d', Bottleneck, [3, 4, 23, 3], pretrained, progress, **kwargs)

def wide_resnet50_2(pretrained=False, progress=True, **kwargs):
    kwargs['width_per_group'] = 64 * 2
    return _resnet('wide_resnet50_2', Bottleneck, [3, 4, 6, 3], pretrained, progress, **kwargs)

def wide_resnet101_2(pretrained=False, progress=True, **kwargs):
    kwargs['width_per_group'] = 64 * 2
    return _resnet('wide_resnet101_2', Bottleneck, [3, 4, 23, 3], pretrained, progress, **kwargs)