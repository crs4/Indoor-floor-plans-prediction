import torch
import torch.nn as nn
import math
import torch.utils.model_zoo as model_zoo

from .gated_modules import *


def conv3x3(in_planes, out_planes, stride=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride, padding=1, bias=False)


class GatedBasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(GatedBasicBlock, self).__init__()
        self.conv1 = GatedConv2d(inplanes, planes, stride=stride, kernel_size=3, padding=1, pad_type = 'spherical', activation = 'elu')
        ##self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = GatedConv2d(planes, planes, kernel_size=3, stride=1, padding=1, activation = 'none')
        ##self.bn2 = nn.BatchNorm2d(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        ##out = self.bn1(out)
        ##out = self.relu(out)

        out = self.conv2(out)
        ##out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)
                    
        out += residual
        out = self.relu(out)

        return out

class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)
                    
        out += residual
        out = self.relu(out)

        return out

class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride,
                               padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, planes * 4, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * 4)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        
        out += residual
        out = self.relu(out)

        return out


class ResNet(nn.Module):

    def __init__(self, block, layers, input_channels = 3, num_classes=1000, st = 2):
        self.inplanes = 64
        super(ResNet, self).__init__()
        self.conv1 = nn.Conv2d(input_channels, 64, kernel_size=7, stride=st, padding=3,
                               bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=st, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=st)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=st)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=st)
        #self.avgpool = nn.AvgPool2d(7, stride=1)
        #self.fc = nn.Linear(512 * block.expansion, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)
        

    def forward(self, x):
        features = []
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x);  features.append(x)  # 1/4
        x = self.layer2(x);  features.append(x)  # 1/8
        x = self.layer3(x);  features.append(x)  # 1/16
        x = self.layer4(x);  features.append(x)  # 1/32

        return features

class GatedResNet(nn.Module):

    def __init__(self, block, layers, input_channels = 3, num_classes=1000, inplanes = 64):
        self.inplanes = inplanes
        super(GatedResNet, self).__init__()

        ##self.conv1 = nn.Conv2d(input_channels, 64, kernel_size=7, stride=2, padding=3, bias=False)### TO DO
        self.conv1 = GatedConv2d(input_channels, inplanes, kernel_size=7, stride=2, padding=3, pad_type = 'spherical', activation = 'elu')

        ##self.bn1 = nn.BatchNorm2d(64)
        ##self.relu = nn.ReLU(inplace=True)
        ##self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        
        self.layer1 = self._make_layer(block, 64, layers[0], stride=2)
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)
        #self.avgpool = nn.AvgPool2d(7, stride=1)
        #self.fc = nn.Linear(512 * block.expansion, num_classes)

        ###TO DO adapt init
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        ##print('planes',self.inplanes, planes * block.expansion)
        if stride != 1 or self.inplanes != planes * block.expansion:
            #downsample = nn.Sequential(
            #    nn.Conv2d(self.inplanes, planes * block.expansion,
            #              kernel_size=1, stride=stride, bias=False),
            #    nn.BatchNorm2d(planes * block.expansion),
            #)

            downsample = GatedConv2d(self.inplanes, planes * block.expansion, kernel_size=1, stride=2, padding=0, activation = 'none')

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)
        

    def forward(self, x):
        features = []
        x = self.conv1(x)
        ##x = self.bn1(x)
        ##x = self.relu(x)
        ##x = self.maxpool(x)
              

        x = self.layer1(x);  features.append(x)  # 1/4
        x = self.layer2(x);  features.append(x)  # 1/8
        x = self.layer3(x);  features.append(x)  # 1/16
        x = self.layer4(x);  features.append(x)  # 1/32

        
        return features
            

def resnet18(pretrained=False, **kwargs):
    """Constructs a ResNet-18 model.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet(BasicBlock, [2, 2, 2, 2], **kwargs)
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls['resnet18']))
    return model

def resnet18_single_channel(pretrained=False, **kwargs):
    """Constructs a ResNet-18 model.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet(BasicBlock, [2, 2, 2, 2], input_channels=1,**kwargs)

    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls['resnet18']))

    return model

def resnet18_channels(pretrained=False, input_channels=4, **kwargs):
    """Constructs a ResNet-18 model.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet(BasicBlock, [2, 2, 2, 2], input_channels=input_channels,**kwargs)

    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls['resnet18']))

    return model

def resnet18_channels_nored(pretrained=False, input_channels=3, **kwargs):
    """Constructs a ResNet-18 model.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet(BasicBlock, [1, 1, 1, 1], input_channels=input_channels,**kwargs, st = 1)

    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls['resnet18']))

    return model

def resnet18_single_channel_gated(inplanes = 64, **kwargs):
    """Constructs a ResNet-18 model.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = GatedResNet(GatedBasicBlock, [2, 2, 2, 2], input_channels=2, inplanes = inplanes, **kwargs)
        
    return model

def resnet18_rgbs_channel_gated(inplanes = 64, input_channels=4, **kwargs):
    """Constructs a ResNet-18 model.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = GatedResNet(GatedBasicBlock, [2, 2, 2, 2], input_channels=input_channels, inplanes = inplanes, **kwargs)
        
    return model

def resnet34_rgbs_channel_gated(inplanes = 64, input_channels=4,**kwargs):
    """Constructs a ResNet-18 model.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = GatedResNet(GatedBasicBlock, [3, 4, 6, 3], input_channels=input_channels, inplanes = inplanes, **kwargs)
        
    return model

def resnet34_channels(input_channels=3,**kwargs):
    """Constructs a ResNet-18 model.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet(BasicBlock, [3, 4, 6, 3], input_channels=input_channels, **kwargs)
        
    return model


def resnet34(pretrained=False, **kwargs):
    """Constructs a ResNet-34 model.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet(BasicBlock, [3, 4, 6, 3], **kwargs)
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls['resnet34']))
    return model


def resnet50(pretrained=False, **kwargs):
    """Constructs a ResNet-50 model.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet(Bottleneck, [3, 4, 6, 3], **kwargs)
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls['resnet50']))
    return model


def resnet101(pretrained=False, **kwargs):
    """Constructs a ResNet-101 model.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet(Bottleneck, [3, 4, 23, 3], **kwargs)
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls['resnet101']))
    return model


def resnet152(pretrained=False, **kwargs):
    """Constructs a ResNet-152 model.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet(Bottleneck, [3, 8, 36, 3], **kwargs)
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls['resnet152']))
    return model


if __name__ == '__main__':
    device = torch.device('cuda')

    ##net = resnet18_rgbs_channel_gated().to(device)
    net = resnet18_channels_nored().to(device)
    ##net = resnet18_single_channel().to(device)


    pytorch_total_params = sum(p.numel() for p in net.parameters())

   
    print('pytorch_total_params', pytorch_total_params)

    pytorch_trainable_params = sum(p.numel() for p in net.parameters() if p.requires_grad)

    print('pytorch_trainable_params', pytorch_trainable_params)

    ##batch = torch.ones(1, 2, 256, 512).to(device)
    batch = torch.ones(2, 3, 512, 1024).to(device)


    with torch.autograd.profiler.profile(use_cuda=True) as prof:
        out_depth = net(batch)   
    print(prof.key_averages().table(sort_by='cuda_time_total'))
                       
    print('out_depth shape', out_depth[0].shape,out_depth[1].shape, out_depth[2].shape, out_depth[3].shape)
   
    ##print(net)

