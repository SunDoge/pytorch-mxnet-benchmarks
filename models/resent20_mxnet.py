from mxnet.gluon import nn


def conv3x3(out_planes, stride=1):
    " 3x3 convolution with padding "
    return nn.Conv2D(out_planes, kernel_size=3, strides=stride, padding=1, use_bias=False)


class BasicBlock(nn.HybridBlock):
    expansion = 1

    def __init__(self, planes, stride=1, downsample=None, **kwargs):
        super().__init__(**kwargs)
        self.conv1 = conv3x3(planes, stride)
        self.bn1 = nn.BatchNorm()
        self.conv2 = conv3x3(planes)
        self.bn2 = nn.BatchNorm()
        self.downsample = downsample
        self.stride = stride

    def hybrid_forward(self, F, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = F.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out = F.relu(residual + out)

        return out


class Bottleneck(nn.HybridBlock):
    expansion = 4

    def __init__(self, planes, stride=1, downsample=None, **kwargs):
        super().__init__(**kwargs)

        self.conv1 = nn.Conv2D(planes, kernel_size=1, use_bias=False)
        self.bn1 = nn.BatchNorm()
        self.conv2 = nn.Conv2D(planes, kernel_size=3,
                               strides=stride, padding=1, use_bias=False)
        self.bn2 = nn.BatchNorm()
        self.conv3 = nn.Conv2D(
            planes, planes * 4, kernel_size=1, use_bias=False)
        self.bn3 = nn.BatchNorm()
        self.downsample = downsample
        self.stride = stride

    def hybrid_forward(self, F, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = F.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out = F.relu(residual + out)

        return out


class ResNet_Cifar(nn.HybridBlock):

    def __init__(self, block, layers, num_classes=10, **kwargs):
        super().__init__(**kwargs)

        self.inplanes = 16
        self.conv1 = nn.Conv2D(
            16, kernel_size=3, strides=1, padding=1, use_bias=False)
        self.bn1 = nn.BatchNorm()
        self.layer1 = self._make_layer(block, 16, layers[0])
        self.layer2 = self._make_layer(block, 32, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 64, layers[2], stride=2)
        self.avgpool = nn.AvgPool2D(8, strides=1)
        self.fc = nn.Dense(num_classes)

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.HybridSequential()
            downsample.add(
                nn.Conv2D(planes * block.expansion,
                          kernel_size=1, strides=stride, use_bias=False)
            )

        layers = nn.HybridSequential()
        layers.add(block(planes, stride, downsample))
        for _ in range(1, blocks):
            layers.add(block(planes))

        return layers

    def hybrid_forward(self, F, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = F.relu(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)

        x = self.avgpool(x)
        x = self.fc(x)

        return x


def resnet20_cifar(**kwargs):
    model = ResNet_Cifar(BasicBlock, [3, 3, 3], **kwargs)
    return model


if __name__ == '__main__':
    import mxnet.ndarray as nd

    x = nd.random.uniform(shape=(2, 3, 32, 32))
    print(x.shape)

    net = resnet20_cifar()
    net.initialize()
    net.hybridize()
    out = net(x)
    print(out.shape)
