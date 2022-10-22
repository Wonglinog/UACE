from typing import Any, Callable, List, Optional, Type, Union

import torch
import torch.nn as nn
import torchvision
from torchvision.models.resnet import BasicBlock, Bottleneck
from torch.nn import functional as F
from uace.utils import truncated_normal
from torchvision.models.resnet import resnet50


class MLP(nn.Module):
    def __init__(self, num_inputs=28*28, num_hiddens=256, num_classes=10):
        super(MLP, self).__init__()
        self.num_inputs = num_inputs
        self.fc = nn.Sequential(
            nn.Linear(num_inputs, num_hiddens),
            nn.ReLU(inplace=True),
            nn.Linear(num_hiddens, num_classes)
        )

    def forward(self, x):
        x = x.view(-1, self.num_inputs)
        x = self.fc(x)
        return x

        
class ConvBrunch(nn.Module):
    def __init__(self, in_planes, out_planes, kernel_size=3):
        super(ConvBrunch, self).__init__()
        padding = (kernel_size - 1) // 2
        self.out_conv = nn.Sequential(
            nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size, padding=padding),
            nn.BatchNorm2d(out_planes),
            nn.ReLU())

    def forward(self, x):
        return self.out_conv(x)


class ToyModel(nn.Module):
    def __init__(self, type='CIFAR10'):
        super(ToyModel, self).__init__()
        self.type = type
        if type == 'CIFAR10':
            self.block1 = nn.Sequential(
                ConvBrunch(3, 64, 3),
                ConvBrunch(64, 64, 3),
                nn.MaxPool2d(kernel_size=2, stride=2))
            self.block2 = nn.Sequential(
                ConvBrunch(64, 128, 3),
                ConvBrunch(128, 128, 3),
                nn.MaxPool2d(kernel_size=2, stride=2))
            self.block3 = nn.Sequential(
                ConvBrunch(128, 196, 3),
                ConvBrunch(196, 196, 3),
                nn.MaxPool2d(kernel_size=2, stride=2))
            # self.global_avg_pool = nn.AdaptiveAvgPool2d(1)
            self.fc1 = nn.Sequential(
                nn.Linear(4*4*196, 256),
                nn.BatchNorm1d(256),
                nn.ReLU())
            self.fc2 = nn.Linear(256, 10)
            self.fc_size = 4*4*196
        elif type == 'MNIST':
            self.block1 = nn.Sequential(
                ConvBrunch(1, 32, 3),
                nn.MaxPool2d(kernel_size=2, stride=2))
            self.block2 = nn.Sequential(
                ConvBrunch(32, 64, 3),
                nn.MaxPool2d(kernel_size=2, stride=2))
            # self.global_avg_pool = nn.AdaptiveAvgPool2d(1)
            self.fc1 = nn.Sequential(
                nn.Linear(64*7*7, 128),
                nn.BatchNorm1d(128),
                nn.ReLU())
            self.fc2 = nn.Linear(128, 10)
            self.fc_size = 64*7*7
        self._reset_prams()

    def _reset_prams(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_uniform_(m.weight, mode='fan_in', nonlinearity='relu')
            elif isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
        return

    def forward(self, x):
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x) if self.type == 'CIFAR10' else x
        # x = self.global_avg_pool(x)
        # x = x.view(x.shape[0], -1)
        x = x.view(-1, self.fc_size)
        x = self.fc1(x)
        x = self.fc2(x)
        return x






def Layer_8(**kwargs):
    return ToyModel(type='CIFAR10')

def Layer_4(**kwargs):
    return ToyModel(type='MNIST')


class LeNet(nn.Module):
    """LeNet-5 from `"Gradient-Based Learning Applied To Document Recognition"
    <http://yann.lecun.com/exdb/publis/pdf/lecun-98.pdf>`_
    """

    def __init__(self) -> None:
        super().__init__()

        self.conv1 = nn.Conv2d(1, 6, kernel_size=5, padding=2)
        self.conv2 = nn.Conv2d(6, 16, kernel_size=5)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

        self.relu = nn.ReLU()
        self.avgpool = nn.AvgPool2d(kernel_size=2)
        self._init_weights()

    # ref: https://discuss.pytorch.org/t/implementing-truncated-normal-initializer/4778
    def _init_weights(self) -> None:
        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
                # truncated normal distribution with std 0.1 (truncate > 2 x std)
                # https://www.tensorflow.org/api_docs/python/tf/random/truncated_normal
                weights = truncated_normal(list(m.weight.shape), threshold=0.1 * 2)
                weights = torch.from_numpy(weights)
                m.weight.data.copy_(weights)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.avgpool(self.relu(self.conv1(x)))
        x = self.avgpool(self.relu(self.conv2(x)))
        x = torch.flatten(x, start_dim=1)
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.fc3(x)
        return x,x



class ConvNet(nn.Module):
    """LeNet++ as described in the Center Loss paper."""
    def __init__(self, num_classes=10):
        super(ConvNet, self).__init__()
        self.conv1_1 = nn.Conv2d(1, 32, 5, stride=1, padding=2)
        self.prelu1_1 = nn.PReLU()
        self.conv1_2 = nn.Conv2d(32, 32, 5, stride=1, padding=2)
        self.prelu1_2 = nn.PReLU()
        
        self.conv2_1 = nn.Conv2d(32, 64, 5, stride=1, padding=2)
        self.prelu2_1 = nn.PReLU()
        self.conv2_2 = nn.Conv2d(64, 64, 5, stride=1, padding=2)
        self.prelu2_2 = nn.PReLU()
        
        self.conv3_1 = nn.Conv2d(64, 128, 5, stride=1, padding=2)
        self.prelu3_1 = nn.PReLU()
        self.conv3_2 = nn.Conv2d(128, 128, 5, stride=1, padding=2)
        self.prelu3_2 = nn.PReLU()
        
        self.fc1 = nn.Linear(128*3*3, 2)
        self.prelu_fc1 = nn.PReLU()
        self.fc2 = nn.Linear(2, num_classes)

    def forward(self, x):
        x = self.prelu1_1(self.conv1_1(x))
        x = self.prelu1_2(self.conv1_2(x))
        x = F.max_pool2d(x, 2)
        
        x = self.prelu2_1(self.conv2_1(x))
        x = self.prelu2_2(self.conv2_2(x))
        x = F.max_pool2d(x, 2)
        
        x = self.prelu3_1(self.conv3_1(x))
        x = self.prelu3_2(self.conv3_2(x))
        x = F.max_pool2d(x, 2)
        
        x = x.view(-1, 128*3*3)
        x = self.prelu_fc1(self.fc1(x))
        y = self.fc2(x)

        return y,x







class ResNet(torchvision.models.ResNet):
    """Modifies `torchvision's ResNet implementation
    <https://pytorch.org/docs/stable/_modules/torchvision/models/resnet.html>`_
    to make it suitable for CIFAR 10/100.

    Removes or replaces some down-sampling layers to increase the size of the feature
    maps, in order to make it suitable for classification tasks on datasets with smaller
    images such as CIFAR 10/100.

    This network architecture is similar to the one used in
    `"Improved Regularization of Convolutional Neural Networks with Cutout"
    <https://arxiv.org/pdf/1708.04552.pdf>`_
    (code `here <https://github.com/uoguelph-mlrg/Cutout>`_) and in the popular
    `pytorch-cifar repository <https://github.com/kuangliu/pytorch-cifar>`_.
    """

    def __init__(
        self,
        block: Type[Union[BasicBlock, Bottleneck]],
        layers: List[int],
        num_classes: int = 10,
        zero_init_residual: bool = False,
        groups: int = 1,
        width_per_group: int = 64,
        replace_stride_with_dilation: Optional[List[bool]] = None,
        norm_layer: Optional[Callable[..., nn.Module]] = None,
    ) -> None:
        super().__init__(
            block,
            layers,
            num_classes,
            zero_init_residual,
            groups,
            width_per_group,
            replace_stride_with_dilation,
            norm_layer,
        )
        # CIFAR: kernel_size 7 -> 3, stride 2 -> 1, padding 3->1
        self.conv1_planes = 64
        self.conv1 = nn.Conv2d(
            3, self.conv1_planes, kernel_size=3, stride=1, padding=1, bias=False
        )
        # Remove maxpool layer from forward by changing it into an identity layer
        self.maxpool = nn.Identity()


def resnet18(**kwargs: Any) -> ResNet:
    r"""ResNet-18 model from
    `"Deep Residual Learning for Image Recognition"
    <https://arxiv.org/pdf/1512.03385.pdf>`_, modified for CIFAR-10/100 images.

    Args:
       **kwargs: Keyword arguments, notably num_classes for the number of classes
    """
    return ResNet(BasicBlock, [2, 2, 2, 2], **kwargs)


def resnet34(**kwargs: Any) -> ResNet:
    r"""ResNet-18 model from
    `"Deep Residual Learning for Image Recognition"
    <https://arxiv.org/pdf/1512.03385.pdf>`_, modified for CIFAR-10/100 images.

    Args:
       **kwargs: Keyword arguments, notably num_classes for the number of classes
    """
    return ResNet(BasicBlock, [3, 4, 6, 3], **kwargs)


def resnet50_clothing1m():
    net = resnet50(pretrained=True)
    net.fc = nn.Linear(2048,14)
    return net 



if __name__ == '__main__':
    net = resnet50_clothing1m()
    print("net",net)
    y = net(torch.randn(4,3,224,224))# mnist 4,1,28,28/ cifar 4,3,32,32
    print(y)