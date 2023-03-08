import torch.nn as nn
import torch
from collections import OrderedDict
import torch.nn.init as init




def conv1x1(in_planes, out_planes, stride=1):
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=True)


class Conv1x1_BN(nn.Module):
    def __init__(self, in_planes, out_planes, stride=1, groups=1, dilation=1):
        super(Conv1x1_BN, self).__init__()
        self.conv1x1_bn = nn.Sequential(OrderedDict([('conv', conv1x1(in_planes, out_planes, stride=stride)),
                                                     ('bn', nn.BatchNorm2d(out_planes))]))
        self.branch1x1_bn = nn.Sequential(OrderedDict([('conv', conv1x1(in_planes, out_planes, stride=stride)),
                                                       ('bn', nn.BatchNorm2d(out_planes))]))
        self.use_branch = False

        # self.zero_branch()

    def forward(self, x):
        z1 = self.conv1x1_bn(x)
        z2 = self.branch1x1_bn(x)
        if self.use_branch:
            return z1 + z2
        else:
            return z1

    def fix_conv(self):
        self.conv1x1_bn.eval()

    def fix_branch(self):
        self.branch1x1_bn.eval()

    def set_branch(self,use_branch=True):
        self.use_branch=use_branch

    def re_param(self):
        with torch.no_grad():
            kernel, bias = self.get_equivalent_kernel_bias()
            self.conv1x1_bn.conv.weight.data = kernel
            self.conv1x1_bn.conv.bias.data = bias
            init.ones_(self.conv1x1_bn.bn.weight)
            init.zeros_(self.conv1x1_bn.bn.bias)
            init.ones_(self.conv1x1_bn.bn.running_var)
            init.zeros_(self.conv1x1_bn.bn.running_mean)
            self.zero_branch()

    def get_equivalent_kernel_bias(self):
        conv_kernel1x1, conv_bias1x1 = self.fuse_convbn(self.conv1x1_bn)
        barnch_kernel1x1, branch_bias1x1 = self.fuse_convbn(self.branch1x1_bn)
        return conv_kernel1x1 + barnch_kernel1x1, conv_bias1x1 + branch_bias1x1


    def fuse_convbn(self, branch):
        kernel = branch.conv.weight
        bias = branch.conv.bias
        running_mean = branch.bn.running_mean
        running_var = branch.bn.running_var
        gamma = branch.bn.weight
        beta = branch.bn.bias
        eps = branch.bn.eps
        std = (running_var + eps).sqrt()
        t = (gamma / std).reshape(-1, 1, 1, 1)
        return kernel * t, bias * gamma / std + beta - running_mean * gamma / std

    def zero_branch(self):
        init.zeros_(self.branch1x1_bn.conv.weight)
        init.zeros_(self.branch1x1_bn.conv.bias)
        init.ones_(self.branch1x1_bn.bn.weight)
        init.zeros_(self.branch1x1_bn.bn.bias)
        init.ones_(self.branch1x1_bn.bn.running_var)
        init.zeros_(self.branch1x1_bn.bn.running_mean)


if __name__ == '__main__':

    x = torch.rand([4, 3, 32, 32])
    m = Conv1x1_BN(in_planes=3, out_planes=6)
    m.use_branch=True
    m.fix_conv()
    m.fix_branch()
    # m.zero_branch()
    # m.eval()
    y = m(x)
    print(y[0][0][0])

    m.re_param()
    y2 = m(x)
    print(y2[0][0][0])

    a=nn.BatchNorm2d(3)
    y3=a(x)
    # with torch.no_grad():
    #     y4=a(x)
    # a.track_running_stats=False
    a.eval()
    y4=a(x)
    print(y3[0][0][0])
    print(y4[0][0][0])