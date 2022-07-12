import torch
import torchvision
from torch import Tensor
from torch.autograd import Variable
import torch.nn as nn
import math
import numpy as np
import torch.nn.functional as F
from torch.hub import load_state_dict_from_url
from torch.distributions import Bernoulli
from torch.nn.utils.weight_norm import WeightNorm


def init_layer(L):
    # Initialization using fan-in
    if isinstance(L, nn.Conv2d):
        n = L.kernel_size[0]*L.kernel_size[1]*L.out_channels
        L.weight.data.normal_(0,math.sqrt(2.0/float(n)))
    elif isinstance(L, nn.BatchNorm2d):
        L.weight.data.fill_(1)
        L.bias.data.fill_(0)

class distLinear(nn.Module):
    def __init__(self, indim, outdim):
        super(distLinear, self).__init__()
        self.L = nn.Linear( indim, outdim, bias = False)
        self.class_wise_learnable_norm = True  #See the issue#4&8 in the github
        if self.class_wise_learnable_norm:
            WeightNorm.apply(self.L, 'weight', dim=0) #split the weight update component to direction and norm

        if outdim <=200:
            self.scale_factor = 2; #a fixed scale factor to scale the output of cos value into a reasonably large input for softmax
        else:
            self.scale_factor = 10; #in omniglot, a larger scale factor is required to handle >1000 output classes.

    def forward(self, x):
        x_norm = torch.norm(x, p=2, dim =1).unsqueeze(1).expand_as(x)
        x_normalized = x.div(x_norm+ 0.00001)
        if not self.class_wise_learnable_norm:
            L_norm = torch.norm(self.L.weight.data, p=2, dim =1).unsqueeze(1).expand_as(self.L.weight.data)
            self.L.weight.data = self.L.weight.data.div(L_norm + 0.00001)
        cos_dist = self.L(x_normalized) #matrix product by forward function, but when using WeightNorm, this also multiply the cosine distance by a class-wise learnable norm, see the issue#4&8 in the github
        scores = self.scale_factor* (cos_dist)

        return scores

class Flatten(nn.Module):
    def __init__(self):
        super(Flatten, self).__init__()

    def forward(self, x):
        return x.view(x.size(0), -1)

# For meta-learning based algorithms (task-specific weight)
class Linear_fw(nn.Linear): #used in MAML to forward input with fast weight
    def __init__(self, in_features, out_features):
        super(Linear_fw, self).__init__(in_features, out_features)
        self.weight.fast = None #Lazy hack to add fast weight link
        self.bias.fast = None

    def forward(self, x):
        if self.weight.fast is not None and self.bias.fast is not None:
            out = F.linear(x, self.weight.fast, self.bias.fast) #weight.fast (fast weight) is the temporaily adapted weight
        else:
            out = super(Linear_fw, self).forward(x)
        return out

class Conv2d_fw(nn.Conv2d): #used in MAML to forward input with fast weight
    def __init__(self, in_channels, out_channels, kernel_size, stride=1,padding=0, bias = True):
        super(Conv2d_fw, self).__init__(in_channels, out_channels, kernel_size, stride=stride, padding=padding, bias=bias)
        self.weight.fast = None
        if not self.bias is None:
            self.bias.fast = None

    def forward(self, x):
        if self.bias is None:
            if self.weight.fast is not None:
                out = F.conv2d(x, self.weight.fast, None, stride= self.stride, padding=self.padding)
            else:
                out = super(Conv2d_fw, self).forward(x)
        else:
            if self.weight.fast is not None and self.bias.fast is not None:
                out = F.conv2d(x, self.weight.fast, self.bias.fast, stride= self.stride, padding=self.padding)
            else:
                out = super(Conv2d_fw, self).forward(x)

        return out

class BatchNorm2d_fw(nn.BatchNorm2d): #used in MAML to forward input with fast weight
    def __init__(self, num_features):
        super(BatchNorm2d_fw, self).__init__(num_features)
        self.weight.fast = None
        self.bias.fast = None

    def forward(self, x):
        running_mean = torch.zeros(x.data.size()[1]).cuda()
        running_var = torch.ones(x.data.size()[1]).cuda()
        if self.weight.fast is not None and self.bias.fast is not None:
            out = F.batch_norm(x, running_mean, running_var, self.weight.fast, self.bias.fast, training = True, momentum = 1)
            #batch_norm momentum hack: follow hack of Kate Rakelly in pytorch-maml/src/layers.py
        else:
            out = F.batch_norm(x, running_mean, running_var, self.weight, self.bias, training = True, momentum = 1)
        return out

# Simple ResNet Block
class SimpleBlock(nn.Module):
    def __init__(self, method, indim, outdim, half_res, track_bn):
        super(SimpleBlock, self).__init__()
        self.indim = indim
        self.outdim = outdim
        
        self.C1 = nn.Conv2d(indim, outdim, kernel_size=3, stride=2 if half_res else 1, padding=1, bias=False)
        self.BN1 = nn.BatchNorm2d(outdim, track_running_stats=track_bn)
        self.C2 = nn.Conv2d(outdim, outdim,kernel_size=3, padding=1, bias=False)
        self.BN2 = nn.BatchNorm2d(outdim, track_running_stats=track_bn)
            
        self.relu1 = nn.ReLU(inplace=True)
        self.relu2 = nn.ReLU(inplace=True)

        self.parametrized_layers = [self.C1, self.C2, self.BN1, self.BN2]

        self.half_res = half_res

        # if the input number of channels is not equal to the output, then need a 1x1 convolution
        if indim!=outdim:
            self.shortcut = nn.Conv2d(indim, outdim, 1, 2 if half_res else 1, bias=False)
            self.BNshortcut = nn.BatchNorm2d(outdim, track_running_stats=track_bn)

            self.parametrized_layers.append(self.shortcut)
            self.parametrized_layers.append(self.BNshortcut)
            self.shortcut_type = '1x1'
        else:
            self.shortcut_type = 'identity'

        for layer in self.parametrized_layers:
            init_layer(layer)

    def forward(self, x):
        out = self.C1(x)
        out = self.BN1(out)
        out = self.relu1(out)
        out = self.C2(out)
        out = self.BN2(out)
        short_out = x if self.shortcut_type == 'identity' else self.BNshortcut(self.shortcut(x))
        out = out + short_out
        out = self.relu2(out)
        
        return out

class ResNet(nn.Module):
    def __init__(self, method, block, list_of_num_layers, list_of_out_dims, flatten, track_bn, reinit_bn_stats):
        # list_of_num_layers specifies number of layers in each stage
        # list_of_out_dims specifies number of output channel for each stage
        super(ResNet,self).__init__()
        assert len(list_of_num_layers)==4, 'Can have only four stages'

        self.reinit_bn_stats = reinit_bn_stats
        
        conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        bn1 = nn.BatchNorm2d(64, track_running_stats=track_bn)

        relu = nn.ReLU()
        pool1 = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        init_layer(conv1)
        init_layer(bn1)

        trunk = [conv1, bn1, relu, pool1]

        indim = 64
        for i in range(4):
            for j in range(list_of_num_layers[i]):
                half_res = (i>=1) and (j==0)
                B = block(method, indim, list_of_out_dims[i], half_res, track_bn)
                trunk.append(B)
                indim = list_of_out_dims[i]

        if flatten:
            avgpool = nn.AvgPool2d(7)
            trunk.append(avgpool)
            trunk.append(Flatten())
            self.final_feat_dim = indim
        else:
            self.final_feat_dim = [indim, 7, 7]

        self.trunk = nn.Sequential(*trunk)

    def forward(self,x):
        if self.reinit_bn_stats:
            self._reinit_running_batch_statistics()
        out = self.trunk(x)
        return out
        
    def _reinit_running_batch_statistics(self):
        with torch.no_grad():
            self.trunk[1].running_mean.data.fill_(0.)
            self.trunk[1].running_var.data.fill_(1.)

            self.trunk[4].BN1.running_mean.data.fill_(0.)
            self.trunk[4].BN1.running_var.data.fill_(1.)
            self.trunk[4].BN2.running_mean.data.fill_(0.)
            self.trunk[4].BN2.running_var.data.fill_(1.)

            self.trunk[5].BN1.running_mean.data.fill_(0.)
            self.trunk[5].BN1.running_var.data.fill_(1.)
            self.trunk[5].BN2.running_mean.data.fill_(0.)
            self.trunk[5].BN2.running_var.data.fill_(1.)
            self.trunk[5].BNshortcut.running_mean.data.fill_(0.)
            self.trunk[5].BNshortcut.running_var.data.fill_(1.)

            self.trunk[6].BN1.running_mean.data.fill_(0.)
            self.trunk[6].BN1.running_var.data.fill_(1.)
            self.trunk[6].BN2.running_mean.data.fill_(0.)
            self.trunk[6].BN2.running_var.data.fill_(1.)
            self.trunk[6].BNshortcut.running_mean.data.fill_(0.)
            self.trunk[6].BNshortcut.running_var.data.fill_(1.)

            self.trunk[7].BN1.running_mean.data.fill_(0.)
            self.trunk[7].BN1.running_var.data.fill_(1.)
            self.trunk[7].BN2.running_mean.data.fill_(0.)
            self.trunk[7].BN2.running_var.data.fill_(1.)
            self.trunk[7].BNshortcut.running_mean.data.fill_(0.)
            self.trunk[7].BNshortcut.running_var.data.fill_(1.)


    def return_features(self, x, return_avg=False):
        flat = Flatten()
        m = nn.AdaptiveAvgPool2d((1,1))

        with torch.no_grad():
            block1_out = self.trunk[4](self.trunk[3](self.trunk[2](self.trunk[1](self.trunk[0](x)))))
            block2_out = self.trunk[5](block1_out)
            block3_out = self.trunk[6](block2_out)
            block4_out = self.trunk[7](block3_out)
            
        if return_avg:
            return flat(m(block1_out)), flat(m(block2_out)), flat(m(block3_out)), flat(m(block4_out))
        else:
            return flat(block1_out), flat(block2_out), flat(block3_out), flat(block4_out)
    
    def forward_bodyfreeze(self,x):
        flat = Flatten()
        m = nn.AdaptiveAvgPool2d((1,1))

        with torch.no_grad():
            block1_out = self.trunk[4](self.trunk[3](self.trunk[2](self.trunk[1](self.trunk[0](x)))))
            block2_out = self.trunk[5](block1_out)
            block3_out = self.trunk[6](block2_out)
            
            out = self.trunk[7].C1(block3_out)
            out = self.trunk[7].BN1(out)
            out = self.trunk[7].relu1(out)
        
        out = self.trunk[7].C2(out)
        out = self.trunk[7].BN2(out)
        short_out = self.trunk[7].BNshortcut(self.trunk[7].shortcut(block3_out))
        out = out + short_out
        out = self.trunk[7].relu2(out)
        
        return flat(m(out))

def ResNet10(method='baseline', track_bn=True, reinit_bn_stats=False):
    return ResNet(method, block=SimpleBlock, list_of_num_layers=[1,1,1,1], list_of_out_dims=[64,128,256,512], flatten=True, track_bn=track_bn, reinit_bn_stats=reinit_bn_stats)

# -*- coding: utf-8 -*-
# https://github.com/ElementAI/embedding-propagation/blob/master/src/models/backbones/resnet12.py

class Block(torch.nn.Module):
    def __init__(self, ni, no, stride, dropout, track_bn, reinit_bn_stats):
        super().__init__()
        self.reinit_bn_stats = reinit_bn_stats
        
        self.dropout = nn.Dropout2d(dropout) if dropout > 0 else lambda x: x
        self.C0 = nn.Conv2d(ni, no, 3, stride, padding=1, bias=False)
        self.BN0 = nn.BatchNorm2d(no, track_running_stats=track_bn)
        self.C1 = nn.Conv2d(no, no, 3, 1, padding=1, bias=False)
        self.BN1 = nn.BatchNorm2d(no, track_running_stats=track_bn)
        self.C2 = nn.Conv2d(no, no, 3, 1, padding=1, bias=False)
        self.BN2 = nn.BatchNorm2d(no, track_running_stats=track_bn)
        if stride == 2 or ni != no:
            self.shortcut = nn.Conv2d(ni, no, 1, stride=1, padding=0, bias=False)
            self.BNshortcut = nn.BatchNorm2d(no, track_running_stats=track_bn)

    def get_parameters(self):
        return self.parameters()

    def forward(self, x):
        if self.reinit_bn_stats:
            self._reinit_running_batch_statistics()
        
        out = self.C0(x)
        out = self.BN0(out)
        out = F.relu(out)
        out = self.dropout(out)
        out = self.C1(out)
        out = self.BN1(out)
        out = F.relu(out)
        out = self.dropout(out)
        out = self.C2(out)
        out = self.BN2(out)
        out += self.BNshortcut(self.shortcut(x))
        out = F.relu(out)
        
        return out
    
    def _reinit_running_batch_statistics(self):
        with torch.no_grad():
            self.BN0.running_mean.data.fill_(0.)
            self.BN0.running_var.data.fill_(1.)
            self.BN1.running_mean.data.fill_(0.)
            self.BN1.running_var.data.fill_(1.)
            self.BN2.running_mean.data.fill_(0.)
            self.BN2.running_var.data.fill_(1.)
            self.BNshortcut.running_mean.data.fill_(0.)
            self.BNshortcut.running_var.data.fill_(1.)

class ResNet12(torch.nn.Module):
    def __init__(self, track_bn, reinit_bn_stats, width=1, dropout=0):
        super().__init__()
        self.final_feat_dim = 512
        assert(width == 1) # Comment for different variants of this model
        self.widths = [x * int(width) for x in [64, 128, 256]]
        self.widths.append(self.final_feat_dim * width)
        # self.bn_out = nn.BatchNorm1d(self.final_feat_dim)

        start_width = 3
        for i in range(len(self.widths)):
            setattr(self, "group_%d" %i, Block(start_width, self.widths[i], 1, dropout, track_bn, reinit_bn_stats))
            start_width = self.widths[i]

    def add_classifier(self, nclasses, name="classifier", modalities=None):
        setattr(self, name, torch.nn.Linear(self.final_feat_dim, nclasses))

    def up_to_embedding(self, x):
        """ Applies the four residual groups
        Args:
            x: input images
            n: number of few-shot classes
            k: number of images per few-shot class
        """
        for i in range(len(self.widths)):
            x = getattr(self, "group_%d" % i)(x)
            x = F.max_pool2d(x, 3, 2, 1)
        return x

    def forward(self, x):
        """Main Pytorch forward function
        Returns: class logits
        Args:
            x: input mages
        """
        *args, c, h, w = x.size()
        x = x.view(-1, c, h, w)
        x = self.up_to_embedding(x)
        # return F.relu(self.bn_out(x.mean(3).mean(2)), True)
        return F.relu(x.mean(3).mean(2), True)


class ResNet18(torchvision.models.resnet.ResNet):
    def __init__(self, track_bn=True):
        def norm_layer(*args, **kwargs):
            return nn.BatchNorm2d(*args, **kwargs, track_running_stats=track_bn)
        super().__init__(torchvision.models.resnet.BasicBlock, [2, 2, 2, 2], norm_layer=norm_layer)
        del self.fc
        self.final_feat_dim = 512

    def load_imagenet_weights(self, progress=True):
        state_dict = load_state_dict_from_url(torchvision.models.resnet.model_urls['resnet18'],
                                              progress=progress)
        missing, unexpected = self.load_state_dict(state_dict, strict=False)
        if len(missing) > 0:
            raise AssertionError('Model code may be incorrect')

    def _forward_impl(self, x: Tensor) -> Tensor:
        # See note [TorchScript super()]
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        # x = self.fc(x)

        return x



##########################################################################################################
# code from https://github.com/WangYueFt/rfs/blob/f8c837ba93c62dd0ac68a2f4019c619aa86b8421/models/resnet.py#L88

def conv3x3(in_planes, out_planes, stride=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)


class SELayer(nn.Module):
    def __init__(self, channel, reduction=16):
        super(SELayer, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
                nn.Linear(channel, channel // reduction),
                nn.ReLU(inplace=True),
                nn.Linear(channel // reduction, channel),
                nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y


class DropBlock(nn.Module):
    def __init__(self, block_size):
        super(DropBlock, self).__init__()

        self.block_size = block_size
        #self.gamma = gamma
        #self.bernouli = Bernoulli(gamma)

    def forward(self, x, gamma):
        # shape: (bsize, channels, height, width)

        if self.training:
            batch_size, channels, height, width = x.shape
            
            bernoulli = Bernoulli(gamma)
            mask = bernoulli.sample((batch_size, channels, height - (self.block_size - 1), width - (self.block_size - 1))).cuda()
            block_mask = self._compute_block_mask(mask)
            countM = block_mask.size()[0] * block_mask.size()[1] * block_mask.size()[2] * block_mask.size()[3]
            count_ones = block_mask.sum()

            return block_mask * x * (countM / count_ones)
        else:
            return x

    def _compute_block_mask(self, mask):
        left_padding = int((self.block_size-1) / 2)
        right_padding = int(self.block_size / 2)
        
        batch_size, channels, height, width = mask.shape
        #print ("mask", mask[0][0])
        non_zero_idxs = mask.nonzero()
        nr_blocks = non_zero_idxs.shape[0]

        offsets = torch.stack(
            [
                torch.arange(self.block_size).view(-1, 1).expand(self.block_size, self.block_size).reshape(-1), # - left_padding,
                torch.arange(self.block_size).repeat(self.block_size), #- left_padding
            ]
        ).t().cuda()
        offsets = torch.cat((torch.zeros(self.block_size**2, 2).cuda().long(), offsets.long()), 1)
        
        if nr_blocks > 0:
            non_zero_idxs = non_zero_idxs.repeat(self.block_size ** 2, 1)
            offsets = offsets.repeat(nr_blocks, 1).view(-1, 4)
            offsets = offsets.long()

            block_idxs = non_zero_idxs + offsets
            #block_idxs += left_padding
            padded_mask = F.pad(mask, (left_padding, right_padding, left_padding, right_padding))
            padded_mask[block_idxs[:, 0], block_idxs[:, 1], block_idxs[:, 2], block_idxs[:, 3]] = 1.
        else:
            padded_mask = F.pad(mask, (left_padding, right_padding, left_padding, right_padding))
            
        block_mask = 1 - padded_mask#[:height, :width]
        return block_mask


class BasicBlock(torch.nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None, drop_rate=0.0, drop_block=False,
                 block_size=1, use_se=False):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(inplanes, planes)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.LeakyReLU(0.1)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = conv3x3(planes, planes)
        self.bn3 = nn.BatchNorm2d(planes)
        self.maxpool = nn.MaxPool2d(stride)
        self.downsample = downsample
        self.stride = stride
        self.drop_rate = drop_rate
        self.num_batches_tracked = 0
        self.drop_block = drop_block
        self.block_size = block_size
        self.DropBlock = DropBlock(block_size=self.block_size)
        self.use_se = use_se
        if self.use_se:
            self.se = SELayer(planes, 4)

    def forward(self, x):
        self.num_batches_tracked += 1

        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)
        if self.use_se:
            out = self.se(out)

        if self.downsample is not None:
            residual = self.downsample(x)
        out += residual
        out = self.relu(out)
        out = self.maxpool(out)

        if self.drop_rate > 0:
            if self.drop_block == True:
                feat_size = out.size()[2]
                keep_rate = max(1.0 - self.drop_rate / (20*2000) * (self.num_batches_tracked), 1.0 - self.drop_rate)
                gamma = (1 - keep_rate) / self.block_size**2 * feat_size**2 / (feat_size - self.block_size + 1)**2
                out = self.DropBlock(out, gamma=gamma)
            else:
                out = F.dropout(out, p=self.drop_rate, training=self.training, inplace=True)

        return out


class ResNet18_84x84(torch.nn.Module):
    def __init__(self, track_bn=True, block=BasicBlock, n_blocks=[1,1,2,2], keep_prob=1.0, avg_pool=True, drop_rate=0.1,
                 dropblock_size=5, num_classes=-1, use_se=False):
        super(ResNet18_84x84, self).__init__()
        self.final_feat_dim = 640

        self.inplanes = 3
        self.use_se = use_se
        self.layer1 = self._make_layer(block, n_blocks[0], 64,
                                       stride=2, drop_rate=drop_rate)
        self.layer2 = self._make_layer(block, n_blocks[1], 160,
                                       stride=2, drop_rate=drop_rate)
        self.layer3 = self._make_layer(block, n_blocks[2], 320,
                                       stride=2, drop_rate=drop_rate, drop_block=True, block_size=dropblock_size)
        self.layer4 = self._make_layer(block, n_blocks[3], 640,
                                       stride=2, drop_rate=drop_rate, drop_block=True, block_size=dropblock_size)
        if avg_pool:
            # self.avgpool = nn.AvgPool2d(5, stride=1)
            self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.keep_prob = keep_prob
        self.keep_avg_pool = avg_pool
        self.dropout = nn.Dropout(p=1 - self.keep_prob, inplace=False)
        self.drop_rate = drop_rate
        
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='leaky_relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
                if not track_bn:
                    m.track_running_stats = False

        # self.num_classes = num_classes
        # if self.num_classes > 0:
        #     self.classifier = nn.Linear(640, self.num_classes)

    def _make_layer(self, block, n_block, planes, stride=1, drop_rate=0.0, drop_block=False, block_size=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=1, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        if n_block == 1:
            layer = block(self.inplanes, planes, stride, downsample, drop_rate, drop_block, block_size, self.use_se)
        else:
            layer = block(self.inplanes, planes, stride, downsample, drop_rate, self.use_se)
        layers.append(layer)
        self.inplanes = planes * block.expansion

        for i in range(1, n_block):
            if i == n_block - 1:
                layer = block(self.inplanes, planes, drop_rate=drop_rate, drop_block=drop_block,
                              block_size=block_size, use_se=self.use_se)
            else:
                layer = block(self.inplanes, planes, drop_rate=drop_rate, use_se=self.use_se)
            layers.append(layer)

        return nn.Sequential(*layers)

    def forward(self, x, is_feat=False):
        x = self.layer1(x)
        # f0 = x
        x = self.layer2(x)
        # f1 = x
        x = self.layer3(x)
        # f2 = x
        x = self.layer4(x)
        # f3 = x
        if self.keep_avg_pool:
            x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        # feat = x
        # if self.num_classes > 0:
        #     x = self.classifier(x)

        # if is_feat:
        #     return [f0, f1, f2, f3, feat], x
        # else:
        #     return x
        return x


def Torch_ResNet18():
    resnet18 = torchvision.models.resnet18(pretrained=True)
    modules=list(resnet18.children())[:-1]
    resnet18=nn.Sequential(*modules)
    for p in resnet18.parameters():
        p.requires_grad = True
    return resnet18

def Torch_ResNet34():
    resnet34 = torchvision.models.resnet34(pretrained=True)
    modules=list(resnet34.children())[:-1]
    resnet34=nn.Sequential(*modules)
    for p in resnet34.parameters():
        p.requires_grad = True
    return resnet34

def Torch_ResNet50():
    resnet50 = torchvision.models.resnet50(pretrained=True)
    modules=list(resnet50.children())[:-1]
    resnet50=nn.Sequential(*modules)
    for p in resnet50.parameters():
        p.requires_grad = True
    return resnet50

def Torch_ResNet101():
    resnet101 = torchvision.models.resnet101(pretrained=True)
    modules=list(resnet101.children())[:-1]
    resnet101=nn.Sequential(*modules)
    for p in resnet101.parameters():
        p.requires_grad = True
    return resnet101

def Torch_ResNet152():
    resnet152 = torchvision.models.resnet152(pretrained=True)
    modules=list(resnet152.children())[:-1]
    resnet152=nn.Sequential(*modules)
    for p in resnet152.parameters():
        p.requires_grad = True
    return resnet152


_backbone_class_map = {
    'resnet10': ResNet10,
    'resnet18': ResNet18,
    'torch_resnet18' : Torch_ResNet18,
    'torch_resnet34' : Torch_ResNet34,
    'torch_resnet50' : Torch_ResNet50,
    'torch_resnet101' : Torch_ResNet101,
    'torch_resnet152' : Torch_ResNet152
}


def get_backbone_class(key):
    if key in _backbone_class_map:
        return _backbone_class_map[key]
    else:
        raise ValueError('Invalid backbone: {}'.format(key))