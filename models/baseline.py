from torch import nn
import torch
from models.encoder.vgg import vgg
from models.encoder.resnet import resnet as resnet50
#from models.encoder.res import resnet50
from models.encoder.mobile import mobilenet
from .baseU import baseU
from torch.nn import functional as F
#from .adapter import vgg_adapter, resnet_adapter, mobile_adapter

res_inc = [64, 256, 512, 1024, 2048]
vgg_inc = [64, 128, 256, 512, 512]
mobile_inc = [16, 24, 32, 64, 160]


class vgg_adapter(nn.Module):
    def __init__(self, in1, channel=64):
        super(vgg_adapter, self).__init__()
        self.channel = channel

    def forward(self, x):
        batch, cat, height, width = x.size()
        x = torch.max(x.view(batch, self.channel, -1, height, width), dim=2)[0]
        return x
'''

class vgg_adapter(nn.Module):
    def __init__(self, in1, channel=64):
        super(vgg_adapter, self).__init__()
        self.conv = nn.Conv2d(in1, channel, 3, padding=1)
        self.relu = nn.ReLU()
        self.channel = channel

    def forward(self, x):
        x = self.relu(self.conv(x))
        #batch, cat, height, width = x.size()
        #x = torch.max(x.view(batch, self.channel, -1, height, width), dim=2)[0]
        return x
'''
        
class resnet_adapter(nn.Module):
    def __init__(self, in1=64, out=64):
        super(resnet_adapter, self).__init__()
        self.reduce = in1 > 64
        self.conv = nn.Conv2d(in1//4 if self.reduce else in1, out, 3, padding=1)
        self.relu = nn.ReLU()

    def forward(self, X):
        if self.reduce:
            batch, cat, height, width = X.size()
            X = torch.max(X.view(batch, -1, 4, height, width), dim=2)[0]
        X = self.relu(self.conv(X))
        #X = F.interpolate(X, scale_factor=2, mode='bilinear', align_corners=True)

        return X

class mobile_adapter(nn.Module):
    def __init__(self, in1=64, out=64):
        super(mobile_adapter, self).__init__()
        self.conv = nn.Conv2d(in1, out, 3, padding=1)
        self.relu = nn.ReLU()

    def forward(self, X):
        X = self.relu(self.conv(X))
        #X = F.interpolate(X, scale_factor=2, mode='bilinear', align_corners=True)
        return X


class Encoder(nn.Module):
    def __init__(self, backbone, c=64):
        super(Encoder, self).__init__()

        # resnet50
        if backbone.startswith('resnet'):
            self.encoder = resnet50(pretrained=True)
            self.adapters = nn.ModuleList([resnet_adapter(in1, c) for in1 in res_inc])
        # mobilenet
        elif backbone.startswith('mobile'):
            self.encoder = mobilenet()
            c = 16
            self.adapters = nn.ModuleList([mobile_adapter(in1, c) for in1 in mobile_inc])
        # vgg
        else:
            self.encoder = vgg('vgg16', multi=True, pretrain=True)
            self.adapters = nn.ModuleList([vgg_adapter(in1, c) for in1 in vgg_inc])
    
    def forward(self, x):
        enc_feats = self.encoder(x)
        enc_feats = [self.adapters[i](e_feat) for i, e_feat in enumerate(enc_feats)]
        return enc_feats

class baseline(nn.Module):
    def __init__(self, backbone, c=64):
        super(baseline, self).__init__()
        self.name = backbone

        self.encoder = Encoder(backbone, c)
        self.decoder = baseU(backbone, c)

    def forward(self, X):
        encoders = self.encoder(X)
        OutDict = self.decoder(encoders)
        
        return OutDict

