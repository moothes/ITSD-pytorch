import torch
from torch import nn, autograd, optim, Tensor, cuda
from torch.nn import functional as F
from torch.autograd import Variable

from src import utils

#VGG-16
NUM = [3, 2, 2, 1, 1]

class sep_conv(nn.Module):
    def __init__(self, In, Out):
        super(sep_conv, self).__init__()

        self.pw_conv = nn.Conv2d(In, In, kernel_size=3, padding=1, groups=In)
        self.bn1 = nn.BatchNorm2d(In)
        self.dw_conv = nn.Conv2d(In, Out, kernel_size=1)
        self.bn2 = nn.BatchNorm2d(Out)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.bn1(self.pw_conv(x))
        x = self.relu(self.bn2(self.dw_conv(x)))
        return x


def get_centers(x, scores, K=3):
    score = scores.sigmoid_()
    centers = torch.sum(torch.sum(score * x, dim=-1, keepdim=True), dim=-2, keepdim=True)
    weights = torch.sum(torch.sum(score, dim=-1, keepdim=True), dim=-2, keepdim=True)
    centers = centers / weights
    return centers

def cls_atten(x, heat):
    centers = get_centers(x, heat, 1)
    centers = centers.view(centers.size(0), centers.size(1), 1, 1).expand_as(x)
    cos_map = F.cosine_similarity(x, centers)
    #print(x.size(), centers.size())
    return cos_map.unsqueeze(1)

def gen_convs(In, Out, num=1):
    for i in range(num):
        yield nn.Conv2d(In, In, 3, padding=1)
        yield nn.ReLU(inplace=True)

def gen_fuse(In, Out):
    yield nn.Conv2d(In, Out, 3, padding=1)
    yield nn.GroupNorm(Out//2, Out)
    yield nn.ReLU(inplace=True)

def cp(x, n=2):
    batch, cat, w, h = x.size()
    xn = x.view(batch, cat//n, n, w, h)
    xn = torch.max(xn, dim=2)[0]
    return xn

def gen_final(In, Out):
    yield nn.Conv2d(In, Out, 3, padding=1)
    yield nn.ReLU(inplace=True)

# ---------------------decode method------------------------------
def decode_conv(layer, c):
    for i in range(4 - layer):
        yield nn.Conv2d(c, c, 3, padding=1)
        yield nn.ReLU(inplace=True)
        #yield sep_conv(c, c)
        yield nn.Upsample(scale_factor=2, mode='bilinear' if i == 2 else 'nearest')

    yield nn.Conv2d(c, 8, 3, padding=1)
    yield nn.ReLU(inplace=True)

def decode_conv_new(layer, c):
    temp = c
    nc = c
    
    for i in range(4 - layer):
        oc = min(temp, nc)
        nc = temp // 2
        temp = temp // 2 if temp > 16 else 16
        yield nn.Conv2d(oc, nc, 3, padding=1)
        yield nn.ReLU(inplace=True)
        yield nn.Upsample(scale_factor=2, mode='nearest')

    yield nn.Conv2d(nc, 8, 3, padding=1)
    yield nn.ReLU(inplace=True)

class pred_block(nn.Module):
    def __init__(self, In, Out, up=False):
        super(pred_block, self).__init__()

        self.final_conv = nn.Conv2d(In, Out, 3, padding=1)
        self.pr_conv = nn.Conv2d(Out, 4, 3, padding=1)
        self.up = up

    def forward(self, X):
        a = nn.functional.relu(self.final_conv(X))
        a1 = self.pr_conv(a)
        pred = torch.max(a1, dim=1)[0]
        if self.up: 
            a = nn.functional.interpolate(a, scale_factor=2, mode='bilinear')
        return [a, pred]

class res_block(nn.Module):
    def __init__(self, cat, layer):
        super(res_block, self).__init__()

        if layer:
            self.conv4 = nn.Sequential(*list(gen_fuse(cat, cat // 2)))

        self.convs = nn.Sequential(*list(gen_convs(cat, cat, NUM[layer])))
        self.conv2 = nn.Sequential(*list(gen_fuse(cat, cat//2)))

        self.final = nn.Sequential(*list(gen_final(cat, cat)))
        self.layer = layer
        self.initialize()

    def forward(self, X, encoder):
        if self.layer:
            X = nn.functional.interpolate(X, scale_factor=2, mode='bilinear')
            c = cp(X)
            d = self.conv4(encoder)
            X = torch.cat([c, d], 1)

        X = self.convs(X)
        a = cp(X)
        b = self.conv2(encoder)
        f = torch.cat([a, b], 1)
        f = self.final(f)
        return f

    def initialize(self):
        utils.initModule(self.convs)
        utils.initModule(self.conv2)
        utils.initModule(self.final)

        if self.layer:
            utils.initModule(self.conv4)

class ctr_block(nn.Module):
    def __init__(self, cat, layer):
        super(ctr_block, self).__init__()
        self.conv1 = nn.Sequential(*list(gen_convs(cat, cat, NUM[layer])))
        self.conv2 = nn.Sequential(*list(gen_fuse(cat, cat)))
        self.final = nn.Sequential(*list(gen_final(cat, cat)))
        self.layer = layer
        self.initialize()

    def forward(self, X):
        X = self.conv1(X)
        if self.layer:
            X = nn.functional.interpolate(X, scale_factor=2, mode='bilinear')
        X = self.conv2(X)
        x = self.final(X)
        return x

    def initialize(self):
        utils.initModule(self.conv1)
        utils.initModule(self.conv2)
        utils.initModule(self.final)

class final_block(nn.Module):
    def __init__(self, backbone, channel):
        super(final_block, self).__init__()
        self.slc_decode = nn.ModuleList([nn.Sequential(*list(decode_conv(i, channel))) for i in range(5)])
        self.conv = nn.Conv2d(40, 8, 3, padding=1)
        self.backbone = backbone

    def forward(self, xs):
        feats = [self.slc_decode[i](xs[i]) for i in range(5)]
        x = torch.cat(feats, 1)
        
        x = self.conv(x)
        if not self.backbone.startswith('vgg'):
            x = nn.functional.interpolate(x, scale_factor=2, mode='bilinear')
        
        #x = cp(x, 4)
        #print(x.size())
        x = torch.max(x, dim=1)[0]
        return x

        
class cat_block(nn.Module):
    def __init__(self, backbone, channel):
        super(cat_block, self).__init__()
        self.convs = nn.ModuleList([nn.Sequential(*list(gen_convs(channel//2, channel//4))) for i in range(5)])
        self.up = nn.Upsample(size=(224, 224), mode='bilinear')
        self.conv = nn.Conv2d(int(5 * channel / 8), 8, 3, padding=1)
        self.backbone = backbone

    def forward(self, xs):
        assert len(xs) == 5
        flist = []
        for i, x in enumerate(xs):
            xt = cp(x)
            xt = self.convs[i](xt)
            flist.append(self.up(cp(xt, 4)))

        feat = torch.cat(flist, dim=1)
        x = self.conv(feat)
        x = torch.max(x, dim=1)[0]
        return x
        
def slc_to_ctr(x):  
    x = 2 * x - nn.functional.avg_pool2d(x, 3, stride=1, padding=1)
    return x
    
def ctr_to_slc(x, mask):
    channel = x.size(1)
    mask = (1 - mask.sigmoid_()).unsqueeze(1)
    mask = nn.functional.interpolate(mask, scale_factor=2, mode='bilinear')
    #m = mask.expand(-1, channel, -1, -1)
    
    x_mean = torch.sum(mask * x, dim=(2, 3))
    m_sum = torch.sum(mask, dim=(2, 3))
    #print(x_mean.size(), x.size())
    x = x - (x_mean / m_sum).unsqueeze(-1).unsqueeze(-1)
    return x

class baseU(nn.Module):
    def __init__(self, backbone=False, channel=64):
        super(baseU, self).__init__()
        self.name = 'baseU'
        self.layer = 5

        self.slc_blocks = nn.ModuleList([res_block(channel, i) for i in range(self.layer)])
        self.slc_preds = nn.ModuleList([pred_block(channel, channel//2)  for i in range(self.layer)])

        self.ctr_blocks = nn.ModuleList([ctr_block(channel, i) for i in range(self.layer)])
        self.ctr_preds = nn.ModuleList([pred_block(channel, channel//2, up=True)  for i in range(self.layer)])

        self.slc_decode = nn.ModuleList([nn.Sequential(*list(decode_conv(i, channel))) for i in range(5)])
        self.final = final_block(backbone, channel)

    def forward(self, encoders):
        slcs, slc_maps = [encoders[-1]], []
        ctrs, ctr_maps = [], []
        stc, cts = None, None

        for i in range(self.layer):
            slc = self.slc_blocks[i](slcs[-1], encoders[self.layer - 1 - i])
            if cts is not None:
                #slc = torch.cat([fuse(slc), ctr_to_slc(cts, slc_maps[-1])], dim=1)
                slc = torch.cat([cp(slc), cts], dim=1)
            else:
                ctrs.append(slc)
            stc, slc_map = self.slc_preds[i](slc)

            ctr = self.ctr_blocks[i](ctrs[-1])
            #ctr = torch.cat([fuse(ctr), slc_to_ctr(stc)], dim=1)
            ctr = torch.cat([cp(ctr), stc], dim=1)
            cts, ctr_map = self.ctr_preds[i](ctr)

            slcs.append(slc)
            ctrs.append(ctr)
            slc_maps.append(slc_map)
            ctr_maps.append(ctr_map)

        #slc_feats = slcs[1:]
        #feats = [self.slc_decode[i](slc_feats[i]) for i in range(5)]

        #feat = torch.cat(feats, 1)
        final = self.final(slcs[1:])
        #slc_maps.append(final)
        #ctr_maps.append(None)

        OutPuts = {'final':final, 'preds':slc_maps, 'contour':ctr_maps}
        return OutPuts
