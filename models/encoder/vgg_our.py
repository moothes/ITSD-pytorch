import torch
from torch import nn, autograd, optim, Tensor, cuda
from torch.nn import functional as F

PretrainPath = 'PretrainModel/'

def gen_convs(inchannel, outchannel, bn=False):

    yield nn.Conv2d(inchannel, outchannel, 3, padding=1)

    if bn:
        yield nn.BatchNorm2d(outchannel)

    yield nn.ReLU(inplace=True)

    
def gen_fcs(infeature, outfeature):

    yield nn.Linear(infeature, outfeature)
    yield nn.ReLU(inplace=True)
    yield nn.Dropout(p=0.5)

class VGGNet(nn.Module):

    def __init__(self, name, cls=False, multi=False, bn=False):

        super(VGGNet, self).__init__()
        C = [3, 64, 128, 256, 512, 512]
        FC = [25088, 4096, 4096]
        N = [2, 2, 3, 3, 3] if 'vgg16' in name else [2, 2, 4, 4, 4]

        self.cls = cls
        self.multi = multi
        self.convs = nn.ModuleList([nn.Sequential(*[m for j in range(N[i]) for m in gen_convs(C[min(i+j, i+1)], C[i+1], bn=bn)]) for i in range(5)])
        self.fc = nn.Sequential(*[m for i in range(2) for m in gen_fcs(FC[i], FC[i+1])]) if cls else None

    def forward(self, X):

        features = []
        for i in range(5):
            Out = self.convs[i](X)
            features.append(Out)
            X = F.max_pool2d(Out, kernel_size=2, stride=2)
            
        if self.cls:
            fc = X.view(X.shape[0], -1)
            fc = self.fc(fc)

            return fc

        else:
            return features if self.multi else features[-1]

    def load(self, path):

        convs = [0, 2, 5, 7, 10, 12, 14, 16, 19, 21, 23, 25, 28, 30, 32, 34]
        fcs = [0, 3, 6]

        count = 0
        state_dict = torch.load(path, map_location='cpu')
        for conv in self.convs:
            print(len(conv))
            for i in range(0, len(conv), 2):
                conv[i].weight.data = state_dict['features.'+str(convs[count])+'.weight']
                conv[i].bias.data = state_dict['features.'+str(convs[count])+'.bias']
                count += 1

        count = 0
        print(len(self.fc))
        for i in range(0, len(self.fc), 3):
            self.fc[i].weight.data = state_dict['classifier.'+str(fcs[count])+'.weight']
            self.fc[i].bias.data = state_dict['classifier.'+str(fcs[count])+'.bias']
            count += 1


def vgg(name=None, cls=False, multi=True, pretrain=True):

    assert 'vgg16' in name or 'vgg19' in name

    vgg = VGGNet(name, cls=cls, multi=multi, bn=True if 'bn' in name else False)

    if pretrain:
        #print('It should load pre-trained VGG16, but we omit it')
        vgg.load_state_dict(torch.load('../PretrainModel/'+name+'.pkl', map_location='cpu'), strict=False)

    return vgg

if __name__ == '__main__':

    '''
    a = vgg('vgg19', multi=True, cls=True)
    print a

    X = torch.empty(5, 3, 224, 224)
    for encoder in a(X):
        print encoder.shape
    '''
    '''
    m = vgg('vgg19', cls=True, multi=True)
    m.load('../../PretrainModel/vgg19.pth')
    torch.save(m.state_dict(), 'vgg19.pkl')
    '''
    a = torch.load('vgg19.pkl', map_location='cpu')
    b = torch.load('../../PretrainModel/vgg19.pth', map_location='cpu')

    afk = sorted(filter(lambda fc : 'fc' in fc, a.keys()))
    bfk = sorted(filter(lambda cl : 'classifier' in cl, b.keys()))
    for af, bf in zip(afk, bfk):
        print(af, bf)
        assert torch.equal(a[af], b[bf])
    '''
    ack = sorted(filter(lambda convs : 'convs' in convs, a.keys()))
    bck = sorted(filter(lambda convs : 'features' in convs, b.keys()), key=lambda a : int(a.split('.')[1]))
    '''
    ack = filter(lambda convs : 'convs' in convs, a.keys())
    bck = filter(lambda convs : 'features' in convs, b.keys())
    for ak, bk in zip(ack, bck):
        print(ak, bk)
        assert torch.equal(a[ak], b[bk])