import torch
from torch import nn, autograd, optim, Tensor, cuda
from torch.nn import functional as F
from torch.autograd import Variable

import torchvision.models as models


C = [64, 256, 512, 1024, 2048]
M = [64, 128, 256, 512]

res_block = {'50':[3, 4, 6, 3], '101':[3, 4, 23, 3]}

def gen_layer(layer_num, ith):

	inchannel, midchannel, outchannel = C[min(layer_num-1+ith, layer_num)], M[layer_num-1], C[layer_num]

	yield nn.Conv2d(inchannel, midchannel, 1, stride=1, bias=False)
	yield nn.BatchNorm2d(midchannel)
	yield nn.ReLU(inplace=True)

	if layer_num == 1 or ith > 0:
		yield nn.Conv2d(midchannel, midchannel, 3, stride=1, padding=1, bias=False)

	else:
		yield nn.Conv2d(midchannel, midchannel, 3, stride=2, padding=1, bias=False)

	yield nn.BatchNorm2d(midchannel)
	yield nn.ReLU(inplace=True)

	yield nn.Conv2d(midchannel, outchannel, 1, stride=1, bias=False)
	yield nn.BatchNorm2d(outchannel)

def gen_downsample(inchannel, outchannel, layer_num):

	if layer_num == 1:
		yield nn.Conv2d(inchannel, outchannel, 1, stride=1, bias=False)

	else:
		yield nn.Conv2d(inchannel, outchannel, 1, stride=2, bias=False)

	yield nn.BatchNorm2d(outchannel)

def gen_first():

	yield nn.Conv2d(3, 64, 7, stride=2, padding=3, bias=False)
	yield nn.BatchNorm2d(64)
	yield nn.ReLU(inplace=True)

class ResNet(nn.Module):

	def __init__(self, name, block, size, cls=False, multi=False):

		super(ResNet, self).__init__()

		self.name = name
		self.cls = cls
		self.multi = multi
		self.size = size

		layers = [nn.Sequential(*[block(i+1, j) for j in range(size[i])]) for i in range(4)]
		layers.insert(0, nn.Sequential(*list(gen_first())))
		self.layers = nn.ModuleList(layers)
		self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

	def forward(self, X):

		features = []

		for i in range(5):
			X = self.layers[i](X)
			features.append(X)
			
			if i == 0:
				X = self.maxpool(X)

		if self.cls:
			fc = F.avg_pool2d(X, kernel_size=7, stride=1)
			fc = fc.view(fc.shape[0], -1)

			return fc

		else:
			return features if self.multi else features[-1]

class Bottleneck(nn.Module):

	def __init__(self, layer_num, ith):

		super(Bottleneck, self).__init__()
		self.ith = ith
		self.layer_num = layer_num
		self.residual = nn.Sequential(*list(gen_layer(layer_num, ith)))

		if self.ith == 0:
			self.downsample = nn.Sequential(*list(gen_downsample(C[layer_num-1], C[layer_num], layer_num)))

	def forward(self, X):
		C = self.residual(X)

		if self.ith == 0:
			X = self.downsample(X)

		return F.relu(X + C, inplace=True)

def resnet(name='resnet50', pretrained=True):

	res = ResNet(name, Bottleneck, res_block[name[6:]], cls=False, multi=True)

	if pretrained:
		res.load_state_dict(torch.load('../PretrainModel/'+name+'.pkl', map_location='cpu'), strict=False)

	return res
'''

class ResNet(nn.Module):

	def __init__(self, name, pretrain=False):

		super(ResNet, self).__init__()
		self.model = getattr(models, name)(pretrained=pretrain)

	def forward(self, X):

		encoder = [self.model.relu(self.model.bn1(self.model.conv1(X)))]
		X = self.model.maxpool(encoder[-1])

		for i in range(1, 5):
			X = getattr(self.model, 'layer'+str(i))(X)
			encoder.append(X)

		return encoder

def resnet(name, pretrain=False):

	return ResNet(name, pretrain)
'''



if __name__ == '__main__':

	X = torch.ones(5, 3, 224, 224)

	res = resnet('resnet18', pretrain=True)
	for r in res(X):
		print(r.shape)

	res = resnet('resnet50', pretrain=True)
	#print [out.shape for out in res(X)]
	for r in res(X):
		print(r.shape)

	res = resnet('resnet101', pretrain=True)
	for r in res(X):
		print(r.shape)
