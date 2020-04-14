import numpy as np
from models import baseline
import torch
import src
import time

dataset = src.DataSet('DUTS-TE', mode='test', shape=288).getFull()
xs = torch.from_numpy(dataset['X']).cuda()
print(xs.size())
#x = torch.zeros((1, 3, 224, 224))
model = baseline('resnet', 64, False).cuda()

st = time.time()
for i in range(len(xs)):
    out = model(xs[i:i+1])
    pred = out['final'].cpu().detach().numpy()
print(len(xs) / (time.time() - st))