from PIL import Image
import numpy as np
import random
import time
import json
import os

import torch
from torch import nn
from functools import reduce

def deStr(param):
    if '=' not in param:
        return param, True

    else:
        key, value = param.split('=')

        if value.isalnum():
            return (key, int(value)) if value.isdigit() else (key, value)

        else:
            assert '.' in value
            return key, float(value)

def loadJson(fileName):
    with open(fileName + '.json', 'r') as jfile:
        return json.load(jfile)

def checkKey(key):
    while key[:7] == 'module.':
        key = key[7:]

    return key

def makeDict(sdict):
    return {checkKey(k) : v for k, v in sdict.items()}

#param: eg. encoder:vgg16:0.0001:pretrain,k=5,multilayers
def deParams(params, mode):
    mdict = {param.split(':')[0] : param.split(':')[1] for param in params}
    mparams = {param.split(':')[1] : {key : value for key, value in filter(lambda p : p[0] != '', map(deStr, param.split(':')[-1].split(',')))} for param in params}

    if mode == 'train' or mode == 'ft':
        plist = [[param.split(':')[0], float(param.split(':')[2])] for param in params]
        return mdict, plist, mparams

    return mdict, mparams

def genParams(plist, model):
    return [{'params' : getattr(model, p[0]).parameters(), 'lr' : p[1]} for p in plist]

def genPath(*paths):
    return reduce(lambda x, y : x + '/' + y, paths)

def initModule(modules):
    for module in modules:
        if type(module) is nn.Conv2d or type(module) is nn.Linear:
            nn.init.kaiming_normal_(module.weight)