import os
import math
import argparse

import src
import collections
from src import utils
import os

import torch
from models.baseline import baseline
from thop import profile
#from flop import print_model_parm_flops, print_model_parm_nums

nece_args = {
    'normal': ['batch', 'model', 'ids', 'spath', 'metrics', 'highers', 'mpath'],
    'train': ['optim', 'loss', 'trset', 'sub', 'iter', 'epoch', 'scheduler', 'weights', 'no_contour'],
    'test': ['save', 'rpath']
}

def args(mode):
    assert mode in ['train', 'test', 'debug']
    parser = argparse.ArgumentParser()

    if mode == 'train':
        parser.add_argument('--optim', default='SGD', help='set the optimizer of model [Adadelta, Adagrad, Adam, SparseAdam, Adamax, ASGD, LBFGS, RMSprop, Rprop, SGD]')
        parser.add_argument('--trset', default='DUTS-TR', help='set the traing set')
        parser.add_argument('--scheduler', default='StepLR', help='set the scheduler')
        parser.add_argument('--lr', default=0.01, type=float, help='set base learning rate')

    parser.add_argument('--model', default='resnet', help='Set the model')
    parser.add_argument('--batch', default=8, type=int, help='Batch Size')
    parser.add_argument('--size', default=288, type=int, help='Image Size')
    parser.add_argument('--vals', default='', help='Validation sets')
    parser.add_argument('--ids', default='0,1', help='Set the cuda devices')
    parser.add_argument('--sub', default='baseline', help='The name of network')
    parser.add_argument('--cpu', action='store_true')
    parser.add_argument('--debug', action='store_true')

    parser.add_argument('--save', action='store_false')
    parser.add_argument('--supervised', action='store_true')
    parser.add_argument('--spath', default='save', help='model path')
    parser.add_argument('--rpath', default='result', help='visualization path')

    return parser.parse_args()

class Loader(object):
    def __init__(self, MODE):
        assert MODE in ['train', 'test', 'debug']
        self.MODE = MODE
        
        opt = args(MODE)

        print('loading the settings')
        self.loading(opt)

    def check(self, nece, args):
        self.nece = nece['normal'] + nece[self.MODE]
        for arg in self.nece:
            if getattr(args, arg) is None:
                print('miss the %s' % (arg))
                return False

        for arg in self.nece:
            self.__setattr__(arg, getattr(args, arg))

        return True

    def loading(self, opt):
        self.sub = opt.sub
        self.debug = opt.debug
        self.model = opt.model
        
        if self.MODE == 'train':
            self.weights = [0.1, 0.3, 0.5, 0.7, 0.9, 1.5]

            self.optim = opt.optim
            self.batch = opt.batch
            self.scheduler = opt.scheduler
            self.lr = opt.lr

            self.plist = [['encoder', self.lr*0.1], ['decoder', self.lr]]

            self.trset = 'SOD' if self.debug else opt.trset
            self.trSet = src.DataSet(self.trset, mode='train', shape=opt.size, debug=self.debug)

            self.epoch = int(math.ceil(self.trSet.size / self.batch)) if self.MODE == 'train' else 10
            self.iter = self.epoch * 25
        else:
            self.batch = 1


        os.environ["CUDA_VISIBLE_DEVICES"] = opt.ids
        num_gpu = len(opt.ids.split(','))
        self.ids = list(range(num_gpu))
        print('Backbone: {}, Using Gpu: {}'.format(self.model, opt.ids))
        
        self.cpu = opt.cpu
        self.save = opt.save
        self.supervised = opt.supervised
        self.rpath = opt.rpath
        if self.save and not os.path.exists(self.rpath):
            os.makedirs(self.rpath)
        npath = utils.genPath(self.rpath, self.sub)
        if not os.path.exists(npath):
            os.makedirs(npath)

        self.spath = utils.genPath(opt.spath, self.model, self.sub)
        if not os.path.exists(self.spath):
            os.makedirs(self.spath)
        self.mpath = self.spath + '/present.pkl'
        
        
        if self.debug:
            self.vals = ['SOD']
        elif opt.vals == '':
            self.vals = ['SOD', 'PASCAL-S', 'ECSSD', 'DUTS-TE', 'HKU-IS', 'DUT-OMRON']
        else:
            self.vals = [opt.vals, ]

        self.valdatas = collections.OrderedDict()
        for val in self.vals:
            self.valdatas[val] = src.DataSet(val, mode='test', shape=opt.size).getFull()

        self.channel = 16 if self.model.startswith('mobile') else 64
        self.Model = baseline(self.model, self.channel) 
        
        #print_model_parm_flops(self.Model)
        #print_model_parm_nums(self.Model)
        
        input = torch.randn(1, 3, opt.size, opt.size)
        flops, params = profile(self.Model, inputs=(input, ))
        print('FLOPs: {:.2f}, Params: {:.2f}.'.format(flops / 1e9, params / 1e6))


        self.mode = self.MODE
