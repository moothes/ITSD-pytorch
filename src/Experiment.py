import time
import torch
import src
from src import Loss

class Experiment(object):
    def __init__(self, L, E):
        self.Loader = L
        self.Eval = E
        self.Model = L.Model
        if L.MODE in ('test') or L.mode == 'ft':
            self.Model.load_state_dict(torch.load(L.mpath, map_location='cpu'))

        self.Model = self.Model.eval() if L.MODE == 'test' else self.Model.train()
        if not L.cpu:
            self.Model = torch.nn.DataParallel(self.Model.cuda(L.ids[0]), device_ids=L.ids)

    def optims(self, optim, params):

        if optim == 'SGD':
            print('using SGD')
            return torch.optim.SGD(params=params, momentum=0.9, weight_decay=0.0005)

        elif optim == 'Adam':
            print('using Adam')
            return torch.optim.Adam(params=params, weight_decay=0.0005)

    def schedulers(self, scheduler, optimizer):

        if scheduler == 'StepLR':
            return torch.optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.1)

        elif scheduler == 'ReduceLROnPlateau':
            return torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer)
