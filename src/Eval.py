import torch
import src
from src import Metrics, utils
import collections

class Eval:
    def __init__(self, L):
        self.Loader = L
        self.scores = collections.OrderedDict()
        for val in L.vals:
            self.scores[val] = src.Score(val, L)

    def eval_Saliency(self, Model, epoch=0, supervised=True):
        savedict = {}
        Outputs = {val : Metrics.getOutPuts(Model, valdata['X'], self.Loader, supervised=supervised) for val, valdata in self.Loader.valdatas.items()}

        for val in self.Loader.valdatas.keys():
            pred, valdata = Outputs[val]['final'], self.Loader.valdatas[val]['Y']
            F = Metrics.maxF(pred, valdata, self.Loader.ids[-1])
            M = Metrics.mae(pred, valdata)

            saves = self.scores[val].update([F, M], epoch)
            savedict[val] = saves

        for val, score in self.scores.items():
            score.print_present()
        print('-----------------------------------------')

        if self.Loader.MODE == 'train':
            torch.save(utils.makeDict(Model.state_dict()), utils.genPath(self.Loader.spath, 'present.pkl'))
            for val, saves in savedict.items():
                for idx, save in enumerate(saves):
                    if save:
                        torch.save(utils.makeDict(Model.state_dict()), utils.genPath(self.Loader.spath, val+'_'+['F', 'M'][idx]+'.pkl'))
            
            for val, score in self.scores.items():
                score.print_best()

        else:
            for val in self.Loader.valdatas.keys():
                Outputs[val]['Name'] = self.Loader.valdatas[val]['Name']
                Outputs[val]['Shape'] = self.Loader.valdatas[val]['Shape']

            return Outputs if self.Loader.save else None
