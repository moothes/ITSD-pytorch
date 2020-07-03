import torch
import src
from src import utils, Metrics
import numpy as np
import os
from PIL import Image

class Tester(src.Experiment):
    def __init__(self, L, E):
        super(Tester, self).__init__(L, E)

    def test(self):
        self.supervised = self.Loader.supervised
        self.Outputs = self.Eval.eval_Saliency(self.Model, supervised=self.supervised)
        if self.Loader.save:
        
            self.save_preds()

    def save_preds(self):
        for valname, output in self.Outputs.items():
            rpath = utils.genPath(self.Loader.rpath, self.Loader.sub, valname)
            if not os.path.exists(rpath):
                os.makedirs(rpath)

            names, shapes, finals, time = output['Name']['Y'], output['Shape'], output['final'] * 255., output['time']
            for name, shape, final in zip(names, shapes, finals):
                ppath = utils.genPath(rpath, 'final')
                if not os.path.exists(ppath):
                    os.makedirs(ppath)
                Image.fromarray(np.uint8(final)).resize((shape), Image.BICUBIC).save(utils.genPath(ppath, name))

            if self.supervised:
                preds, conts = output['preds'] * 255., output['contour'] * 255.
                for name, shape, pred, cont in zip(names, shapes, preds, conts):
                    for idx, pre in enumerate(pred):
                        pred_path = utils.genPath(rpath, 'pred_'+ str(idx+1))
                        if not os.path.exists(pred_path):
                            os.makedirs(pred_path)
                        Image.fromarray(np.uint8(pre)).convert('L').save(utils.genPath(pred_path, name.split('.')[0]+'.png'))
                        
                    for idx, pre in enumerate(cont):
                        pred_path = utils.genPath(rpath, 'cont_'+ str(idx+1))
                        if not os.path.exists(pred_path):
                            os.makedirs(pred_path)
                        Image.fromarray(np.uint8(pre)).convert('L').save(utils.genPath(pred_path, name.split('.')[0]+'.png'))
            
            print('Save predictions of datasets: {}.'.format(valname))
