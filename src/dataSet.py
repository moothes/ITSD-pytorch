from PIL import Image
import numpy as np
import random
import time
import json
import os
import cv2
from src import utils

class _DataSet(object):
    def __init__(self, name, mode='train', shape=224, debug=False):

        super(_DataSet, self).__init__()
        self.name = name

        self.tic = time.time()
        self.mean = np.array([0.485, 0.458, 0.407]).reshape([1, 3, 1, 1])
        self.std = np.array([0.229, 0.224, 0.225]).reshape([1, 3, 1, 1])
        self.dataset = utils.loadJson(utils.genPath('setup'))['DATASET']
        assert name in self.dataset
        self.path = self.dataset[self.name]

        assert mode in ['train', 'test']
        self.mode = mode
        
        self.shape = shape
        self.flip_code = [1, 0, -1]

        self.getNames()

    def getNames(self):
        names = list(map(lambda x : sorted(os.listdir(x)), self.path))
        paths = [list(map(lambda x : utils.genPath(path, x), name)) for name, path in zip(names, self.path)]

        self.names = {'X' : names[0], 'Y' : names[1]}
        self.paths = paths
        self.size = len(paths[0])


    def Crop_new(self, X, Y, C):
        dice = random.random()
        if dice < 0.3:
            rand_size = random.randint(self.shape[0], self.aug_shape[0])

            rand_w = random.randint(0, self.aug_shape[0] - rand_size)
            rand_h = random.randint(0, self.aug_shape[1] - rand_size)
            X = X[rand_w:rand_w+rand_size, rand_h:rand_h+rand_size]
            Y = Y[rand_w:rand_w+rand_size, rand_h:rand_h+rand_size]
            C = C[rand_w:rand_w+rand_size, rand_h:rand_h+rand_size]

        X, Y, C = self.resize(X, Y, C)
        return X, Y, C
        
    def Crop(self, X, Y, shape):
        dice = random.randint(0, 1)
        w, h, _ = X.shape

        if dice == 1:
            rand_w = random.randint(0, w - shape)
            rand_h = random.randint(0, h - shape)
            X = X[rand_w:rand_w+shape, rand_h:rand_h+shape]
            Y = Y[rand_w:rand_w+shape, rand_h:rand_h+shape]
        else:
            X = cv2.resize(X, (shape, shape))
            Y = cv2.resize(Y, (shape, shape))

        return X, Y

    def random_rotate(self, X, Y):
        angle = np.random.randint(-25,25)
        h, w = Y.shape
        center = (w / 2, h / 2)
        M = cv2.getRotationMatrix2D(center, angle, 1.0)
        
        X = cv2.warpAffine(X, M, (w, h))
        Y = cv2.warpAffine(Y, M, (w, h))
        return X, Y

    def random_light(self, x):
        contrast = np.random.rand(1) + 0.5
        light = np.random.randint(-20,20)
        x = contrast * x + light
        return np.clip(x,0,255)

    def Flip(self, X, Y):
        dice = random.randint(0, 1)
        return (X, Y) if dice == 0 else (cv2.flip(X, self.flip_code[dice-1]), cv2.flip(Y, self.flip_code[dice-1]))
        
    def Normalize(self, X):
        X = X.transpose((0, 3, 1, 2))
        X /= 255.
        X -= self.mean
        X /= self.std
        return X

    def shuffle(self):
        self.index = 0
        random.shuffle(self.idlist)

    def reProcess(self, X):
        X = ((X * self.std) + self.mean) * 255
        X = X.transpose((0, 2, 3, 1))
        return X

    def help(self):
        print('shape     : %s' % (list(self.shape)))
        print('[0:N,1:H,2:V,3H+V]')
        print('mode      : %s' % (self.mode))
        print('[train,test,trainval]')
        
class DataSet(_DataSet):
    def __init__(self, name, mode='train', shape=224, debug=False):
        super(DataSet, self).__init__(name=name, mode=mode, shape=shape, debug=debug)

        if mode == 'train':
            self.idlist = list(range(self.size))
            self.shuffle()
        
        self.debug = debug

    def getFull(self):
        DataDict = {
            'X':np.empty((self.size, self.shape, self.shape, 3), dtype=np.float32), 
            'Y':np.empty((self.size, self.shape, self.shape), dtype=np.float32),
            'C':np.empty((self.size, self.shape, self.shape), dtype=np.float32)
        }
        self.sizes = []
        if self.debug:
            print('debug mode! random images are generated.')
            DataDict['X'] = np.random.rand(self.size, self.shape, self.shape, 3) * 255
            DataDict['Y'] = np.random.rand(self.size, self.shape, self.shape) * 255
            DataDict['C'] = np.random.rand(self.size, self.shape, self.shape)
        else:
            for idx in range(self.size):
                DataDict['X'][idx] = np.array(Image.open(self.paths[0][idx]).convert('RGB').resize((self.shape, self.shape), Image.ANTIALIAS))
                imgY = Image.open(self.paths[1][idx]).convert('L')
                self.sizes.append(imgY.size)
                DataDict['Y'][idx] = (np.array(imgY.resize((self.shape, self.shape), Image.ANTIALIAS)) > 127).astype(np.float64)

            kernel = np.ones((5, 5))
            for idx, y in enumerate(DataDict['Y']):
                DataDict['C'][idx] = cv2.dilate(y, kernel) - cv2.erode(y, kernel)
            
        DataDict['X'] = self.Normalize(DataDict['X'])
        FullDict = {'X' : DataDict['X'], 'Y' : np.int32(DataDict['Y']), 'C' : np.int32(DataDict['C']), 'Name' : self.names, 'Shape' : self.sizes}
        print('loading {} images from {} using time: {}s.'.format(self.size, self.name, round(time.time() - self.tic, 3)))

        return FullDict
    
    def mixup(self, outputs, batch):
        dice = random.randint(0, 1)

        if dice == 1:
            b_list = np.random.permutation(batch)
        else:
            b_list = np.ones((batch)).astype(np.int32)
        
        a_list = np.array(range(batch))
        
        theta = np.random.sample(batch).reshape((batch, 1, 1, 1))
        
        b_X = outputs['X'][b_list]
        outputs['X'] = theta * outputs['X'] + (1 - theta) * b_X
        
        outputs['Ym'] = outputs['Y'][b_list]
        outputs['Cm'] = outputs['C'][b_list]
        outputs['theta'] = theta
        
        return outputs
    
    def getBatch(self, batch):
        scales = [-1, 0, 1] 
        b_size = int(np.random.choice(scales, 1)) * 32 + self.shape
        aug_shape = int(b_size * 1.1)

        OutPuts = {
            'X':np.empty((batch, b_size, b_size, 3), dtype=np.float32),
            'Y':np.empty((batch, b_size, b_size), dtype=np.float32),
            'C':np.empty((batch, b_size, b_size), dtype=np.float32)
        }

        for idx in range(batch):
            if self.index == self.size:
                self.shuffle()

            index = self.idlist[self.index]
            X = np.array(Image.open(self.paths[0][index]).convert('RGB').resize((aug_shape, aug_shape), Image.ANTIALIAS))
            Y = (np.array(Image.open(self.paths[1][index]).convert('L').resize((aug_shape, aug_shape), Image.ANTIALIAS)) > 127).astype(np.float64)

            X, Y = self.Crop(X, Y, b_size)
            X, Y = self.Flip(X, Y)

            kernel = np.ones((5, 5))
            C = cv2.dilate(Y, kernel) - cv2.erode(Y, kernel)
            OutPuts['X'][idx], OutPuts['Y'][idx], OutPuts['C'][idx] = X, Y, C
            self.index += 1

        OutPuts['X'] = self.Normalize(OutPuts['X'])
        return OutPuts

if __name__ == '__main__':

    name = 'SOD'

    data = DataSet(name, 'train') 
    batchsize = 1
    out = data.getBatch(batchsize)
    
    for i in range(300):
        out = data.getBatch(batchsize)
    
        Y = out['Y'] * 255
        C = out['C'] * 255
    
        X = data.reProcess(out['X'])
        for x, y, c in zip(X, Y, C):
            imx = Image.fromarray(np.uint8(x))
            imy = Image.fromarray(np.uint8(y))
            imz = Image.fromarray(np.uint8(c))
            
            imx.save('temp/{}x.jpg'.format(i))
            imy.save('temp/{}y.jpg'.format(i))
            imz.save('temp/{}z.jpg'.format(i))
    
    '''
    out = data.getFull()
    for i in range(300):
        #print(out['X'][i])
        
        X = np.expand_dims(out['X'][i], 0)
        X = data.reProcess(X)
        #out = data.getBatch(batchsize)
    
        Y = np.expand_dims(out['Y'][i], 0) * 255
        C = np.expand_dims(out['C'][i], 0) * 255
    
        #X = data.reProcess(out['X'])
        for x, y, c in zip(X, Y, C):
            imx = Image.fromarray(np.uint8(x))
            imy = Image.fromarray(np.uint8(y))
            imz = Image.fromarray(np.uint8(c))
            
            imx.save('temp/{}x.jpg'.format(i))
            imy.save('temp/{}y.jpg'.format(i))
            imz.save('temp/{}z.jpg'.format(i))
    #print(fulldict['Name']['X'])
    #print(fulldict['Shape'])
    '''