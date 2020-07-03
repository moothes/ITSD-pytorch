from src import *

def train():
    L = Loader('train')
    E = Eval(L)
    TR = Trainer(L, E)
    TR.train()

if __name__ == '__main__':
    train()