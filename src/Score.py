class Score:
    def __init__(self, name, loader):
        self.name = name
        self.metrics = ['F', 'M']
        self.highers = [1, 0]
        self.scores = [0. if higher else 1. for higher in self.highers]
        self.best = self.scores
        self.best_epoch = [0] * len(self.scores)
        self.present = self.scores

    def update(self, scores, epoch):

        self.present = scores
        self.epoch = epoch
        self.best = [max(best, score) if self.highers[idx] else min(best, score) for idx, (best, score) in enumerate(zip(self.best, scores))]
        self.best_epoch = [epoch if present == best else best_epoch for present, best, best_epoch in zip(self.present, self.best, self.best_epoch)]
        saves = [epoch == best_epoch for best_epoch in self.best_epoch]

        return saves

    def print_present(self):
        m_str = '{} : {:.4f}, {} : {:.4f} on ' + self.name
        m_list = []
        for metric, present in zip(self.metrics, self.present):
            m_list.append(metric)
            m_list.append(present)
        print(m_str.format(*m_list))


    def print_best(self):
        m_str = 'Best score: {}_{} : {:.4f}, {}_{} : {:.4f} on ' + self.name
        m_list = []
        for metric, best, best_epoch in zip(self.metrics, self.best, self.best_epoch):
            m_list.append(metric)
            m_list.append(best_epoch)
            m_list.append(best)
        print(m_str.format(*m_list))
        
