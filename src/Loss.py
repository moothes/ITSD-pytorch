import torch
from torch import nn, autograd, optim, Tensor, cuda
from torch.nn import functional as F
from torch.autograd import Variable


class FS_loss(nn.Module):
  def __init__(self, weights, b=0.3):
    super(FS_loss, self).__init__()
    self.contour = weights
    self.b = b

  def forward(self, X, Y, weights):
    loss = 0
    batch = Y.size(0)
    
    for weight, x in zip(weights, X):
        pre = x.sigmoid_()
        scale = int(Y.size(2) / x.size(2))
        pos = F.avg_pool2d(Y, kernel_size=scale, stride=scale).gt(0.5).float()
        tp = pre * pos
        
        tp = (tp.view(batch, -1)).sum(dim = -1)
        posi = (pos.view(batch, -1)).sum(dim = -1)
        pre = (pre.view(batch, -1)).sum(dim = -1)
        
        f_score = tp * (1 + self.b) / (self.b * posi + pre)
        loss += weight * (1 - f_score.mean())
    return loss



def ACT(X, batchs, args):
    bce = nn.BCEWithLogitsLoss(reduction='none')
    slc_gt = torch.tensor(batchs['Y']).cuda()
    ctr_gt = torch.tensor(batchs['C']).cuda()

    slc_loss, ctr_loss = 0, 0
    for slc_pred, ctr_pred, weight in zip(X['preds'], X['contour'], args.weights):
        scale = int(slc_gt.size(-1) / slc_pred.size(-1))
        ys = F.avg_pool2d(slc_gt, kernel_size=scale, stride=scale).gt(0.5).float()
        yc = F.max_pool2d(ctr_gt, kernel_size=scale, stride=scale)
        
        slc_pred = slc_pred.squeeze(1)
        
        # contour loss
        #w = torch.yc

        # ACT loss
        pc = ctr_pred.sigmoid_()
        w = torch.where(pc > yc, pc, yc)

        slc_loss += (bce(slc_pred, ys) * (w * 4 + 1)).mean() * weight
            
        if ctr_pred is not None:
            ctr_pred = ctr_pred.squeeze(1)
            ctr_loss += bce(ctr_pred, yc).mean() * weight

    pc = F.interpolate(pc.unsqueeze(1), size=ctr_gt.size()[-2:], mode='bilinear').squeeze(1)
    w = torch.where(pc > ctr_gt, pc, ctr_gt)
    fnl_loss = (bce(X['final'], slc_gt.gt(0.5).float()) * (w * 4 + 1)).mean() * args.weights[-1]

    return fnl_loss + ctr_loss + slc_loss





