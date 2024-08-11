import math

import torch
from torch import nn
import torch.nn.functional as F
# custom weights initialization called on netG
def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        m.weight.data.normal_(0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)


class Loss_exp(nn.Module):  # Generator's max loss=e^0=1
    def __init__(self):
        super(Loss_exp, self).__init__()
        return

    def forward(self, pred, truth, proba, beta):
        criterion_1 = nn.MSELoss()
        criterion = nn.CrossEntropyLoss()
        pred_prob = F.softmax(pred, dim=1)
        loss = criterion(pred, truth) + criterion_1(pred_prob, proba) * beta
        # loss = criterion(pred, truth)
        final_loss = torch.exp(loss * -1)
        return final_loss


def chunks(arr, m):
    n = int(math.ceil(arr.size(0) / float(m)))
    return [arr[i:i + n] for i in range(0, arr.size(0), n)]
