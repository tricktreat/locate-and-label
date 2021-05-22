import pdb
import torch
import torch.nn as nn
import torch.nn.functional as F

class GiouLoss(nn.Module):
    r"""
    一维GiouLoss
    """
    def __init__(self, t = "giou", reduction = "none"):
        super(GiouLoss, self).__init__()
        self.reduction = reduction
        self.t = t

    def forward(self, inputs, targets):
        # import pdb; pdb.set_trace()
        i_left = torch.max(inputs[:, 0], targets[:, 0])
        i_right = torch.min(inputs[:, 1], targets[:, 1])
        o_left = torch.min(inputs[:, 0], targets[:, 0])
        o_right = torch.max(inputs[:, 1], targets[:, 1])
        iou = (i_right - i_left)/( o_right - o_left + 1e-30)
        if self.t=="iou":
            inx = iou<0
            iou[inx] = 0
        loss = 1 - iou
        if self.reduction=="mean":
            loss = loss.mean()
        if self.reduction=="sum":
            loss = loss.sum()
        return loss