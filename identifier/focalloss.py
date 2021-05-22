import torch
import torch.nn as nn
import torch.nn.functional as F

class FocalLoss(nn.Module):
    r"""
    """
    def __init__(self, class_num, alpha=None, gamma=2, reduction = "none"):
        super(FocalLoss, self).__init__()
        if alpha is None:
            self.alpha = torch.ones(class_num, 1)
        else:
            self.alpha = alpha
        self.gamma = gamma
        self.class_num = class_num
        self.reduction = reduction

    def forward(self, inputs, targets):
        # import pdb; pdb.set_trace()
        N = inputs.size(0)
        C = inputs.size(1)
        P = F.softmax(inputs, dim=-1)

        class_mask = inputs.data.new(N, C).fill_(0).to(device=inputs.device)
        ids = targets.view(-1, 1)
        class_mask.scatter_(1, ids.data, 1.)

        self.alpha = self.alpha.to(device=inputs.device)
        alpha = self.alpha[ids.data.view(-1)]

        probs = (P*class_mask).sum(1).view(-1,1)

        log_p = probs.log()

        batch_loss = -alpha*(torch.pow((1-probs), self.gamma))*log_p 

        loss = batch_loss
        if self.reduction=="mean":
            loss = batch_loss.mean()
        if self.reduction=="sum":
            loss = batch_loss.sum()
        return loss