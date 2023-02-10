import torch
import torch.utils.data
from torch import nn
import torch.nn.functional as F

from collections import OrderedDict

def aux_criterion(inputs, target):
    # for FCN and deeplab
    losses = {}
    for name, x in inputs.items():
        #losses[name] = nn.functional.cross_entropy(x, target, ignore_index=255)
        losses[name] = nn.functional.cross_entropy(x, target)

    if len(losses) == 1:
        return losses['out']

    return losses['out'] + 0.5 * losses['aux']

class criterion_CEloss(nn.Module):
    def __init__(self,weight=None):
        super(criterion_CEloss, self).__init__()
        self.loss = nn.NLLLoss(weight)

    def forward(self, output, target):
        return self.loss(F.log_softmax(output, dim=1), target)

class Bicriterion(nn.Module):
    def __init__(self, mask_loss):
        super(Bicriterion, self).__init__()
        self.mask_loss = mask_loss

    def forward(self, output, target):
        mask_loss = self.mask_loss(output['out'], target)
        loss = mask_loss
        return loss

class DiceLoss(nn.Module):
    def __init__(self):
        super(DiceLoss, self).__init__()
        self.ep = 1e-8

    def forward(self, output, target):
        pred = (F.softmax(output, dim=1)[:, 0]).float()
        target = (target > 0).float()
        intersection = 2 * torch.sum(pred * target) + self.ep
        union = torch.sum(pred) + torch.sum(target) + self.ep
        loss = 1 - intersection / union
        return loss


class FocalLoss(nn.Module):
    # Wraps focal loss around existing loss_fcn(), i.e. criteria = FocalLoss(nn.BCEWithLogitsLoss(), gamma=1.5)
    def __init__(self, loss_fcn, gamma=1.5, alpha=0.25):
        # alpha  balance positive & negative samples
        # gamma  focus on difficult samples
        super(FocalLoss, self).__init__()
        self.loss_fcn = loss_fcn  # must be nn.BCEWithLogitsLoss()
        self.gamma = gamma
        self.alpha = alpha
        self.reduction = loss_fcn.reduction
        self.loss_fcn.reduction = 'none'  # required to apply FL to each element

    def forward(self, pred, true):
        if isinstance(pred, OrderedDict):
            pred = pred['out']
        if true.dim() == 3:
            true.unsqueeze_(1)
            true = torch.cat([1-true, true], dim=1)
        true = true.float()
        loss = self.loss_fcn(pred, true)
        # p_t = torch.exp(-loss)
        # loss *= self.alpha * (1.000001 - p_t) ** self.gamma  # non-zero power for gradient stability

        # TF implementation https://github.com/tensorflow/addons/blob/v0.7.1/tensorflow_addons/losses/focal_loss.py
        pred_prob = torch.sigmoid(pred)  # prob from logits
        p_t = true * pred_prob + (1 - true) * (1 - pred_prob)
        alpha_factor = true * self.alpha + (1 - true) * (1 - self.alpha)
        modulating_factor = (1.0 - p_t) ** self.gamma
        loss *= alpha_factor * modulating_factor

        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        else:  # 'none'
            return loss

def get_loss(name, loss_weight=None):
    print("Loss: {}".format(name))
    if name == 'focalloss':
        BCEseg = nn.BCEWithLogitsLoss()
        FLseg = FocalLoss(BCEseg)
        return FLseg
    elif name == 'ce':
        return nn.CrossEntropyLoss(loss_weight, ignore_index=255)
    elif name == 'aux':
        return aux_criterion
    elif name == 'bi':
        mask_loss = nn.CrossEntropyLoss(loss_weight)
        return Bicriterion(mask_loss)
    elif name == 'bi_dice':
        dice_loss = DiceLoss()
        return Bicriterion(dice_loss)
    else:
        raise ValueError(name)

        
