# encoding: utf-8
"""
@author:    Andreas Gebhardt
@contact:   AGebhardt1999@gmail.com
"""
import torch
import torch.nn.functional as F

def cross_entropy_loss(pred_class_outputs, gt_classes, eps, alpha=0.2, reduce=False):
    num_classes = pred_class_outputs.size(1)

    if eps >= 0:
        smooth_param = eps
    else:
        # Adaptive label smooth regularization
        soft_label = F.softmax(pred_class_outputs, dim=1)
        smooth_param = alpha * soft_label[torch.arange(soft_label.size(0)), gt_classes].unsqueeze(1)

    log_probs = F.log_softmax(pred_class_outputs, dim=1)
    with torch.no_grad():
        targets = torch.ones_like(log_probs)
        targets *= smooth_param / (num_classes - 1)
        targets.scatter_(1, gt_classes.data.unsqueeze(1), (1 - smooth_param))

    if reduce:
        loss = (-targets * log_probs).sum(dim=1).mean(dim=0)
    else:
        loss = (-targets * log_probs).sum(dim=1)
    return loss


def data_uncertainty_loss(pred_class_outputs, gt_classes, feat_var, eps, alpha=0.2, lamda=1.0):
    
    loss = cross_entropy_loss(pred_class_outputs, gt_classes, eps, alpha, reduce=False)
    
    feat_var = feat_var.mean(dim=1)

    loss = loss / (feat_var + 1e-10) + lamda * torch.log(feat_var + 1e-10)
    
    return loss.mean()



