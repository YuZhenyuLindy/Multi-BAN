# -*- coding: utf-8 -*
import math
import pdb
import numpy as np
from typing import Optional, List, Tuple, Dict

import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F
from grl import WarmStartGradientReverseLayer, GradientReverseFunction
from _util import binary_accuracy, accuracy

def MI(outputs_target):
    batch_size = outputs_target.size(0)
    softmax_outs_t = nn.Softmax(dim=1)(outputs_target)
    avg_softmax_outs_t = torch.sum(softmax_outs_t, dim=0) / float(batch_size)
    log_avg_softmax_outs_t = torch.log(avg_softmax_outs_t + 1e-8)
    item1 = -torch.sum(avg_softmax_outs_t * log_avg_softmax_outs_t)
    item2 = -torch.sum(softmax_outs_t * torch.log(softmax_outs_t + 1e-8)) / float(batch_size)
    return item1, item2

class AdversarialLoss_MND(nn.Module):
    def __init__(self, classifier: nn.Module, args):
        super(AdversarialLoss_MND, self).__init__()
        self.grl = WarmStartGradientReverseLayer(alpha=1., lo=0., hi=1., max_iters=1000, auto_step=True)
        self.classifier = classifier
        self.iter_nums = 0

        self.cache_feature = args.cache_feature
        self.queue_K = args.queue_K 
        self.queue_size = args.batch_size
        self.queue_dim = args.bottle_neck if self.cache_feature else args.num_classes

        self.queue_pointer_s = np.arange(self.queue_size) * self.queue_K
        self.queue_pointer_t = np.arange(self.queue_size) * self.queue_K
        self.queue_s = torch.zeros((self.queue_size * self.queue_K, self.queue_dim)).cuda()
        self.queue_t = torch.zeros((self.queue_size * self.queue_K, self.queue_dim)).cuda()
        
    def update_queue(self, data, labels, name='s'):
        if name=='s':
            cur_queue = self.queue_s
            cur_queue_pointer = self.queue_pointer_s
        else:
            cur_queue = self.queue_t
            cur_queue_pointer = self.queue_pointer_t

        for i in torch.unique(labels):
            cur_mask = labels == i
            cur_num = torch.sum(cur_mask)
            cur_data = data[cur_mask]
              
            start_pointer = i * self.queue_K
            end_pointer = (i + 1)  * self.queue_K
            if cur_num >= self.queue_K:
                cur_queue[start_pointer:end_pointer,:] = data[:self.queue_K,:]
                cur_queue_pointer[i] = start_pointer
            else:
                ptr = cur_queue_pointer[i]
                new_end_pointer = ptr + cur_num
                cur_queue_pointer[i] = new_end_pointer % self.queue_K
                if new_end_pointer <= end_pointer:
                    cur_queue[ptr:new_end_pointer,:] = data[:cur_num,:]
                else:
                    cur_queue[ptr:end_pointer,:] = data[:end_pointer-ptr,:]
                    cur_queue[start_pointer: cur_queue_pointer[i],:] = data[end_pointer-ptr:cur_num,:]

    def forward(self, f, labels_s, args):
        f_grl = self.grl(f, coeff_auto=False)
        y = self.classifier(f_grl)
        if args.feature_normal:
            y = y / args.feature_temp
        y_s, y_t = y.chunk(2, dim=0)
                
        # test temperature for y
        source_logit = nn.Softmax(dim=1)(y_s)
        target_logit = nn.Softmax(dim=1)(y_t / args.logit_temp)

        
        pseudo_label_t = y_t.argmax(1)
        labels = torch.cat((labels_s, pseudo_label_t), dim=0)

        self.iter_nums += 1

        if args.weight_type == 0:
            loss_cls_target = -(torch.norm(target_logit, 'nuc') - torch.norm(source_logit, 'nuc')) / (target_logit.size()[0])

        elif args.weight_type == 1:
            if self.cache_feature:
                output_queue_s = nn.Softmax(dim=1)(self.classifier(self.queue_s)).detach()
                output_queue_t = nn.Softmax(dim=1)(self.classifier(self.queue_t) / args.logit_temp).detach()
            else:
                output_queue_s = self.queue_s
                output_queue_t = self.queue_t
            all_queue_s = torch.cat([source_logit, output_queue_s], dim=0)
            all_queue_t = torch.cat([target_logit, output_queue_t], dim=0)

            if self.iter_nums > self.queue_K:
                queue_pointer = self.queue_size * (1 + self.queue_K)
            else:
                queue_pointer = self.queue_size * self.iter_nums 
            loss_cls_target = -(torch.norm(all_queue_t[:queue_pointer, :], p='nuc') - torch.norm(all_queue_s[:queue_pointer,:], p='nuc')) / (target_logit.size()[0])


            start = ((self.iter_nums - 1) % self.queue_K) * self.queue_size
            if self.cache_feature:
                feature_s, feature_t = f.chunk(2, dim=0)
                self.queue_s[start: start + self.queue_size,:] = feature_s.data
                self.queue_t[start: start + self.queue_size,:] = feature_t.data
            else:
                self.queue_s[start: start + self.queue_size,:] = source_logit.data
                self.queue_t[start: start + self.queue_size,:] = target_logit.data

        return loss_cls_target

class DomainAdversarialLoss(nn.Module):
    r"""
    The Domain Adversarial Loss proposed in
    `Domain-Adversarial Training of Neural Networks (ICML 2015) <https://arxiv.org/abs/1505.07818>`_

    Domain adversarial loss measures the domain discrepancy through training a domain discriminator.
    Given domain discriminator :math:`D`, feature representation :math:`f`, the definition of DANN loss is

    .. math::
        loss(\mathcal{D}_s, \mathcal{D}_t) = \mathbb{E}_{x_i^s \sim \mathcal{D}_s} \text{log}[D(f_i^s)]
            + \mathbb{E}_{x_j^t \sim \mathcal{D}_t} \text{log}[1-D(f_j^t)].

    Args:
        domain_discriminator (torch.nn.Module): A domain discriminator object, which predicts the domains of features. Its input shape is (N, F) and output shape is (N, 1)
        reduction (str, optional): Specifies the reduction to apply to the output:
            ``'none'`` | ``'mean'`` | ``'sum'``. ``'none'``: no reduction will be applied,
            ``'mean'``: the sum of the output will be divided by the number of
            elements in the output, ``'sum'``: the output will be summed. Default: ``'mean'``
        grl (WarmStartGradientReverseLayer, optional): Default: None.

    Inputs:
        - f_s (tensor): feature representations on source domain, :math:`f^s`
        - f_t (tensor): feature representations on target domain, :math:`f^t`
        - w_s (tensor, optional): a rescaling weight given to each instance from source domain.
        - w_t (tensor, optional): a rescaling weight given to each instance from target domain.

    Shape:
        - f_s, f_t: :math:`(N, F)` where F means the dimension of input features.
        - Outputs: scalar by default. If :attr:`reduction` is ``'none'``, then :math:`(N, )`.

    Examples::

        >>> from tllib.modules.domain_discriminator import DomainDiscriminator
        >>> discriminator = DomainDiscriminator(in_feature=1024, hidden_size=1024)
        >>> loss = DomainAdversarialLoss(discriminator, reduction='mean')
        >>> # features from source domain and target domain
        >>> f_s, f_t = torch.randn(20, 1024), torch.randn(20, 1024)
        >>> # If you want to assign different weights to each instance, you should pass in w_s and w_t
        >>> w_s, w_t = torch.randn(20), torch.randn(20)
        >>> output = loss(f_s, f_t, w_s, w_t)
    """

    def __init__(self, domain_discriminator: nn.Module, reduction: Optional[str] = 'mean',
                 grl: Optional = None, sigmoid=True):
        super(DomainAdversarialLoss, self).__init__()
        self.grl = WarmStartGradientReverseLayer(alpha=1., lo=0., hi=1., max_iters=1000, auto_step=True) if grl is None else grl
        self.domain_discriminator = domain_discriminator
        self.sigmoid = sigmoid
        self.reduction = reduction
        self.bce = lambda input, target, weight: \
            F.binary_cross_entropy(input, target, weight=weight, reduction=reduction)
        self.domain_discriminator_accuracy = None

    def forward(self, f_s: torch.Tensor, f_t: torch.Tensor,
                w_s: Optional[torch.Tensor] = None, w_t: Optional[torch.Tensor] = None) -> torch.Tensor:
        f = self.grl(torch.cat((f_s, f_t), dim=0))
        d = self.domain_discriminator(f)
        if self.sigmoid:
            d_s, d_t = d.chunk(2, dim=0)
            d_label_s = torch.ones((f_s.size(0), 1)).to(f_s.device)
            d_label_t = torch.zeros((f_t.size(0), 1)).to(f_t.device)
            self.domain_discriminator_accuracy = 0.5 * (
                        binary_accuracy(d_s, d_label_s) + binary_accuracy(d_t, d_label_t))

            if w_s is None:
                w_s = torch.ones_like(d_label_s)
            if w_t is None:
                w_t = torch.ones_like(d_label_t)
            return 0.5 * (
                F.binary_cross_entropy(d_s, d_label_s, weight=w_s.view_as(d_s), reduction=self.reduction) +
                F.binary_cross_entropy(d_t, d_label_t, weight=w_t.view_as(d_t), reduction=self.reduction)
            )
        else:
            d_label = torch.cat((
                torch.ones((f_s.size(0),)).to(f_s.device),
                torch.zeros((f_t.size(0),)).to(f_t.device),
            )).long()
            if w_s is None:
                w_s = torch.ones((f_s.size(0),)).to(f_s.device)
            if w_t is None:
                w_t = torch.ones((f_t.size(0),)).to(f_t.device)
            self.domain_discriminator_accuracy = accuracy(d, d_label)
            loss = F.cross_entropy(d, d_label, reduction='none') * torch.cat([w_s, w_t], dim=0)
            if self.reduction == "mean":
                return loss.mean()
            elif self.reduction == "sum":
                return loss.sum()
            elif self.reduction == "none":
                return loss
            else:
                raise NotImplementedError(self.reduction)
