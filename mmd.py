# -*- coding: utf-8 -*-
"""
Created on Wed Nov  2 10:12:12 2022
Author : Pei Wang
Based on https://github.com/easezyc/deep-transfer-learning/blob/master/UDA/pytorch1.0/DSAN/lmmd.py
"""


import torch
import torch.nn as nn
import numpy as np

class MaximumMeanDiscrepancy(nn.Module):
    def __init__(self, class_num=31, device=None, kernel_type='rbf', kernel_mul=2.0, kernel_num=5, fix_sigma=None):
        super(MaximumMeanDiscrepancy, self).__init__()
        self.class_num = class_num
        self.kernel_num = kernel_num
        self.kernel_mul = kernel_mul
        self.fix_sigma = fix_sigma
        self.kernel_type = kernel_type
        if device:
            self.device = device
        else:
            self.divece = torch.device("cuda" if torch.cuda.is_availabel() else "cpu")

    def guassian_kernel(self, source, target, kernel_mul=2.0, kernel_num=5, fix_sigma=None):
        n_samples = int(source.size()[0]) + int(target.size()[0])
        total = torch.cat([source, target], dim=0)
        total0 = total.unsqueeze(0).expand(
            int(total.size(0)), int(total.size(0)), int(total.size(1)))
        total1 = total.unsqueeze(1).expand(
            int(total.size(0)), int(total.size(0)), int(total.size(1)))
        L2_distance = ((total0-total1)**2).sum(2)
        if fix_sigma:
            bandwidth = fix_sigma
        else:
            bandwidth = torch.sum(L2_distance.data) / (n_samples**2-n_samples)
        bandwidth /= kernel_mul ** (kernel_num // 2)
        bandwidth_list = [bandwidth * (kernel_mul**i)
                          for i in range(kernel_num)]
        kernel_val = [torch.exp(-L2_distance / bandwidth_temp)
                      for bandwidth_temp in bandwidth_list]
        return sum(kernel_val)
    
    def mmd_loss(self, source, target):
        if self.kernel_type == 'linear':
            return self.linear_mmd(source, target)
        elif self.kernel_type == 'rbf':
            source_size = int(source.size()[0])
            target_size = int(target.size()[0])
            kernels = self.guassian_kernel(
                source, target, kernel_mul=self.kernel_mul, kernel_num=self.kernel_num, fix_sigma=self.fix_sigma)
            XX = torch.mean(kernels[:source_size, :source_size])
            YY = torch.mean(kernels[source_size:, source_size:])
            XY = torch.mean(kernels[:source_size, source_size:])
            YX = torch.mean(kernels[source_size:, :source_size])
            loss = torch.sum(XX + YY - XY - YX)

            return loss

    def lmmd_loss(self, source, target, s_label, t_label, is_pseudo_target=True):
        if self.kernel_type == 'linear':
            return self.linear_mmd(source, target)
        elif self.kernel_type == 'rbf':
            source_size = source.size()[0]
            target_size = target.size()[0]
            weight_ss, weight_tt, weight_st = self.cal_weight(
                s_label, t_label, source_size=source_size, target_size=target_size, class_num=self.class_num, pseudo_target=is_pseudo_target)
            weight_ss = torch.from_numpy(weight_ss).to(self.device)
            weight_tt = torch.from_numpy(weight_tt).to(self.device)
            weight_st = torch.from_numpy(weight_st).to(self.device)
    
            kernels = self.guassian_kernel(source, target,
                                    kernel_mul=self.kernel_mul, kernel_num=self.kernel_num, fix_sigma=self.fix_sigma)
            loss = torch.Tensor([0]).to(self.device)
            if torch.sum(torch.isnan(sum(kernels))):
                return loss
            SS = kernels[:source_size, :source_size]
            TT = kernels[source_size:, source_size:]
            ST = kernels[:source_size, source_size:]
    
            loss += torch.sum(weight_ss * SS) + torch.sum(weight_tt * TT) - 2 * torch.sum(weight_st * ST)
            return loss

    def convert_to_onehot(self, sca_label, class_num=31):
        return np.eye(class_num)[sca_label]

    def cal_weight(self, s_label, t_label, source_size=32, target_size=32, class_num=31, pseudo_target=True):
        s_sca_label = s_label.cpu().data.numpy()
        s_vec_label = self.convert_to_onehot(s_sca_label, class_num=self.class_num)
        s_sum = np.sum(s_vec_label, axis=0).reshape(1, class_num)
        s_sum[s_sum == 0] = 100
        s_vec_label = s_vec_label / s_sum

        if pseudo_target:
            t_sca_label = t_label.cpu().data.max(1)[1].numpy()
            t_vec_label = t_label.cpu().data.numpy()
            t_sum = np.sum(t_vec_label, axis=0).reshape(1, class_num)
            t_sum[t_sum == 0] = 100
            t_vec_label = t_vec_label / t_sum
        else:
            t_sca_label = t_label.cpu().data.numpy()
            t_vec_label = self.convert_to_onehot(t_sca_label, class_num=self.class_num)
            t_sum = np.sum(t_vec_label, axis=0).reshape(1, class_num)
            t_sum[t_sum == 0] = 100
            t_vec_label = t_vec_label / t_sum


        index = list(set(s_sca_label) & set(t_sca_label))
        mask_arr_s = np.zeros((source_size, class_num))
        mask_arr_t = np.zeros((target_size, class_num))
        mask_arr_s[:, index] = 1
        mask_arr_t[:, index] = 1
        s_vec_label = s_vec_label * mask_arr_s
        t_vec_label = t_vec_label * mask_arr_t

        weight_ss = np.matmul(s_vec_label, s_vec_label.T)
        weight_tt = np.matmul(t_vec_label, t_vec_label.T)
        weight_st = np.matmul(s_vec_label, t_vec_label.T)

        length = len(index)
        if length != 0:
            weight_ss = weight_ss / length
            weight_tt = weight_tt / length
            weight_st = weight_st / length
        else:
            weight_ss = np.array([0])
            weight_tt = np.array([0])
            weight_st = np.array([0])
        return weight_ss.astype('float32'), weight_tt.astype('float32'), weight_st.astype('float32')
