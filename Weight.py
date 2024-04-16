import numpy as np
import torch
# from Config import class_num
from cvxopt import matrix, solvers

def convert_to_onehot(sca_label, class_num=31):
    return np.eye(class_num)[sca_label]

class Weight:
    
    def __init__(self, ma=0.0, class_num=31, **kwargs):
        self.ma = ma
        self.im_weights = torch.nn.Parameter(torch.ones(class_num,1), requires_grad=False)
    def im_weights_update(self, source_y, target_y, cov, device=None):
        """
        Solve a Quadratic Program to compute the optimal importance weight under the generalized label shift assumption.
        :param source_y:    The marginal label distribution of the source domain.
        :param target_y:    The marginal pseudo-label distribution of the target domain from the current classifier.
        :param cov:         The covariance matrix of predicted-label and true label of the source domain.
        :param device:      Device of the operation.
        :return:
        """
        # Convert all the vectors to column vectors.
        dim = cov.shape[0]
        source_y = source_y.reshape(-1, 1).astype(np.double)
        target_y = target_y.reshape(-1, 1).astype(np.double)
        cov = cov.astype(np.double)

        P = matrix(np.dot(cov.T, cov), tc="d")
        q = -matrix(np.dot(cov.T, target_y), tc="d") # q = -matrix(np.dot(cov, target_y), tc="d") 
        G = matrix(-np.eye(dim), tc="d")
        h = matrix(np.zeros(dim), tc="d")
        A = matrix(source_y.reshape(1, -1), tc="d")
        b = matrix([1.0], tc="d")
        sol = solvers.qp(P, q, G, h, A, b)
        new_im_weights = np.array(sol["x"])

        # EMA for the weights
        self.im_weights = (1 - self.ma) * torch.tensor(new_im_weights, dtype=torch.float32) + self.ma * self.im_weights

        return self.im_weights
    
    def im_weights_update_plus(self, source_y, target_y, cov, device=None):
        """
        Solve a Quadratic Program to compute the optimal importance weight under the generalized label shift assumption.
        :param source_y:    The marginal label distribution of the source domain.
        :param target_y:    The marginal pseudo-label distribution of the target domain from the current classifier.
        :param cov:         The covariance matrix of predicted-label and true label of the source domain.
        :param device:      Device of the operation.
        :return:
        """
        # Convert all the vectors to column vectors.
        dim = cov.shape[0]
        source_y = source_y.reshape(-1, 1).astype(np.double)
        target_y = target_y.reshape(-1, 1).astype(np.double)
        cov = cov.astype(np.double)

        P = matrix(np.dot(cov.T, cov), tc="d")
        q = -matrix(np.dot(cov.T, target_y), tc="d")  
        G = matrix(-np.eye(dim), tc="d")
        h = matrix(np.zeros(dim), tc="d")
        # A = matrix((source_y/target_y).reshape(1,-1), tc="d")
        # b = matrix([dim], tc="d")
        A = matrix(np.concatenate(( source_y.reshape(1, -1),(source_y/target_y).reshape(1,-1))), tc="d")
        b = matrix([1.0, dim], tc="d")
        sol = solvers.qp(P, q, G, h, A, b)
        new_im_weights = np.array(sol["x"])

        # EMA for the weights
        self.im_weights = (1 - self.ma) * torch.tensor(new_im_weights, dtype=torch.float32) + self.ma * self.im_weights

        return self.im_weights
           
    @staticmethod
    def cal_joint_weight(s_label, true_t_label, t_label, src_cls_dis, type='visual', batch_size=32, class_num=31, gamma=0):
        batch_size = s_label.size()[0]
        s_sca_label = s_label.cpu().data.numpy()
        s_vec_label = convert_to_onehot(s_sca_label, class_num)
        # source class weight        
        s_sum = np.sum(s_vec_label, axis=0).reshape(1, class_num)
        s_class_dis = s_sum/np.sum(s_sum)          
        s_sum[s_sum == 0] = 100
        s_vec_label = s_vec_label / s_sum

        # True Target Label
        # true_t_sca_label = true_t_label.cpu().data.numpy()
        # t_sca_label = true_t_sca_label
        # true_t_vec_label = convert_to_onehot(true_t_sca_label)
        # true_t_sum = np.sum(true_t_vec_label, axis=0).reshape(1,class_num)
        # t_class_dis = true_t_sum / np.sum(true_t_sum)
        # true_t_sum[true_t_sum ==0 ] =1
        # t_vec_label = true_t_vec_label / true_t_sum
               
        # Pseudo Target Label One-Hot
        t_sca_label = t_label.cpu().data.max(1)[1].numpy()
        t_vec_label = convert_to_onehot(t_sca_label, class_num)
        t_sum = np.sum(t_vec_label, axis=0).reshape(1, class_num)
        
        t_class_dis = t_sum / np.sum(t_sum)        
        #目标类别概率分布
        t_sum[t_sum==0] = 1        
        t_vec_label = t_label.cpu().data.numpy() * t_vec_label#使用y_t_hat
        t_vec_label = t_vec_label / t_sum
        

                
        # 目标域样本按照类别进行归一化操作
# =============================================================================
        t_vec_label_sum = np.sum(t_vec_label, axis=0).reshape(1, class_num)
        t_vec_label_sum[t_vec_label_sum==0] = 1
        t_vec_label = t_vec_label / t_vec_label_sum
# =============================================================================               

        weight_ss = np.zeros((batch_size, batch_size))
        weight_tt = np.zeros((batch_size, batch_size))
        weight_st = np.zeros((batch_size, batch_size))

        set_s = set(s_sca_label)
        set_t = set(t_sca_label)
        count = 0
        for i in range(class_num):
            #if i in set_s:
            if i in set_s and i in set_t:
                s_tvec = s_vec_label[:, i].reshape(batch_size, -1)
                t_tvec = t_vec_label[:, i].reshape(batch_size, -1)
                
                ss = np.dot(s_tvec, s_tvec.T)
                weight_ss = weight_ss + ss# / np.sum(s_tvec) / np.sum(s_tvec)
                tt = np.dot(t_tvec, t_tvec.T)
                weight_tt = weight_tt + tt# / np.sum(t_tvec) / np.sum(t_tvec)
                st = np.dot(s_tvec, t_tvec.T)
                weight_st = weight_st + st# / np.sum(s_tvec) / np.sum(t_tvec)
                count += 1

        length = count  # len( set_s ) * len( set_t )
        if length != 0:
            weight_ss = weight_ss / length
            weight_tt = weight_tt / length
            weight_st = weight_st / length
        else:
            weight_ss = np.array([0])
            weight_tt = np.array([0])
            weight_st = np.array([0])
        weight_ss = weight_ss.astype('float32')
        weight_tt = weight_tt.astype('float32')
        weight_st = weight_st.astype('float32')
        return weight_ss, weight_tt, weight_st

    @staticmethod
    def cal_weight(s_label, t_label, type='visual', batch_size=32, class_num=31):
        batch_size = s_label.size()[0]
        s_sca_label = s_label.cpu().data.numpy()
        s_vec_label = convert_to_onehot(s_sca_label, class_num)
        s_sum = np.sum(s_vec_label, axis=0).reshape(1, class_num)
        s_sum[s_sum == 0] = 100
        s_vec_label = s_vec_label / s_sum

        t_sca_label = t_label.cpu().data.max(1)[1].numpy()
        #t_vec_label = convert_to_onehot(t_sca_label)

        t_vec_label = t_label.cpu().data.numpy()
        t_sum = np.sum(t_vec_label, axis=0).reshape(1, class_num)
        t_sum[t_sum == 0] = 100
        t_vec_label = t_vec_label / t_sum

        weight_ss = np.zeros((batch_size, batch_size))
        weight_tt = np.zeros((batch_size, batch_size))
        weight_st = np.zeros((batch_size, batch_size))

        set_s = set(s_sca_label)
        set_t = set(t_sca_label)
        count = 0
        for i in range(class_num):
            if i in set_s and i in set_t:
                s_tvec = s_vec_label[:, i].reshape(batch_size, -1)
                t_tvec = t_vec_label[:, i].reshape(batch_size, -1)
                ss = np.dot(s_tvec, s_tvec.T)
                weight_ss = weight_ss + ss# / np.sum(s_tvec) / np.sum(s_tvec)
                tt = np.dot(t_tvec, t_tvec.T)
                weight_tt = weight_tt + tt# / np.sum(t_tvec) / np.sum(t_tvec)
                st = np.dot(s_tvec, t_tvec.T)
                weight_st = weight_st + st# / np.sum(s_tvec) / np.sum(t_tvec)
                count += 1

        length = count  # len( set_s ) * len( set_t )
        if length != 0:
            weight_ss = weight_ss / length
            weight_tt = weight_tt / length
            weight_st = weight_st / length
        else:
            weight_ss = np.array([0])
            weight_tt = np.array([0])
            weight_st = np.array([0])
        return weight_ss.astype('float32'), weight_tt.astype('float32'), weight_st.astype('float32')
    @staticmethod
    def cal_wmmd_weight(s_label, t_label, type='visual', batch_size=32, class_num=31):
        batch_size = s_label.size()[0]
        s_sca_label = s_label.cpu().data.numpy()
        s_vec_label = convert_to_onehot(s_sca_label, class_num)
        # source class weight        
        s_sum = np.sum(s_vec_label, axis=0).reshape(1, class_num)
        s_class_dis = s_sum/np.sum(s_sum)  
        
        s_sum[s_sum == 0] = 100
        s_vec_label = s_vec_label / s_sum
        
        #目标域预测结果One-Hot编码
        t_sca_label = t_label.cpu().data.max(1)[1].numpy()
        t_vec_label = convert_to_onehot(t_sca_label, class_num)
        t_sum = np.sum(t_vec_label, axis=0).reshape(1, class_num)
        t_class_dis = t_sum / np.sum(t_sum)
        #目标类别概率分布
        t_sum[t_sum==0] = 1
        
        t_vec_label = t_label.cpu().data.numpy() * t_vec_label#使用y_t_hat
        #t_vec_label = H_weight_cpu * t_vec_label # 使用样本分类结果进行熵加权
        
        t_vec_label = t_vec_label / t_sum
        
                
        # 目标域样本按照类别进行归一化操作
# =============================================================================
        t_vec_label_sum = np.sum(t_vec_label, axis=0).reshape(1, class_num)
        t_vec_label_sum[t_vec_label_sum==0] = 1
        t_vec_label = t_vec_label / t_vec_label_sum
# =============================================================================
        weight_ss = np.zeros((batch_size, batch_size))
        weight_tt = np.zeros((batch_size, batch_size))
        weight_st = np.zeros((batch_size, batch_size))

        set_s = set(s_sca_label)
        set_t = set(t_sca_label)
        count = 0
        for i in range(class_num):
            #if i in set_s:
            if i in set_s and i in set_t:
                s_tvec = s_vec_label[:, i].reshape(batch_size, -1)
                t_tvec = t_vec_label[:, i].reshape(batch_size, -1)
                
                ss = np.dot(s_tvec, s_tvec.T)
                weight_ss = weight_ss + ss# / np.sum(s_tvec) / np.sum(s_tvec)
                tt = np.dot(t_tvec, t_tvec.T)
                weight_tt = weight_tt + tt# / np.sum(t_tvec) / np.sum(t_tvec)
                st = np.dot(s_tvec, t_tvec.T)
                weight_st = weight_st + st# / np.sum(s_tvec) / np.sum(t_tvec)
                count += 1

        length = count  # len( set_s ) * len( set_t )
        if length != 0:
            weight_ss = weight_ss / length
            weight_tt = weight_tt / length
            weight_st = weight_st / length
        else:
            weight_ss = np.array([0])
            weight_tt = np.array([0])
            weight_st = np.array([0])
        weight_ss = weight_ss.astype('float32')
        weight_tt = weight_tt.astype('float32')
        weight_st = weight_st.astype('float32')
        ts_class_weight = ts_class_weight.astype('float32')
        return weight_ss, weight_tt, weight_st
