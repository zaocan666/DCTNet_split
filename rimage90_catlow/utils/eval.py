from __future__ import print_function, absolute_import
import numpy as np
__all__ = ['accuracy', 'cal_his_acc']

def accuracy(output, target, topk=(1,)):
    """Computes the precision@k for the specified values of k"""
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].view(-1).float().sum(0)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res

def cal_his_acc(bounds_updown, output_all, target_all, histogram_all_flag=False):
    max_indexes = np.argmax(output_all, axis=1) #[10000]
    max_value = output_all[[i for i in range(output_all.shape[0])], max_indexes] #[10000]
    correct_max_value = max_value[max_indexes==target_all]
    correct_num = correct_max_value.shape[0]
    # print('correct acc: %.2f'%(correct_num/output_all.shape[0]))

    histogram_correct = [] #portion histogram of prob confidence [0~10%, 10%~20%, ..., 90%~100%]
    value_acc = [] #acc of each prob confidence
    bound_num = [] #num of samples belonging to each bound
    for bound in bounds_updown:
        bound_up = bound[1]
        bound_down = bound[0]

        if not histogram_all_flag:
            bound_flag_correct = ((correct_max_value<=bound_up) & (correct_max_value>bound_down))
            histogram_correct.append(bound_flag_correct.sum()/correct_num)
        else:
            bound_flag_correct = ((max_value<=bound_up) & (max_value>bound_down))
            histogram_correct.append(bound_flag_correct.sum()/max_value.shape[0])

        bound_flag_all = ((max_value<=bound_up) & (max_value>bound_down))
        if not bound_flag_all.any():
            value_acc.append(0.0)
            bound_num.append(0)
            continue
        bound_indexes = max_indexes[bound_flag_all]
        bound_target = target_all[bound_flag_all]
        bound_correct = (bound_indexes==bound_target)
        value_acc.append(bound_correct.sum()/bound_flag_all.sum())
        bound_num.append(bound_flag_all.sum())
    
    return histogram_correct, value_acc, correct_num, bound_num