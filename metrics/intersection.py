import torch
import numpy as np


def get_confusion_matrix(label, pred, num_class, ignore=-1):
    """
    calculate the confusion matix by given label and pred
    """
    device = pred.device

    ignore_index = label != ignore
    seg_gt = label[ignore_index].to(torch.int32)
    seg_pred = pred[ignore_index].to(torch.int32)

    # 1 + because 0 - background, + actual classes
    index = (seg_gt * (num_class + 1) + seg_pred)
    confusion_matrix = torch.bincount(index)
    confusion_matrix = torch.cat([confusion_matrix,
                                  torch.zeros((num_class + 1) * (num_class + 1) - confusion_matrix.shape[0],
                                              device=device)])
    confusion_matrix = torch.reshape(confusion_matrix, ((num_class + 1), (num_class + 1)))
    return confusion_matrix


def get_iou_metrics(confusion_matrix):
    """
    Our matrix is :
       pred classes
    g | TN | FP |
    t | FN | TP |
    columns - predict data
    rows - ground truth data
    (according to FP FN positions)
    :param confusion_matrix: np.array(N, N)
    :return: per_class IoU, mean_iou
    """
    tp_values = np.diag(confusion_matrix)
    gt_condition = confusion_matrix.sum(axis=1)
    pred_condition = confusion_matrix.sum(axis=0)
    total_sum = confusion_matrix.sum()

    total_acc = tp_values.sum() / total_sum
    mean_cls_acc = tp_values / gt_condition
    mean_cls_acc = np.nanmean(mean_cls_acc)
    iu = tp_values / (gt_condition + pred_condition - tp_values)
    valid = gt_condition > 0
    mean_iu = np.nanmean(iu[valid])
    freq = gt_condition / total_sum
    freq_mask = freq > 0
    fwavacc = (freq[freq_mask] * iu[freq_mask]).sum()
    return {'overall_accuracy': float(total_acc),
            'mean_accuracy': float(mean_cls_acc),
            'freqW_acc': float(fwavacc),
            'mean_iou': float(mean_iu)}
