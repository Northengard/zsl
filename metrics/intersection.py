import torch
import numpy as np


def get_confision_matrix(label, pred, num_class, ignore=-1, seg_threshold=0.5):
    """
    calculate the confusion matix by given label and pred
    """
    device = pred.device
    threshold = torch.tensor(seg_threshold, device=device)
    seg_pred = torch.ge(pred, threshold).to(torch.int32)
    seg_gt = torch.ge(label, threshold).to(torch.int32)

    ignore_index = seg_gt != ignore
    seg_gt = seg_gt[ignore_index]
    seg_pred = seg_pred[ignore_index]

    index = (seg_gt * num_class + seg_pred)
    label_count = torch.bincount(index)
    confusion_matrix = torch.zeros((num_class, num_class), device=device)

    for i_label in range(num_class):
        for i_pred in range(num_class):
            cur_idx = i_label * num_class + i_pred
            if cur_idx < len(label_count):
                confusion_matrix[i_label, i_pred] = label_count[cur_idx]
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
    return {'overall_accuracy': total_acc,
            'mean_accuracy': mean_cls_acc,
            'freqW_acc': fwavacc,
            'mean_iou': mean_iu}
