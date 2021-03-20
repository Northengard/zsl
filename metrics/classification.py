import torch


def get_confusion_matix(vector_pairs, labels, threshold=0.4):
    matrix = torch.zeros(4)
    matrix = matrix.to(vector_pairs.device)
    vector_distances = 1 - torch.cosine_similarity(vector_pairs[:, 0], vector_pairs[:, 1], 1)
    matched = (vector_distances < threshold).to(torch.int32)
    confusion_values = torch.bincount(labels.to(torch.int32) * 2 + matched)

    matrix[:confusion_values.shape[0]] += confusion_values
    matrix = matrix.resize(2, 2)

    return matrix.to('cpu')


def accuracy(conf_matr):
    acc = torch.diag(conf_matr).sum() / conf_matr.sum()
    return {"accuracy": acc.item()}


def precision_recall(conf_matr):
    """
    |TN|FN|
    |FP|TP|
    :param conf_matr:
    :return:
    """
    rec = conf_matr[1, 1] / conf_matr.sum(axis=1)[1]
    prec = conf_matr[1, 1] / (conf_matr.sum(axis=0)[1] + 1e-6)
    return {'precision': prec.item(), 'recall': rec.item()}


def prec_rec_acc(conf_matr):
    output = precision_recall(conf_matr)
    output.update(accuracy(conf_matr))
    return output
