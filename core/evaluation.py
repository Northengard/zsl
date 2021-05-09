import torch
from tqdm import tqdm
from utils import AverageMeter


def get_label_vs_support(sup_matr, nn_output, threshold=None):
    cos_sim = torch.cosine_similarity(sup_matr, nn_output, dim=1)
    pred_label = cos_sim.argmax().item()
    if threshold:
        if cos_sim[pred_label] < threshold:
            return -1, cos_sim[pred_label]
    return pred_label, cos_sim[pred_label]


def evaluation(model, dataloader, support_matrix, device, threshold):
    model.eval()
    num_iter = len(dataloader)
    num_classes = support_matrix.shape[0]
    conf_matr = torch.zeros(num_classes, num_classes)
    tq = tqdm(total=num_iter)
    tq.set_description(f'Evaluation:')
    loss_handler = AverageMeter()
    with torch.no_grad():
        for itr, sample in enumerate(dataloader):
            image = sample['image']
            image = image.to(device)
            image_label = sample['image_labels'].to(device)

            output = model(image)

            pred_label, similarity = get_label_vs_support(support_matrix, output, threshold=threshold)

            conf_matr[image_label.item(), pred_label] += 1
            tq.update(1)
            tq.set_postfix(avg_loss=loss_handler.avg)
    tq.close()

    return conf_matr.numpy()
