import torch
from torch import nn
from torch.nn import functional as func


class DiscriminativeLoss(nn.Module):
    def __init__(self, conf):
        super(DiscriminativeLoss, self).__init__()
        self.scale_var = conf.LOSS.PARAMS.SCALE_VAR
        self.scale_dist = conf.LOSS.PARAMS.SCALE_DIST
        self.scale_reg = conf.LOSS.PARAMS.SCALE_REG

        self.embed_dim = conf.MODEL.PARAMS.VECTOR_SIZE
        self.delta_v = conf.LOSS.PARAMS.DELTA_V
        self.delta_d = conf.LOSS.PARAMS.DELTA_D

    def _forward(self, embedding, seg_gt):
        batch_size = embedding.shape[0]

        # varience loss, distance loss, regularisation loss.
        var_loss = torch.tensor(0, dtype=embedding.dtype, device=embedding.device)
        dist_loss = torch.tensor(0, dtype=embedding.dtype, device=embedding.device)
        reg_loss = torch.tensor(0, dtype=embedding.dtype, device=embedding.device)

        for b in range(batch_size):
            embedding_b = embedding[b]  # (embed_dim, H, W)
            seg_gt_b = seg_gt[b]

            # print(f'seg_gt_b shape: {seg_gt_b.shape}')

            labels = torch.unique(seg_gt_b)
            labels = labels[labels != 0]
            # print(f'labels: {labels}')
            num_classes = len(labels)
            if num_classes == 0:
                # please refer to issue here: https://github.com/harryhan618/LaneNet/issues/12
                _nonsense = embedding.sum()
                _zero = torch.zeros_like(_nonsense)
                var_loss = var_loss + _nonsense * _zero
                dist_loss = dist_loss + _nonsense * _zero
                reg_loss = reg_loss + _nonsense * _zero
                continue

            centroid_mean = []
            for label_idx in labels:
                seg_mask_i = (seg_gt_b == label_idx)
                if not seg_mask_i.any():
                    continue
                embedding_i = embedding_b[:, seg_mask_i]

                mean_i = torch.mean(embedding_i, dim=1)
                centroid_mean.append(mean_i)

                # ---------- var_loss -------------
                var_loss = var_loss + torch.mean(func.relu(
                    torch.norm(embedding_i - mean_i.reshape(self.embed_dim, 1), dim=0) - self.delta_v) ** 2) / num_classes
            centroid_mean = torch.stack(centroid_mean)  # (n_lane, embed_dim)

            if num_classes > 1:
                centroid_mean1 = centroid_mean.reshape(-1, 1, self.embed_dim)
                centroid_mean2 = centroid_mean.reshape(1, -1, self.embed_dim)
                dist = torch.norm(centroid_mean1 - centroid_mean2, dim=2)  # shape (num_classes, num_classes)
                dist = dist + torch.eye(num_classes, dtype=dist.dtype,
                                        device=dist.device) * self.delta_d  # diagonal elements are 0,
                # now mask above delta_d

                # divided by two for double calculated loss above, for implementation convenience
                dist_loss = dist_loss + torch.sum(func.relu(-dist + self.delta_d) ** 2) / (
                            num_classes * (num_classes - 1)) / 2

            # reg_loss is not used in original paper
            # reg_loss = reg_loss + torch.mean(torch.norm(centroid_mean, dim=1))

        var_loss = var_loss / batch_size
        dist_loss = dist_loss / batch_size
        reg_loss = reg_loss / batch_size

        total_loss = var_loss * self.scale_var + dist_loss * self.scale_dist + self.scale_reg * reg_loss
        return {'total': total_loss, 'var': var_loss, 'dist': dist_loss, 'reg': reg_loss}

    def forward(self, model_out, gt):
        semantic_emb, instance_emb = model_out
        semantic_gt = gt[:, 0]#.unsqueeze(1)
        instance_gt = gt[:, 1]#.unsqueeze(1)

        sem_loss = self._forward(semantic_emb, semantic_gt)
        inst_loss = self._forward(instance_emb, instance_gt)
        loss = sem_loss['total'] + inst_loss['total']
        return loss
