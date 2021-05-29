import cv2
import numpy as np
import torch
from time import perf_counter
from torchvision.ops import boxes as box_ops

from datasets.transformations import get_pad

SQ_2PI = np.sqrt(2 * np.pi)


def gaussian(dist, bandwidth):
    global SQ_2PI
    return torch.exp(-0.5 * (dist / bandwidth) ** 2) / (bandwidth * SQ_2PI)


def rescale_coords(coords, img_shape, orig_img_shape):
    """Resize coords to original image shape.

    :param coords: ndarray, coords to rescale
    :param img_shape: iterable, current (h,w)
    :param orig_img_shape: iterable, desired (h,w)
    """
    img_h, img_w = img_shape
    src_h, src_w = orig_img_shape
    _, pad_h, _, pad_w = get_pad(img_h, img_w, src_h, src_w)
    src_h += pad_h
    resize_coef = img_h / src_h
    return coords / resize_coef


def cosine_similarity(matr1, matr2, eps=1e-8):
    """Calculate cosine similarity matrix for the given 2d tensors.

    :param matr1: Tensor, first 2d tensor
    :param matr2: Tensor, second 2d tensor
    :param eps: parameter for stable computations
    """
    a_n, b_n = matr1.norm(dim=1)[:, None], matr2.norm(dim=1)[:, None]
    a_norm = matr1 / torch.clamp(a_n, min=eps)
    b_norm = matr2 / torch.clamp(b_n, min=eps)
    sim_mt = torch.mm(a_norm, b_norm.transpose(0, 1))
    return sim_mt


def mean_shift(data, centroids, bandwidth=1.5, matching_bandwidht=0.5, n_steps=0, clusterize_all=False, tol=1e-6):
    """Applies MeanShift algorithm to the given data.

    :param data: Tensor, first 2d tensor
    Data, for clustering.
    :param centroids: Tensor, 2d tensor,
    Class vectors used to initialize kernels.
    :param bandwidth: float, default=1.5,
    Bandwidth of gaussians, generated for each point. Required for mean shift step.
    :param matching_bandwidht: float, default=0.5,
    If weight of sample < bandwidth then sample belongs to that cluster.
    :param n_steps: int, default=0,
    Number of steps for shifting centroids to data. It became more robust to outliers (in theory), but decrease speed.
    :param clusterize_all: bool, default=True,
    If true, then all points are clustered, even those orphans that are not within any kernel.
    Orphans are assigned to the nearest kernel. If false, then orphans are given cluster label -1.
    :param tol: float, default=1e-6,
    stop convergence parameter.
    """
    shifted_centroids = torch.clone(centroids)
    for step in range(n_steps):
        dist = torch.cdist(shifted_centroids[None], data[None], p=2)[0]
        weight = gaussian(dist, bandwidth)
        num = (weight[:, :, None] * data).sum(dim=1)
        new_centroids = num / weight.sum(1)[:, None]
        if torch.abs(shifted_centroids - new_centroids).sum() < tol:
            shifted_centroids = new_centroids
            break
        else:
            shifted_centroids = new_centroids
    scores = torch.cdist(shifted_centroids[None], data[None], p=2)[0]
    weight = gaussian(scores, matching_bandwidht)
    weight = weight
    classes = weight.argmax(0)

    # smoothed_classes = list()
    # for chunk in data.reshape(640, -1, data.shape[-1]):
    #     top_d = torch.cdist(chunk[None], data[None], p=2)[0]
    #     top_d = top_d.topk(1, largest=False)[1]
    #     for knn_set in top_d:
    #         cls, n_cls = torch.unique(classes[knn_set], return_counts=True)
    #         smoothed_classes.append(cls[n_cls.argmax()])
    # classes = torch.stack(smoothed_classes)
    scores = weight[classes, torch.arange(0, weight.shape[1])]

    if not clusterize_all:
        score_mask = scores < gaussian(torch.tensor(matching_bandwidht), matching_bandwidht).item()
        classes[score_mask] = -1
    return classes, scores


class BottomUpPostprocessing:
    def __init__(self, classes_ids, delta_v, delta_d, embedding_size, device):
        self.num_classes = len(classes_ids)
        self.classes_ids = classes_ids
        self._cat_maper = dict(zip(list(range(1, len(self.classes_ids) + 1)), self.classes_ids))
        self.delta_v = delta_v
        self.delta_d = delta_d
        self.embedding_size = embedding_size
        self.device = device
        self._class_containers = {class_id: torch.zeros(embedding_size).to(device)
                                  for class_id in self._cat_maper.keys()}
        self._class_counters = dict.fromkeys(self._cat_maper.keys(), 0)
        self.nms_thresh = 0.1

    def update_centriods(self, net_output, true_labels):
        classes = torch.unique(true_labels)
        for class_id in classes:
            if class_id == 0:
                continue
            mask = true_labels == class_id
            # BCHW --> BHWC and mean over kept pixels
            self._class_containers[int(class_id.item())] += net_output.permute(0, 2, 3, 1)[mask].mean(0)
            self._class_counters[int(class_id.item())] += 1

    @staticmethod
    def _get_bbox(coord_array):
        left = coord_array[coord_array[..., 0].argmin()][0][0]
        top = coord_array[coord_array[..., 1].argmin()][0][1]

        right = coord_array[coord_array[..., 0].argmax()][0][0]
        bottom = coord_array[coord_array[..., 1].argmax()][0][1]
        # bbox
        return np.stack([left, top, right, bottom])

    def _get_top_border_coords(self, target_val_mask, is_reversed=False):
        map_h, map_w = target_val_mask.shape[-2:]
        # Y coords of the top border
        amax = target_val_mask.argmax(-2)
        masks = amax > 0
        coord_arrays = list()
        for img_id, mask in enumerate(masks):
            if is_reversed:
                coord_arrays.append(
                    torch.stack([torch.arange(map_w, device=self.device), map_h - amax[img_id]]).T[mask])
            else:
                coord_arrays.append(torch.stack([torch.arange(map_w, device=self.device), amax[img_id]]).T[mask])
        return coord_arrays

    def get_position(self, values_map, target_value, score_map, get_bbox=True):
        target_val_mask = (values_map == target_value).to(torch.int)
        if target_val_mask.sum() > 0:
            # in BxYX format
            scores = list()
            batch_coord_array = list()
            for batch_id, mask in enumerate(target_val_mask.cpu().numpy()):
                contours = cv2.findContours((mask * 255).astype('uint8'),
                                            cv2.RETR_LIST,
                                            cv2.CHAIN_APPROX_SIMPLE)[0]
                contours = [cont for cont in contours if len(cont) > 20]
                batch_coord_array.extend(contours)
                for contour in contours:
                    score_mask = cv2.drawContours(mask, [contour], contourIdx=-1, color=255, thickness=-1).astype(bool)
                    scores.append(score_map[batch_id][score_mask].mean().cpu().item())

            if not get_bbox:
                return torch.tensor(batch_coord_array), torch.tensor(scores)
            else:
                # extreme points
                return torch.tensor([self._get_bbox(coord_array) for coord_array in batch_coord_array
                                     if coord_array.shape[0] > 0]), torch.tensor(scores)
        else:
            return torch.tensor(list()), torch.tensor(list())

    def get_mapped_category_id(self, reorder_cls_id):
        return self._cat_maper[reorder_cls_id]

    def __call__(self, model_out, real_image_shape,
                 true_labels=None, ret_cls_pos=False,
                 get_bbox=True, method=0):
        if true_labels is not None:
            self.update_centriods(model_out, true_labels)
        centroids = {class_id: self._class_containers[class_id] / self._class_counters[class_id]
                     for class_id in self._cat_maper.keys() if self._class_counters[class_id] > 0}
        batch_size = model_out.shape[0]

        scores = dict()
        batch_classes_and_pos = dict()
        pix_vectors = model_out.permute(0, 2, 3, 1).reshape(-1, self.embedding_size)
        if method == 0:
            seg_map = torch.zeros(batch_size, *model_out.shape[-2:], device=self.device)
            for class_id, centroid in centroids.items():
                distances = torch.cdist(pix_vectors - centroid, p=2)
                mask = distances < self.delta
                seg_map[mask] = class_id

                if ret_cls_pos:
                    for batch_id in range(batch_size):
                        pos, pos_scores = self.get_position(seg_map, class_id, distances, get_bbox=get_bbox)
                        if len(pos) > 0:
                            batch_classes_and_pos[class_id] = pos
                            scores[class_id] = pos_scores

        else:
            # clusterizer = MeanShift(bandwidth=self.delta, seeds=torch.stack(list(centroids.values())).numpy(),
            #                         cluster_all=False)
            # clusters = model_out.cpu().numpy().transpose(0, 2, 3, 1).reshape(-1, self.embedding_size)
            # clusters = clusterizer.fit_predict(clusters)
            start = perf_counter()
            clusters, cls_scores = mean_shift(pix_vectors, torch.stack(list(centroids.values()), 0), n_steps=100,
                                              bandwidth=self.delta_v, matching_bandwidht=self.delta_v)
            end = perf_counter() - start
            print(f'elapsed time: {end}')
            for key_id, key in zip(list(range(len(centroids)))[::-1], list(centroids.keys())[::-1]):
                clusters[clusters == key_id] = key

            seg_map = clusters.reshape(batch_size, *model_out.shape[-2:])
            cls_scores = cls_scores.reshape(batch_size, *model_out.shape[-2:])
            for key in centroids.keys():
                pos, pos_scores = self.get_position(seg_map, key, cls_scores, get_bbox=get_bbox)
                if len(pos) > 0:
                    keep = box_ops.batched_nms(pos.to(torch.float32),
                                               pos_scores,
                                               torch.tensor([key]*pos.shape[0]),
                                               self.nms_thresh)
                    pos = pos[keep]
                    pos_scores = pos_scores[keep]
                    batch_classes_and_pos[key] = pos.numpy().tolist()
                    scores[key] = pos_scores.numpy().tolist()
            seg_map = torch.clamp(seg_map, min=0)
            seg_map = seg_map.to(self.device)

        if ret_cls_pos:
            return seg_map, batch_classes_and_pos, scores
        else:
            return seg_map, batch_classes_and_pos, scores
