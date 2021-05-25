import cv2
import numpy as np
import torch
from sklearn.cluster import MeanShift


class BottomUpPostprocessing:
    def __init__(self, classes_ids, delta, embedding_size, device):
        self.num_classes = len(classes_ids)
        self.classes_ids = classes_ids
        self._cat_maper = dict(zip(list(range(1, len(self.classes_ids) + 1)), self.classes_ids))
        self.delta = delta
        self.embedding_size = embedding_size
        self.device = device
        self._class_containers = {class_id: torch.zeros(embedding_size).to(device)
                                  for class_id in self._cat_maper.keys()}
        self._class_counters = dict.fromkeys(self._cat_maper.keys(), 0)

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
                batch_coord_array.extend(contours)
                for contour in contours:
                    sqore_mask = cv2.drawContours(mask, [contour], contourIdx=-1, color=255, thickness=-1).astype(bool)
                    scores.append(score_map[batch_id][sqore_mask].mean().cpu().item())

            if not get_bbox:
                return batch_coord_array, scores
            else:
                # extreme points
                return [self._get_bbox(coord_array) for coord_array in batch_coord_array
                        if coord_array.shape[0] > 0], scores
        else:
            return list(), list()

    def get_mapped_category_id(self, reorder_cls_id):
        return self._cat_maper[reorder_cls_id]

    def __call__(self, model_out, true_labels=None, ret_cls_pos=False, get_bbox=True, method=0):
        if true_labels is not None:
            self.update_centriods(model_out, true_labels)
        centroids = {class_id: self._class_containers[class_id] / self._class_counters[class_id]
                     for class_id in self._cat_maper.keys() if self._class_counters[class_id] > 0}
        batch_size = model_out.shape[0]

        scores = dict()
        batch_classes_and_pos = dict()
        if method == 0:
            seg_map = torch.zeros(batch_size, *model_out.shape[-2:], device=self.device)
            for class_id, centroid in centroids.items():
                distances = torch.norm(model_out - centroid.reshape(1, -1, 1, 1), p=2, dim=1)
                mask = distances < self.delta
                seg_map[mask] = class_id

                if ret_cls_pos:
                    for batch_id in range(batch_size):
                        pos, pos_scores = self.get_position(seg_map, class_id, distances, get_bbox=get_bbox)
                        if len(pos) > 0:
                            batch_classes_and_pos[class_id] = pos
                            scores[class_id] = pos_scores

        else:
            clusterizer = MeanShift(bandwidth=self.delta, seeds=torch.stack(list(centroids.values())).numpy(),
                                    cluster_all=False)
            clusters = model_out.cpu().numpy().transpose(0, 2, 3, 1).reshape(-1, self.embedding_size)
            clusters = clusterizer.fit_predict(clusters)
            for key_id, key in enumerate(centroids.keys()):
                clusters[clusters == key_id] = key
            seg_map = clusters.reshape(batch_size, *model_out.shape[-2:])
            seg_map = torch.from_numpy(seg_map)
            seg_map = torch.clamp(seg_map, min=0)

        if ret_cls_pos:
            return seg_map, batch_classes_and_pos, scores
        else:
            return seg_map
