import cv2
import numpy as np
import torch
from sklearn.cluster import MeanShift


class BottomUpPostprocessing:
    def __init__(self, classes_ids, delta, embedding_size, device):
        self.num_classes = len(classes_ids)
        self.classes_ids = classes_ids
        self.delta = delta
        self.embedding_size = embedding_size
        self.device = device
        self._class_containers = {class_id: torch.zeros(embedding_size).to(device) for class_id in classes_ids}
        self._class_counters = dict.fromkeys(classes_ids, 0)

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
        top = coord_array[coord_array[:, 0].argmin()]
        left = coord_array[coord_array[:, 1].argmin()]

        bottom = coord_array[coord_array[:, 0].argmax()]
        right = coord_array[coord_array[:, 1].argmax()]
        # bbox
        return torch.stack([top[0], left[1], bottom[0], right[1]])

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
                    scores.append(score_map[batch_id][contour].mean().cpu().item)
            # top_border_coords = self._get_top_border_coords(target_val_mask)
            # bottom_border_coords = self._get_top_border_coords(target_val_mask.flip(dims=(1,)), is_reversed=True)
            # batch_coord_array = [torch.cat((top_border_coords[in_batch_id], bottom_border_coords[in_batch_id]))
            #                      for in_batch_id in range(len(top_border_coords))]

            if not get_bbox:
                return batch_coord_array, scores
            else:
                # extreme points
                return [self._get_bbox(coord_array) for coord_array in batch_coord_array
                        if coord_array.shape[0] > 0], scores
        else:
            return list(), list()

    def __call__(self, semantic_out, instance_out=None, true_labels=None, with_bbox=False, method=0):
        if true_labels is not None:
            self.update_centriods(semantic_out, true_labels)
        centroids = {class_id: self._class_containers[class_id] / self._class_counters[class_id]
                     for class_id in self.classes_ids if self._class_counters[class_id] > 0}
        batch_size = semantic_out.shape[0]

        scores = dict()
        batch_classes_and_pos = dict()
        if method == 0:
            seg_map = torch.zeros(batch_size, *semantic_out.shape[-2:], device=self.device)
            for class_id, centroid in centroids.items():
                distances = torch.norm(semantic_out - centroid.reshape(1, -1, 1, 1), p=2, dim=1)
                mask = distances < self.delta
                seg_map[mask] = class_id

                # if with_bbox:
                #     # instance
                # if instance_out is not None:
                #     for batch_id in range(batch_size):
                #         instance_map = torch.zeros(1, *semantic_out.shape[-2:], device=semantic_out.device)
                #         # TOOOO LOOONG
                #         clusterizer = MeanShift(bandwidth=self.delta, cluster_all=True, n_jobs=8)
                #         instance_clusters = instance_out[batch_id].permute(1, 2, 0)[mask[batch_id]].cpu().numpy()
                #         instance_clusters = clusterizer.fit_predict(instance_clusters)
                #         instance_clusters += 1
                #         num_instances = np.unique(instance_clusters)
                #         instance_map[mask] = torch.from_numpy(instance_clusters).to(self.device).to(torch.float32)
                #         scores = list()
                #         class_bboxes = list()
                #         for instance_id in num_instances:
                #             positions, pos_scores = self.get_position(instance_map, instance_id, distances,
                #                                                       get_bbox=True)
                #             scores.append(pos_scores)
                #             class_bboxes.append(positions)
                #         batch_classes_and_pos[batch_id][class_id] = class_bboxes
                # else:
                for batch_id in range(batch_size):
                    pos, pos_scores = self.get_position(seg_map, class_id, distances, get_bbox=with_bbox)
                    if len(pos) > 0:
                        batch_classes_and_pos[class_id] = pos
                        scores[class_id] = pos_scores

        else:
            clusterizer = MeanShift(bandwidth=self.delta, seeds=torch.stack(list(centroids.values())).numpy(),
                                    cluster_all=False)
            clusters = semantic_out.cpu().numpy().transpose(0, 2, 3, 1).reshape(-1, self.embedding_size)
            clusters = clusterizer.fit_predict(clusters)
            for key_id, key in enumerate(centroids.keys()):
                clusters[clusters == key_id] = key
            seg_map = clusters.reshape(batch_size, *semantic_out.shape[-2:])
            seg_map = torch.from_numpy(seg_map)
            seg_map = torch.clamp(seg_map, min=0)

        return seg_map, batch_classes_and_pos, scores
