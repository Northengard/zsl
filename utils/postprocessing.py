import torch


class BottomUpPostprocessing:
    def __init__(self, classes_ids,  delta, embedding_size):
        self.num_classes = len(classes_ids)
        self.classes_ids = classes_ids
        self.delta = delta
        self._class_containers = {class_id: torch.zeros(embedding_size) for class_id in classes_ids}
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

    def __call__(self, net_output, true_labels=None):
        if true_labels is not None:
            self.update_centriods(net_output, true_labels)
        centroids = {class_id: self._class_containers[class_id] / self._class_counters[class_id]
                     for class_id in self.classes_ids if self._class_counters[class_id] > 0}
        batch_size = net_output.shape[0]
        seg_map = torch.zeros(batch_size, *net_output.shape[-2:])
        for class_id, centroid in centroids.items():
            distances = torch.norm(net_output - centroid.reshape(1, -1, 1, 1), p=2, dim=1)
            mask = distances < self.delta
            seg_map[mask] = class_id

        return seg_map
