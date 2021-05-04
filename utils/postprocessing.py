import torch


class BottomUpPostprocessing:
    def __init__(self, classes_ids,  delta, embedding_size):
        self.num_classes = len(classes_ids)
        self.classes_ids = classes_ids
        self.delta = delta
        self._class_containers = dict.fromkeys(classes_ids, torch.zeros(embedding_size))
        self._class_counters = dict.fromkeys(classes_ids)

    def update_centriods(self, net_output, true_labels):
        classes = torch.unique(true_labels)
        for class_id in classes:
            mask = true_labels == class_id
            self._class_containers[class_id] += torch.mean(net_output[mask], 0)
            self._class_counters[class_id] += 1

    def __call__(self, net_output, true_labels=None):
        if true_labels is not None:
            self.update_centriods(net_output, true_labels)
        centroids = {class_id: self._class_containers[class_id] / self._class_counters[class_id]
                     for class_id in self.classes_ids}
        seg_map = torch.zeros(net_output.shape)
        for class_id, centroid in centroids.items():
            seg_map[(net_output - centroid) < self.delta] = class_id

        return seg_map
