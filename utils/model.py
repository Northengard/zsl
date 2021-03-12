from numpy import median, std


def get_learning_rate(optimizer):
    """
    returns current learning from optimizer
    :param optimizer: class from torch.optim
    :return: float, learning rate
    """
    for param_group in optimizer.param_groups:
        return param_group['lr']


class AverageMeter(object):
    def __init__(self, is_validation=False):
        """
        class to save some statistics
        :param is_validation: bool, set true to store values for each validation iteration
        """
        self.vals = []
        self.val = 0
        self.avg = 0
        self.median = 0
        self.sum = 0
        self.count = 0
        self.std = 0
        self.is_validation = is_validation

    def reset(self):
        """
        reset all values and statistics
        :return: None
        """
        self.vals = []
        self.val = 0
        self.avg = 0
        self.median = 0
        self.sum = 0
        self.count = 0
        self.std = 0

    def update(self, val):
        """
        update stats with new value
        :param val: digit value
        :return: None
        """
        if self.is_validation:
            self.vals.append(val)
            self.median = median(self.vals)
            self.std = std(self.vals)
        self.val = val
        self.sum += val
        self.count += 1
        self.avg = self.sum / self.count
