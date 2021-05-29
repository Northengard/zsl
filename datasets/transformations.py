import numpy as np
import cv2
import torch

from torchvision import transforms
from skimage.draw import polygon
from utils.visualisation import alpha_blend
from PIL import Image


class HorizontalFlip:
    """
    Horizontal Flip of sample (Mirroring)
    """

    def __init__(self):
        self.rot_axis = 1

    @staticmethod
    def flip_coords(coords, img_w):
        coords[:, 2], coords[:, 0] = img_w - coords[:, 0], img_w - coords[:, 2]
        return coords

    def __call__(self, sample):
        img_w = sample['image'].shape[1]
        for key, item in sample.items():
            if key == 'bbox':
                sample[key] = self.flip_coords(sample[key], img_w)
            elif key == 'category_id':
                pass
            else:
                sample[key] = cv2.flip(item, self.rot_axis)
        return sample


# (125. , 123 , 114)
# (0.4914, 0.4822, 0.4465)
class RandomErasing:
    def __init__(self, probability=0.5, sl=0.02, sh=0.08, r1=0.4, mean=(125., 123, 114)):
        """
        Class that performs Random Erasing in Random Erasing Data Augmentation by Zhong et al.
        -------------------------------------------------------------------------------------
        probability: The applying probability that the operation will be performed.
        sl: min erasing area
        sh: max erasing area
        r1: min aspect ratio
        mean: erasing value
        -------------------------------------------------------------------------------------
        """
        self.probability = probability
        self.mean = mean
        self.sl = sl
        self.sh = sh
        self.r1 = r1

    def __call__(self, sample):
        img = sample['image']

        if np.random.uniform(0, 1) > self.probability:
            return sample

        for attempt in range(100):
            src_h, src_w, num_chn = img.shape
            area = src_h * src_w

            target_area = np.random.uniform(self.sl, self.sh) * area
            aspect_ratio = np.random.uniform(self.r1, 1 / self.r1)

            h = int(round(np.sqrt(target_area * aspect_ratio)))
            w = int(round(np.sqrt(target_area / aspect_ratio)))

            if w < src_w and h < src_h:
                x1 = np.random.randint(0, src_h - h)
                y1 = np.random.randint(0, src_w - w)
                if num_chn == 3:
                    img[x1:x1 + h, y1:y1 + w, 0] = self.mean[0]
                    img[x1:x1 + h, y1:y1 + w, 1] = self.mean[1]
                    img[x1:x1 + h, y1:y1 + w, 2] = self.mean[2]
                else:
                    img[x1:x1 + h, y1:y1 + w, 0] = self.mean[0]
                sample['image'] = img
                return sample
        return sample


class RandomShadowFields:
    def __init__(self, sl=0.02, sh=0.4, r1=0.3):
        """
        sl: min erasing area
        sh: max erasing area
        r1: min aspect ratio
        """
        self.sl = sl
        self.sh = sh
        self.r1 = r1

    def __call__(self, sample):
        img = sample['image']

        for attempt in range(100):
            src_h, src_w, num_chn = img.shape
            area = src_h * src_w

            target_area = np.random.uniform(self.sl, self.sh) * area
            aspect_ratio = np.random.uniform(self.r1, 1 / self.r1)

            h = int(round(np.sqrt(target_area * aspect_ratio)))
            w = int(round(np.sqrt(target_area / aspect_ratio)))

            if w < src_w and h < src_h:
                x1 = np.random.randint(0, src_h - h)
                y1 = np.random.randint(0, src_w - w)
                num_points = np.random.randint(low=5, high=50)
                points = list([np.random.randint(low=0, high=h, size=num_points) + x1,
                               np.random.randint(low=0, high=w, size=num_points) + y1])
                blank_img = np.ones(img.shape)
                blank_img[polygon(points[0], points[1])] = np.random.uniform(0.1, 0.8)
                mask = ((1 - blank_img) * 255).astype('uint8')
                blank_img = (blank_img * 255).astype('uint8')
                img = alpha_blend(img, blank_img, mask)
                sample['image'] = img
                return sample
        return sample


# COLOR, ILLUMINATION, BLUR
class ColorTransform:
    """
    Color transform to deal with image brightness, contrast, saturation and hue.
    Wraper over torchvision.transforms.ColorJitter
    """

    def __init__(self, brightness=0, contrast=0, saturation=0, hue=0):
        """
        Color transform to deal with image brightness, contrast, saturation and hue. \n
        Wraper over torchvision.transforms.ColorJitter.
        :param brightness: (tuple of float), set range of positive values for brightness changes
        :param contrast:(tuple of float), set range of positive values for contrast changes
        :param saturation: (tuple of float), set range of positive values for saturation changes
        :param hue: (tuple of float), set range of hue changes. Values must be between (-0.5, 0.5)
        """

        def _check_param(param_name, param, min_val, max_val):
            if param != 0:
                try:
                    assert type(param) == tuple
                    assert len(param) == 2
                    assert min(param) >= min_val
                    assert max(param) <= max_val
                    assert param[0] <= param[1]
                    return True
                except AssertionError(param_name +
                                      ' must be a tuple of float (min>={}, max<={})'.format(min_val, max_val)):
                    return False
            else:
                return True

        if _check_param('brightness', brightness, 0, np.inf):
            self.brightness = brightness
        if _check_param('contrast', contrast, 0, np.inf):
            self.contrast = contrast
        if _check_param('saturation', saturation, 0, np.inf):
            self.saturation = saturation
        if _check_param('hue', hue, -0.5, 0.5):
            self.hue = hue
        self.cj = transforms.ColorJitter(brightness=self.brightness,
                                         contrast=self.contrast,
                                         saturation=self.saturation,
                                         hue=self.hue)

    def __call__(self, sample):
        image = sample['image']
        pil_img = Image.fromarray(image)
        pil_img = self.cj(pil_img)
        sample['image'] = np.array(pil_img)
        return sample


# CHANGE SIZE
def get_pad(nn_h, nn_w, img_h, img_w):
    nn_asp_ratio = nn_h / nn_w
    img_ratio = img_h / img_w
    if img_ratio > nn_asp_ratio:
        pad_w = int(img_h / nn_asp_ratio - img_w)
        pad_h = 0
    elif img_ratio < nn_asp_ratio:
        pad_w = 0
        pad_h = int(img_w * nn_asp_ratio - img_h)
    else:
        pad_h = 0
        pad_w = 0
    return 0, pad_h, 0, pad_w


class Rescale:
    """
    Rescale the image in a sample to a given size.
    """

    def __init__(self, output_size, interpolation=3):
        """
        Init parameters:
        :param output_size: (tuple or int): Desired output size. If tuple, output is
                                            matched to output_size. If int, smaller of image edges is matched
                                            to output_size keeping aspect ratio the same.
        :param interpolation: (int): interpolation method. 0 - nearest, 1 - linear, 2 - cubic, 3 inter area, etc.
        See cv2 interpolation methods.
        """
        if type(output_size) == int:
            self.output_size = (output_size, output_size)
        else:
            self.output_size = output_size
        self.output_size = tuple(self.output_size)
        self.resize_coef = 1.
        self.interpolation = interpolation

    def resize_coords(self, coords):
        return coords * self.resize_coef

    def pad_and_resize(self, img):
        src_h, src_w = img.shape[:2]
        nn_w, nn_h = self.output_size
        if (src_h != nn_h) or (nn_w != src_w):
            _, pad_h, _, pad_w = get_pad(nn_h, nn_w, src_h, src_w)
            img = cv2.copyMakeBorder(img, 0, pad_h, 0, pad_w, cv2.BORDER_CONSTANT, value=0)
            src_h, src_w = img.shape[:2]
            self.resize_coef = nn_h / src_h
            img = cv2.resize(img, self.output_size, interpolation=self.interpolation)
        return img

    def __call__(self, sample):
        for key, item in sample.items():
            if key in ['bbox', 'category_id']:
                continue
            sample[key] = self.pad_and_resize(item)
        if 'bbox' in sample.keys():
            sample['bbox'] = self.resize_coords(sample['bbox'])
        return sample


class GaussianBlur:
    """
    Gaussian filter with different kernel size.
    """
    def __init__(self, max_blur_kernel=8, min_blur_kernel=5):
        """
        Init parameters:
        :param max_blur_kernel: (int), max size of kernel
        :param min_blur_kernel: (int), min size of kernel
        """
        self.kernels = list(range(min_blur_kernel, max_blur_kernel, 2))

    def __call__(self, sample):
        image = sample['image']
        # kernel size
        k_size = np.random.choice(self.kernels)
        # 0 means that sigmaX and sigmaY calculates from kernel
        image = cv2.GaussianBlur(image, (k_size, k_size), 0)
        sample['image'] = image
        return sample


class ToTensor:
    """
    Convert ndarrays in sample to Tensors.
    """
    def __init__(self, normalize=False):
        self.to_tensor = transforms.ToTensor()
        self.normalize = transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
        self.do_normalize = normalize

    def __call__(self, sample):
        for key, item in sample.items():
            if key == 'image':
                sample[key] = self.to_tensor(item)
                if self.do_normalize:
                    sample[key] = self.normalize(sample[key])
            else:
                if len(sample[key].shape) > 2:
                    sample[key] = sample[key].transpose(2, 0, 1)
                sample[key] = torch.Tensor(sample[key])
        return sample


class RandomApply:
    def __init__(self, transformations, prob=0.5):
        self.transforms = transformations
        self.prob = prob

    def __call__(self, sample):
        for transf in self.transforms:
            app_prob = np.random.rand()
            if app_prob < self.prob:
                sample = transf(sample)
        return sample


# GENERAL
class Transforms:
    def __init__(self, conf, is_train=True):
        self.is_train = is_train
        color_transform_params = conf.TRANSFORMATIONS.PARAMS.COLOR
        if is_train:
            self.train_transform = RandomApply([
                # RandomShadowFields(),
                ColorTransform(brightness=color_transform_params.BRIGHTNESS,
                               contrast=color_transform_params.CONTRAST,
                               saturation=color_transform_params.SATURATION,
                               hue=color_transform_params.HUE),
                HorizontalFlip(),
                RandomErasing(sh=conf.TRANSFORMATIONS.PARAMS.RANDOM_ERAZING.MAX_AREA,
                              r1=conf.TRANSFORMATIONS.PARAMS.RANDOM_ERAZING.MIN_RATIO,
                              probability=conf.TRANSFORMATIONS.PARAMS.RANDOM_ERAZING.FLAG),
                GaussianBlur(),
            ])
        self.normalize = Normalize(conf, normalize=conf.TRANSFORMATIONS.PARAMS.NORMALIZE)

    def __call__(self, sample):
        if self.is_train:
            sample = self.train_transform(sample)
        sample = self.normalize(sample)

        return sample


class Normalize:
    def __init__(self, cfg, normalize=False):
        self.input_size = cfg.DATASET.PARAMS.IMAGE_SIZE
        trainsormations_list = [Rescale(tuple(self.input_size), cfg.TRANSFORMATIONS.PARAMS.INTERPOLATION),
                                ToTensor(normalize=normalize)]
        self.normalize = transforms.Compose(trainsormations_list)

    def __call__(self, sample):
        sample = self.normalize(sample)
        return sample
