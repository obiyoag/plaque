import torch
import random
import numpy as np
from scipy import ndimage
from sklearn.metrics import multilabel_confusion_matrix


def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.cuda.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = True


def random_crop(image, crop_size, move, prob):
    channel, depth, height, width = image.shape
    if random.uniform(0, 1) < prob:
        x_move = np.random.randint(-move, move)
        y_move = np.random.randint(-move, move)
    else:
        x_move = 0
        y_move = 0
    x_center = width//2 + x_move
    y_center = height//2 + y_move
    cropped_image = image[:, :, y_center - crop_size//2: y_center + crop_size//2, x_center - crop_size//2: x_center + crop_size//2]
    return cropped_image

def random_rotate(image, prob):
    if random.uniform(0, 1) < prob:
        angle = np.random.randint(-180, 180)
        image = ndimage.rotate(image, angle, axes=(2,3), reshape=False)  # axes确定旋转平面
    return image

def add_gaussian_noise(image, prob, mean=0, var=0.0005):
    if random.uniform(0, 1) < prob:
        noise = np.random.normal(mean, var ** 0.5, image.shape)
    else:
        noise = 0.0
    out = np.clip(image + noise, 0, 1)
    return out


class Data_Augmenter(object):
    def __init__(self, crop_size=50, move=2, prob=0.5):
        self.crop_size = crop_size
        self.move = move
        self.prob = prob

    def __call__(self, image):
        image = random_rotate(image, self.prob)
        image = random_crop(image, self.crop_size, self.move, self.prob)
        image = add_gaussian_noise(image, self.prob)
        return image


class Center_Crop(object):
    def __init__(self, crop_size=50):
        self.crop_size = crop_size

    def __call__(self, image):
        image = random_crop(image, self.crop_size, 0, 0)
        return image


def seg_digitize(type_seg):
    """
    将一段seg_label转化为一个值
    """
    result = 0
    unique, counts = np.unique(type_seg, return_counts=True)
    if len(unique) == 1:  # seg_label为同一个值
        result = unique.item()
    else:  # seg_label不为同一个值，结果为长度大于2的最长非零值的值
        if max(counts[1:]) > 2:
            result = unique[np.argmax(counts[1:]) + 1].item()
    return result


def get_metrics(y_true, y_pred):
    mcm = multilabel_confusion_matrix(y_true, y_pred)

    tp = mcm[:, 1, 1]
    tn = mcm[:, 0, 0]
    fn = mcm[:, 1, 0]
    fp = mcm[:, 0, 1]

    np.seterr(divide='ignore',invalid='ignore')
    recall = tp / (tp + fn)
    recall[np.isnan(recall)] = 1
    precision = tp / (tp + fp)
    precision[np.isnan(precision)] = 1
    acc = (tp + tn) / (tp + tn + fp + fn)
    f1_score = 2 * recall * precision / (recall + precision)
    return acc, f1_score