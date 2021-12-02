import random
import numpy as np
from scipy import ndimage

from utils.mae_utils import RandomMaskingGenerator


def normalize(image):
    max, min = image.max(), image.min()
    out = (image - min) / (max - min)
    return out

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
        image = ndimage.rotate(image, angle, axes=(-2,-1), reshape=False)  # axes确定旋转平面
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
        image = normalize(image)
        image = random_rotate(image, self.prob)
        image = random_crop(image, self.crop_size, self.move, self.prob)
        image = add_gaussian_noise(image, self.prob)
        return image


class Center_Crop(object):
    def __init__(self, crop_size=50):
        self.crop_size = crop_size

    def __call__(self, image):
        image = normalize(image)
        image = random_crop(image, self.crop_size, 0, 0)
        return image


class MAE_Augmenter(object):
    def __init__(self, seg_len, mask_ratio, crop_size=50, move=2, prob=0.5):
        self.crop_size = crop_size
        self.move = move
        self.prob = prob

        self.masked_position_generator = RandomMaskingGenerator(seg_len, mask_ratio)

    def __call__(self, image):
        image = normalize(image)
        image = random_rotate(image, self.prob)
        image = random_crop(image, self.crop_size, self.move, self.prob)
        image = add_gaussian_noise(image, self.prob)
        return image, self.masked_position_generator()
