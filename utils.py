import torch
import random
import numpy as np
from scipy import ndimage
from torch.utils.data import Sampler
from sklearn.metrics import multilabel_confusion_matrix


def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.cuda.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = True

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
    f1_score[np.isnan(f1_score)] = 0
    
    return np.around(100 * acc, 2), np.round(100 * f1_score, 2)


class BalancedSampler(Sampler):  # 每次采样包含两个mini-batch，一个斑块类别平衡，一个狭窄程度平衡
    def __init__(self, type_list, stenosis_list, arr_columns=400, num_samples=50):

        type_unique, type_counts = np.unique(type_list, return_counts=True)
        stenosis_unique, stenosis_counts = np.unique(stenosis_list, return_counts=True)

        print('type label: {}, type label num: {}'.format(type_unique.tolist(), type_counts.tolist()))
        print('stenosis label: {}, stenosis label num: {}'.format(stenosis_unique.tolist(), stenosis_counts.tolist()))
        print('--' * 30)

        assert arr_columns > max(max(type_counts), max(stenosis_counts)), print('num of arr_columns should be greater than the largest counts')
        assert arr_columns % num_samples == 0 and num_samples <= arr_columns, print('arr_columns should be a multiple of num_samples and >= num_samples.')

        self.num_samples = num_samples  # 采样次数
        self.arr_columns = arr_columns  # 得到类别平衡矩阵的列数
        self.type_arr = np.zeros((len(type_unique), self.arr_columns), dtype=np.int)
        self.stenosis_arr = np.zeros((len(stenosis_unique), self.arr_columns), dtype=np.int)

        for i in type_unique:
            self.type_arr[i] = np.tile(np.where(np.array(type_list) == i)[0], 20)[:self.arr_columns]  # 找到对应label在列表中的位置，并复制到最大频率，放入数组
        for i in stenosis_unique:
            self.stenosis_arr[i] = np.tile(np.where(np.array(stenosis_list) == i)[0], 3)[:self.arr_columns]

    def __iter__(self):
        type_arr = self._shuffle_along_axis(self.type_arr, axis=1)
        stenosis_arr = self._shuffle_along_axis(self.stenosis_arr, axis=1)

        type_col_list = [item.reshape(-1).tolist() for item in np.hsplit(type_arr, self.num_samples)]  # 把整个arr分成num_samples份拉直放进list，list中的元素是类别平衡的
        stenosis_col_list = [item.reshape(-1).tolist() for item in np.hsplit(stenosis_arr, self.num_samples)]

        # batch_size = (arr_columns / num_samples) * 7
        batch = []
        for i in range(self.num_samples):
            batch.extend(type_col_list[i])
            batch.extend(stenosis_col_list[i])
            random.shuffle(batch)
            yield batch
            batch = []

    def __len__(self):
        return self.num_samples  # 有num_samples次iter才会遍历完整个训练集

    def _shuffle_along_axis(self, a, axis):  # 在指定轴上打乱numpy数组
        idx = np.random.rand(*a.shape).argsort(axis=axis)
        return np.take_along_axis(a,idx,axis=axis)


def merge_plaque(input_list):
    zero_list = []
    result_list = []

    for item in input_list:
        if item != 0:
            if len(zero_list) <= 2:
                for i in range(len(zero_list)):
                    result_list.append(item)
                zero_list.clear()
            result_list.append(item)
        if item == 0:
            zero_list.append(item)
            if len(zero_list) >2:
                result_list.extend(zero_list)

    return result_list


def get_branch_stenosis(branch_label):
    """
    从一条branch_label中得到对应的stenosis程度
    """
    branch_stenosis = []
    for seg in branch_label:
        branch_stenosis.extend(seg[2])
    return max(branch_stenosis)