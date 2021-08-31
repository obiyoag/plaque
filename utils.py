import torch
import random
import numpy as np
from scipy import ndimage
from torch.utils.data import Sampler
from sklearn.metrics import multilabel_confusion_matrix


if torch.__version__ != 'parrots':
    from torch.nn.functional import pad
else:
    from torch.nn import ReflectionPad1d

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
            self.type_arr[i] = np.tile(np.where(np.array(type_list) == i)[0], 100)[:self.arr_columns]  # 找到对应label在列表中的位置，并复制到最大频率，放入数组
        for i in stenosis_unique:
            self.stenosis_arr[i] = np.tile(np.where(np.array(stenosis_list) == i)[0], 100)[:self.arr_columns]

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
            else:
                result_list.extend(zero_list)
            zero_list.clear()
            result_list.append(item)
        else:
            zero_list.append(item)

    if len(zero_list) != 0:
        result_list.extend(zero_list)

    assert len(result_list) == len(input_list), print("plaque merge error")
    return result_list


def digitize_stenosis(stenosis):
    """
    按照frame的狭窄程度，返回一个狭窄程度list
    0-20没有狭窄，20-50轻度，50-100重度
    """
    result = []
    for frame_stenosis in stenosis:
        if frame_stenosis < 20:
            result.append(0)
        elif 20<= frame_stenosis <50:
            result.append(1)
        elif frame_stenosis >= 50:
            result.append(2)
    return max(result)

def get_branch_stenosis(branch_label):
    """
    从一条branch_label中得到对应的stenosis程度
    """
    branch_stenosis = []
    for seg in branch_label:
        branch_stenosis.extend(seg[2])
    return digitize_stenosis(branch_stenosis)

def deduplicate_sort_seg(seg):
    """
    消除label中的重复帧并排序
    """
    seg_list = list()
    seg_list.append(seg[0] - 1)

    index, stenosis = seg[1], seg[2]
    sort_list = list()
    for i in range(len(index)):
        sort_list.append((index[i], stenosis[i]))
    sort_list = sorted(list(set(sort_list)))

    index = list()
    stenosis =list()
    for item in sort_list:
        index.append(item[0])
        stenosis.append(item[1])
    
    seg_list.append(index)
    seg_list.append(stenosis)
    return seg_list


def get_bounds(label, seg_len):
    """
    1. 通过label得到应该从图像中截取的boundary(考虑斑块有重叠的情况)
    2. 取一段seg中frames最大的stenosis作为整个seg的stenosis
    """
    index_stenosis_list = []
    max_pos_list = []
    seg_boundary_list = []
    for seg in label:
        index_stenosis_list.append(dict(zip(seg[1], seg[2])))
    for index_stenosis_dict in index_stenosis_list:
        max_pos = max(index_stenosis_dict, key=index_stenosis_dict.get)
        max_pos_list.append(max_pos)
    
    seg_boundary_list.append(max_pos_list[0] - seg_len//2)  # 第一个seg左边界按理想取，长度不够后面再说
    
    for idx in range(len(max_pos_list) - 1):
        first_max = max_pos_list[idx]
        second_max = max_pos_list[idx + 1]
        max_gap = second_max - first_max
        if max_gap >= seg_len:  # 斑块之间没有重叠
            seg_boundary_list.append(first_max + seg_len//2)
            seg_boundary_list.append(second_max - seg_len//2)
        else:  # 斑块之间有重叠
            seg_boundary_list.append((first_max + second_max) // 2)
            seg_boundary_list.append((first_max + second_max) // 2)
    
    seg_boundary_list.append(max_pos_list[-1] + seg_len//2)  # 最后一个seg右边界按理想取，长度不够后面再说

    assert len(seg_boundary_list) % 2 == 0, print('Boundary Error!')

    bound_list = []
    for left_bound, center_pos, right_bounnd in zip(seg_boundary_list[::2], max_pos_list, seg_boundary_list[1::2]):
        bound_list.append([left_bound, center_pos, right_bounnd])
    
    new_label = []
    for seg, bounds in zip(label, bound_list):
        new_label.append([seg[0], digitize_stenosis(seg[2]), bounds])
    
    return new_label

def process_label(label, seg_len):
    label_list = list()
    for seg in label:
        label_list.append(deduplicate_sort_seg(seg))
    label_list = sorted(label_list, key=lambda x: x[1])  # 按照index从小到大排列seg
    label_list = get_bounds(label_list, seg_len)
    return label_list

def reflect_pad(image, left_pad, right_pad):
    if torch.__version__ != 'parrots':
        image = pad(image.transpose(2, 0), (left_pad, right_pad), mode="reflect").transpose(2, 0)
    else:
        func_pad = ReflectionPad1d((left_pad, right_pad))
        image = func_pad(image.transpose(2, 0)).transpose(2, 0)
    return image