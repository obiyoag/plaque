import os
import json
import random
import numpy as np
from math import ceil
import SimpleITK as sitk
import matplotlib.pyplot as plt
from torchvision import transforms
from utils import set_seed, Data_Augmenter
from torch.utils.data import Dataset, Sampler, DataLoader


def split_dataset(case_list):
    random.shuffle(case_list)
    case_num = len(case_list)
    train_num = int(case_num * 0.4)
    val_num = int(case_num * 0.1)
    train_paths = case_list[:train_num]
    val_paths = case_list[train_num: train_num + val_num]
    test_paths = case_list[train_num + val_num:]

    print('case num in train_paths: {}'.format(len(train_paths)))
    print('case num in val_paths: {}'.format(len(val_paths)))
    print('case num in test_paths: {}'.format(len(test_paths)))
    print('--' * 30)

    return train_paths, val_paths, test_paths

def process_label(label):  # 消除label中的重复帧，并排序
    """
    消除label中的重复帧并排序
    """
    def process_one_seg(seg):
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

    label_list = list()
    for seg in label:
        label_list.append(process_one_seg(seg))
    return label_list

def sample_normal(branch_path, stage='train', avg=15.84, std=11.53, min_len=2, max_len=67, ):  # 这里的统计值是训练集标注的segment的
    """
    从正常branch中采样出一段正常的segment
    """
    mpr_path = os.path.join(branch_path, 'mpr.nii.gz')
    mpr_itk = sitk.ReadImage(mpr_path)
    mpr_vol = sitk.GetArrayFromImage(mpr_itk)
    img_len = len(mpr_vol)

    if stage == "train":
        max_len = min(max_len, img_len)  # segment的最大值取mask_len和max_len中小的一个
        sample_len = int(np.clip(np.random.normal(avg, std), min_len, max_len))  # 用正态分布采样得到segment长度

        end_idx = (img_len - sample_len) if (img_len - sample_len != 0) else 1
        loc = int(np.random.choice(range(0, end_idx), 1))  # 确定segment起始位置
        sampled_idx = list(range(loc, loc + sample_len))

        label_list = [[0, sampled_idx, [0 for _ in range(sample_len)]]]

    elif stage == 'eval':
        label_list = [[0, list(range(img_len)), [0 for _ in range(img_len)]]]

    return label_list

def get_path2label_dict(case_list, failed_branch_list, stage='train'):
    """
    得到一个地址到label的映射字典，key为地址，value为label
    """
    path2label_dict = {}
    normal_branch_num = 0
    abnormal_branch_num = 0
    sample_normal_prob = 0 if stage == 'eval' else 0.7  # 训练只采样一部分正常branch，验证测试采样所有
    
    for case_path in case_list:
        branch_list = os.listdir(case_path)
        for branch_id in branch_list:
            branch_path = os.path.join(case_path, str(branch_id))

            if branch_path in failed_branch_list:  # 排除没有通过检测的branch
                continue

            json_path = os.path.join(branch_path, 'plaque.json')

            try:
                with open(json_path, 'r') as f:
                    dict = json.load(f)
                    if len(dict['plaques']) != 0:  # 如果相应branch不是正常的，则得到segmentlabel
                        path2label_dict[branch_path] = process_label(dict['plaques'])
                        abnormal_branch_num += 1
                    else:  # 如果相应branch是正常的，则有概率在branch中sample一段正常的segment
                        if random.uniform(0, 1) > sample_normal_prob:
                            path2label_dict[branch_path] = sample_normal(branch_path, stage)
                            normal_branch_num += 1
            except IOError:
                print("plaque json file not found.")

    print('normal branch num: {}'.format(normal_branch_num))
    print('abnormal branch num: {}'.format(abnormal_branch_num))
    print('total branch num: {}'.format(normal_branch_num + abnormal_branch_num))
    print('--' * 30)
    return path2label_dict, normal_branch_num


class Train_Dataset(Dataset):
    def __init__(self, paths, transform, pad_len=70):

        self.transform = transform
        self.pad_len = pad_len

        try:
            with open('failed_branches.json', 'r') as f:
                json_dict = json.load(f)
                failed_branch_list = json_dict['failed_branches']  # 没有通过检测的branch
                print('failed branch num in the dataset: {}'.format(len(failed_branch_list)))
        except IOError:
            print('failed_branches.json not found.')
        
        path2label_dict, normal_branch_num = get_path2label_dict(paths, failed_branch_list, stage='train')
    
        self.path_list = []
        self.type_list = []
        self.index_list = []
        self.stenosis_list = []

        for branch_path in list(path2label_dict.keys()):
            label = path2label_dict[branch_path]
            for seg in label:
                self.path_list.append(branch_path)
                self.type_list.append(seg[0])
                self.index_list.append(seg[1])
                self.stenosis_list.append(self._digitize_stenosis(seg[2]))

        print('normal seg num: {}'.format(normal_branch_num))
        print('abnormal seg num: {}'.format(len(self.path_list) - normal_branch_num))
        print('total seg num: {}'.format(len(self.path_list)))
        print('--' * 30)

    def __len__(self):
        return len(self.path_list)

    def __getitem__(self, idx):
        path, plaque_type, plaque_idx, stenosis = self.path_list[idx], self.type_list[idx], self.index_list[idx], self.stenosis_list[idx]
        
        mpr_path = os.path.join(path, 'mpr.nii.gz')
        mpr_itk = sitk.ReadImage(mpr_path)
        image = sitk.GetArrayFromImage(mpr_itk)
        image = image[plaque_idx[0]: plaque_idx[-1] + 1, :, :]  # 标注中最后一帧也有斑块
        image = self._pad_img(image)
        image = self.transform(image)
    
        return image, plaque_type, stenosis
        
    def _pad_img(self, image):
        """
        给segment补零到70帧
        """
        pad_img = np.zeros((self.pad_len, image.shape[1], image.shape[2]))
        pad_img[0: image.shape[0], :, :] = image
        pad_img = np.expand_dims(pad_img, axis=0)
        return pad_img

    def _digitize_stenosis(self, stenosis):
        """
        按照segment中最大的狭窄程度，返回一个狭窄程度label
        """
        result = 0
        max_stenosis = max(stenosis)
        if 0 < max_stenosis < 50:
            result = 1
        elif max_stenosis >= 50:
            result = 2
        return result


class Eval_Dataset(Dataset):
    def __init__(self, paths, pred_unit=45):
        self.pred_unit = pred_unit

        try:
            with open('failed_branches.json', 'r') as f:
                json_dict = json.load(f)
                failed_branch_list = json_dict['failed_branches']  # 没有通过检测的branch
        except IOError:
            print('failed_branches.json not found.')
        
        self.path2label_dict, normal_branch_num = get_path2label_dict(paths, failed_branch_list, stage='eval')
    
        self.path_list = list(self.path2label_dict.keys())
        self.label_list = list(self.path2label_dict.values())

        print('normal branch num: {}'.format(normal_branch_num))
        print('abnormal branch num: {}'.format(len(self.path_list) - normal_branch_num))
        print('total branch num: {}'.format(len(self.path_list)))
        print('--' * 30)

    def __len__(self):
        return len(self.path_list)

    def __getitem__(self, idx):
        path, label = self.path_list[idx], self.label_list[idx]
        
        mpr_path = os.path.join(path, 'mpr.nii.gz')
        mpr_itk = sitk.ReadImage(mpr_path)
        image = sitk.GetArrayFromImage(mpr_itk)
        length = image.shape[0]
        pad_len = ceil(length / self.pred_unit) * self.pred_unit

        image, plaque_type, stenosis = self._pad_img_label(image, label, pad_len)
    
        return image, plaque_type, stenosis
        
    def _digitize_stenosis(self, stenosis):
        """
        将每一帧的狭窄程度转变为离散化label
        """
        result = []
        for item in stenosis:
            if item == 0:
                result.append(0)
            elif 0 < item < 50:
                result.append(1)
            elif item >= 50:
                result.append(2)
        return result

    def _pad_img_label(self, image, label, pad_len):
        """
        将image和label填充到45的整数倍
        """
        pad_img = np.zeros((pad_len, image.shape[1], image.shape[2]))
        pad_img[0: image.shape[0], :, :] = image
        pad_img = np.expand_dims(pad_img, axis=0)

        pad_type = np.zeros((pad_len,), dtype=np.int)
        pad_stenosis = np.zeros((pad_len,), dtype=np.int)
        for seg in label:
            plaque_type = seg[0]
            plaque_idx = seg[1]
            stenosis = self._digitize_stenosis(seg[2])
            pad_type[plaque_idx[0]: plaque_idx[-1] + 1] = plaque_type
            pad_stenosis[plaque_idx[0]: plaque_idx[-1] + 1] = stenosis

        return pad_img, pad_type, pad_stenosis


class BalancedSampler(Sampler):  # 每次采样包含两个mini-batch，一个斑块类别平衡，一个狭窄程度平衡
    def __init__(self, type_list, stenosis_list, arr_columns=256, num_samples=32):
        assert arr_columns > 217, print('num of arr_columns should be greater than 217')
        assert arr_columns % num_samples == 0 and num_samples <= arr_columns, print('arr_columns should be a multiple of num_samples and >= num_samples.')
        self.num_samples = num_samples  # 采样次数
        self.arr_columns = arr_columns  # 得到类别平衡矩阵的列数

        type_unique, type_counts = np.unique(type_list, return_counts=True)
        stenosis_unique, stenosis_counts = np.unique(stenosis_list, return_counts=True)

        print('type label: {}, type label num: {}'.format(type_unique.tolist(), type_counts.tolist()))
        print('stenosis label: {}, stenosis label num: {}'.format(stenosis_unique.tolist(), stenosis_counts.tolist()))
        print('--' * 30)

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
        return self.num_samples

    def _shuffle_along_axis(self, a, axis):  # 在指定轴上打乱numpy数组
        idx = np.random.rand(*a.shape).argsort(axis=axis)
        return np.take_along_axis(a,idx,axis=axis)


if __name__ == "__main__":
    data_path = '/Users/gaoyibo/Datasets/plaque_data_whole'
    set_seed(57)

    case_list = sorted(os.listdir(data_path))  # 病例列表
    case_list = [os.path.join(data_path, case) for case in case_list]
    print('total case num: ' + str(len(case_list)))

    train_paths, val_paths, test_paths = split_dataset(case_list)

    # 调试Train_Dataset
    train_dataset = Train_Dataset(train_paths, transform=transforms.Compose([Data_Augmenter()]))
    balanced_sampler = BalancedSampler(train_dataset.type_list, train_dataset.stenosis_list)
    train_loader = DataLoader(train_dataset, batch_sampler=balanced_sampler)

    for idx, (image, plaque_type, stenosis) in enumerate(train_loader):
        print(image.shape)
        plt.imshow(image[0, 0, 2, :, :], cmap='gray')
        plt.show()
        break

    # 调试Eval_Dataset
    # val_dataset = Eval_Dataset(val_paths)
    # for i in range(len(val_dataset)):
    #     image, type, stenosis = val_dataset[i]
    #     if max(type) != 0:
    #         print(type)
    #         print(stenosis)
    #         break
