# -*- coding: utf-8 -*-
import os
import json
import random
import numpy as np
import SimpleITK as sitk
from torch.utils.data import Dataset
from utils import set_seed, process_label


def split_dataset(case_list, fold_idx, train_ratio):
    random.shuffle(case_list)
    if fold_idx is None:
        case_num = len(case_list)
        train_num = int(case_num * train_ratio)
        train_paths = case_list[:train_num]
        val_paths = case_list[train_num:]
    else:
        fold_list = [[] for i in range(4)]
        for idx, case_path in enumerate(case_list):
            fold_list[idx % 4].append(case_path)
        val_paths = fold_list[fold_idx]
        train_paths = []
        for idx, fold in enumerate(fold_list):
            if idx != fold_idx:
                train_paths.extend(fold)

    print('case num in train_paths: {}'.format(len(train_paths)))
    print('case num in val_paths: {}'.format(len(val_paths)))
    print('--' * 30)

    return train_paths, val_paths


class Train_Dataset(Dataset):
    def __init__(self, case_list, failed_branch_list, sample_normal_prob, seg_len, transform):
        assert seg_len % 2 != 0, print("the length of segments should be odd")
        self.seg_len = seg_len
        self.transform = transform
        self.images_list = []
        self.type_list = []
        self.stenosis_list = []

        normal_num = 0
        abnormal_num = 0

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
                        if len(dict['plaques']) != 0:  # 如果相应branch不是正常的，则得到segment
                            mpr_path = os.path.join(branch_path, 'mpr.nii.gz')
                            mpr_itk = sitk.ReadImage(mpr_path)
                            images = sitk.GetArrayFromImage(mpr_itk)
                            label = process_label(dict['plaques'], seg_len)

                            for seg in label:
                                self.images_list.append(self._crop_pad_img(images, seg[2]))
                                self.type_list.append(seg[0])
                                self.stenosis_list.append(seg[1])
                                abnormal_num += 1

                        else:  # 如果相应branch是正常的，则有概率在branch中sample一段正常的segment
                            if random.uniform(0, 1) < sample_normal_prob:
                                mpr_path = os.path.join(branch_path, 'mpr.nii.gz')
                                mpr_itk = sitk.ReadImage(mpr_path)
                                images = sitk.GetArrayFromImage(mpr_itk)
                                label = self._sample_normal(len(images))

                                for seg in label:
                                    self.images_list.append(self._crop_pad_img(images, seg[2]))
                                    self.type_list.append(seg[0])
                                    self.stenosis_list.append(seg[1])
                                    normal_num += 1

                except IOError:
                    print("plaque json file not found.")

        print('normal seg num in trainset: {}'.format(normal_num))
        print('abnormal seg num in trainset: {}'.format(abnormal_num))
        print('total seg num in trainset: {}'.format(normal_num + abnormal_num))
        print('--' * 30)

    def __len__(self):
        return len(self.images_list)

    def __getitem__(self, idx):
        image, type, stenosis = self.images_list[idx], self.type_list[idx], self.stenosis_list[idx]
        image = self.transform(image)
        return image, type, stenosis
        
    def _crop_pad_img(self, image, bounds):
        """
        把图像截取或补充成定长
        """
        left_bound, center_pos, right_bound = bounds
        left_bound = max(0, left_bound)
        right_bound = min(right_bound, len(image) - 1)
        image = image[left_bound: right_bound + 1]

        left_len = center_pos - left_bound
        right_len = right_bound - center_pos

        if left_len == 0:
            left_pad = min(len(image) - 1, self.seg_len//2)
            pad_width = ((left_pad, 0), (0, 0), (0, 0))
            image = np.pad(image, pad_width, mode='reflect')
            left_len = left_pad
        if right_len == 0:
            right_pad = min(len(image) - 1, self.seg_len//2)
            pad_width = ((0, right_pad), (0, 0), (0, 0))
            image = np.pad(image, pad_width, mode='reflect')
            right_len = right_pad

        while(len(image) != self.seg_len):
            left_pad = min(self.seg_len//2 - left_len, right_len)
            right_pad = min(self.seg_len//2 - right_len, left_len)

            pad_width = ((left_pad, right_pad), (0, 0), (0, 0))
            image = np.pad(image, pad_width, mode='reflect')
            left_len, right_len = left_len + left_pad, right_len + right_pad

        assert len(image) == self.seg_len, print('crop or pad failed')
        return np.expand_dims(image, axis=0)

    def _sample_normal(self, img_len):
        """
        从正常branch中采样出一段正常的segment
        """
        center_pos = int(np.random.choice(range(0, img_len), 1))
        left_bound = center_pos - self.seg_len//2
        right_bound = center_pos + self.seg_len//2  # 这里不应该加一，保持长度为seg_len

        label_list = [[0, 0, [left_bound, center_pos, right_bound]]]

        return label_list


class Pretrain_Dataset(Dataset):
    def __init__(self, paths, seg_len, transform):
        assert seg_len % 2 != 0, print("the length of segments should be odd")
        self.seg_len = seg_len
        self.transform = transform

        images_list = []
        for case_path in paths:
            branch_list = os.listdir(case_path)
            for branch_id in branch_list:
                mpr_path = os.path.join(case_path, str(branch_id), 'mpr.nii.gz')
                mpr_itk = sitk.ReadImage(mpr_path)
                image = sitk.GetArrayFromImage(mpr_itk)
                images_list.append(image)
        
        sample_stride = seg_len // 2
        self.seg_list = []
        for image in images_list:
            img_len = image.shape[0]
            if img_len < seg_len:
                pad_width = ((0, seg_len - img_len), (0, 0), (0, 0))
                image = np.pad(image, pad_width, mode='reflect')
            step = (img_len - seg_len + sample_stride) // sample_stride
            for i in range(step):
                self.seg_list.append(np.expand_dims(image[i * sample_stride: i * sample_stride + seg_len, :, :], axis=0))

        print('total seg num in pretrainset: {}'.format(len(self.seg_list)))
        
    def __len__(self):
        return len(self.seg_list)

    def __getitem__(self, idx):
        image = self.seg_list[idx]
        image = self.transform(image)
    
        return image
        

if __name__ == "__main__":
    data_path = '/home/gaoyibo/Datasets/plaque_data_whole_new/'
    set_seed(57)

    case_list = sorted(os.listdir(data_path))
    case_list = [os.path.join(data_path, case) for case in case_list]
    print('total case num: ' + str(len(case_list)))

    dataset = Pretrain_Dataset(case_list, 17, None)
