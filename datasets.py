import os
import json
import torch
import random
import numpy as np
import SimpleITK as sitk
from torch.utils.data import Dataset, DataLoader
from utils import set_seed, process_label


def split_dataset(args, case_list):
    random.shuffle(case_list)
    if args.fold_idx is None:
        case_num = len(case_list)
        train_num = int(case_num * args.train_ratio)
        train_paths = case_list[:train_num]
        val_paths = case_list[train_num:]
    else:
        fold_list = [[] for i in range(4)]
        for idx, case_path in enumerate(case_list):
            fold_list[idx % 4].append(case_path)
        val_paths = fold_list[args.fold_idx]
        train_paths = []
        for idx, fold in enumerate(fold_list):
            if idx != args.fold_idx:
                train_paths.extend(fold)

    print('case num in train_paths: {}'.format(len(train_paths)))
    print('case num in val_paths: {}'.format(len(val_paths)))
    print('--' * 30)

    return train_paths, val_paths


def sample_normal(branch_path, seg_len):
    """
    从正常branch中采样出一段正常的segment
    """
    mpr_path = os.path.join(branch_path, 'mpr.nii.gz')
    mpr_itk = sitk.ReadImage(mpr_path)
    mpr_vol = sitk.GetArrayFromImage(mpr_itk)
    img_len = len(mpr_vol)

    center_pos = int(np.random.choice(range(0, img_len), 1))
    left_bound = center_pos - seg_len//2
    right_bound = center_pos + seg_len//2 + 1

    label_list = [[0, 0, [left_bound, center_pos, right_bound]]]

    return label_list


def get_path2label_dict(case_list, failed_branch_list, sample_normal_prob, seg_len):
    """
    得到一个地址到label的映射字典，key为地址，value为label
    """
    path2label_dict = {}
    normal_branch_num = 0
    abnormal_branch_num = 0
    
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
                        path2label_dict[branch_path] = process_label(dict['plaques'], seg_len)
                        abnormal_branch_num += 1
                    else:  # 如果相应branch是正常的，则有概率在branch中sample一段正常的segment
                        if random.uniform(0, 1) < sample_normal_prob:
                            path2label_dict[branch_path] = sample_normal(branch_path, seg_len)
                            normal_branch_num += 1
            except IOError:
                print("plaque json file not found.")

    print('normal branch num in trainset: {}'.format(normal_branch_num))
    print('abnormal branch num in trainset: {}'.format(abnormal_branch_num))
    print('total branch num in trainset: {}'.format(normal_branch_num + abnormal_branch_num))
    print('--' * 30)
    return path2label_dict, normal_branch_num


class Train_Dataset(Dataset):
    def __init__(self, paths, failed_branch_list, sample_normal_prob, seg_len, transform):
        assert seg_len % 2 != 0, print("the length of segments should be odd")
        self.seg_len = seg_len
        self.transform = transform
        self.path_list = []
        self.type_list = []
        self.stenosis_list = []
        self.bound_list = []

        path2label_dict, normal_branch_num = get_path2label_dict(paths, failed_branch_list, sample_normal_prob, seg_len)
        for branch_path in list(path2label_dict.keys()):
            label = path2label_dict[branch_path]
            for seg in label:
                self.path_list.append(branch_path)
                self.type_list.append(seg[0])
                self.stenosis_list.append(seg[1])
                self.bound_list.append(seg[2])

        print('normal seg num in trainset: {}'.format(normal_branch_num))
        print('abnormal seg num in trainset: {}'.format(len(self.path_list) - normal_branch_num))
        print('total seg num in trainset: {}'.format(len(self.path_list)))
        print('--' * 30)

    def __len__(self):
        return len(self.path_list)

    def __getitem__(self, idx):
        path, plaque_type, stenosis, bounds = self.path_list[idx], self.type_list[idx], self.stenosis_list[idx], self.bound_list[idx]

        mpr_path = os.path.join(path, 'mpr.nii.gz')
        mpr_itk = sitk.ReadImage(mpr_path)
        image = sitk.GetArrayFromImage(mpr_itk)

        image = self._crop_pad_img(image, bounds)
        image = self.transform(image)
    
        return image, plaque_type, stenosis
        
    def _crop_pad_img(self, image, bounds):
        """
        把图像截取或补充成定长
        """
        left_bound, center_pos, right_bound = bounds
        left_bound = max(0, left_bound)
        right_bound = min(right_bound, len(image) - 1)
        image = torch.from_numpy(image[left_bound: right_bound + 1]).double()

        left_len = center_pos - left_bound
        right_len = right_bound - center_pos

        if left_len == 0:
            left_pad = min(len(image) - 1, self.seg_len//2)
            image = torch.nn.functional.pad(image.transpose(2, 0), (left_pad, 0), mode="reflect").transpose(2, 0)
            left_len = left_pad
        if right_len == 0:
            right_pad = min(len(image) - 1, self.seg_len//2)
            image = torch.nn.functional.pad(image.transpose(2, 0), (right_pad, 0), mode="reflect").transpose(2, 0)
            right_len = right_pad

        while(len(image) != self.seg_len):
            left_pad = min(left_len, self.seg_len//2 - left_len)
            right_pad = min(right_len, self.seg_len//2 - right_len)

            image = torch.nn.functional.pad(image.transpose(2, 0), (left_pad, right_pad), mode="reflect").transpose(2, 0)
            left_len, right_len = left_len + left_pad, right_len + right_pad

        assert len(image) == self.seg_len, print('crop or pad failed')
        return np.array(image.unsqueeze(0))


class Eval_Dataset(Dataset):
    def __init__(self, paths, failed_branch_list, sample_normal_prob, pred_unit, transform):
        assert pred_unit % 2 != 0, print("pred_unit should be odd.")
        self.pad_len = (pred_unit - 1) // 2
        self.transform = transform
        self.patient_list = []

        normal_branch_num = 0
        abnormal_branch_num = 0
        abnormal_seg_num = 0

        for case_path in paths:
            case_list = []
            # assert os.listdir(case_path) != 0, print('error')
            for branch_id in os.listdir(case_path):
                branch_path = os.path.join(case_path, str(branch_id))

                if branch_path in failed_branch_list:  # 排除没有通过检测的branch
                    continue

                json_path = os.path.join(branch_path, 'plaque.json')

                try:
                    with open(json_path, 'r') as f:
                        dict = json.load(f)
                        if len(dict['plaques']) != 0:  # 如果相应branch不是正常的，则得到segmentlabel
                            label = process_label(dict['plaques'])
                            case_list.append({branch_path: label})
                            abnormal_branch_num += 1
                            abnormal_seg_num += len(label)
                        else:  # 如果相应branch是正常的，则在branch中sample一段正常的segment
                            if random.uniform(0, 1) < sample_normal_prob:
                                case_list.append({branch_path: sample_normal(branch_path)})
                                normal_branch_num += 1
                except IOError:
                    print("plaque json file not found.")
            
            if len(case_list) != 0:
                self.patient_list.append(case_list)

        print('normal branch num in evalset: {}'.format(normal_branch_num))
        print('abnormal branch num in evalset: {}'.format(abnormal_branch_num))
        print('total branch num in evalset: {}'.format(normal_branch_num + abnormal_branch_num))
        print('--' * 30)

        print('normal seg num in evalset: {}'.format(normal_branch_num))
        print('abnormal seg num in evalset: {}'.format(abnormal_seg_num))
        print('total seg num in evalset: {}'.format(normal_branch_num + abnormal_seg_num))
        print('--' * 30)

    def __len__(self):
        return len(self.patient_list)
    
    def _read_img(self, branch_path):
        mpr_path = os.path.join(branch_path, 'mpr.nii.gz')
        mpr_itk = sitk.ReadImage(mpr_path)
        image = sitk.GetArrayFromImage(mpr_itk)
        length, height, width = image.shape
        pad_img = np.zeros((2 * self.pad_len + length, height, width))
        pad_img[self.pad_len: self.pad_len + image.shape[0], :, :] = image
        pad_img = np.expand_dims(pad_img, axis=0)
        pad_img = self.transform(pad_img)
        pad_img = np.expand_dims(pad_img, axis=0)
        pad_img = torch.FloatTensor(pad_img)
        return pad_img

    def __getitem__(self, idx):
        case_list = self.patient_list[idx]
        branches_list = []
        for branch in case_list:
            (branch_path, branch_label), = branch.items()
            image = self._read_img(branch_path)
            sample = {'image': image, 'label': branch_label}
            branches_list.append(sample)
        return branches_list


if __name__ == "__main__":
    data_path = '/Users/gaoyibo/Datasets/plaque_data_whole/'
    set_seed(57)

    case_list = sorted(os.listdir(data_path))  # 病例列表
    case_list = [os.path.join(data_path, case) for case in case_list]
    print('total case num: ' + str(len(case_list)))

    train_paths, val_paths = split_dataset(case_list, train_ratio=0.7)

    try:
        with open('failed_branches.json', 'r') as f:
            json_dict = json.load(f)
            failed_branch_list = json_dict['failed_branches']  # 没有通过检测的branch
            print('failed branch num in the dataset: {}'.format(len(failed_branch_list)))
    except IOError:
        print('failed_branches.json not found.')

    #  调试Train_Dataset
    import matplotlib.pyplot as plt
    from torchvision.utils import make_grid

    train_dataset = Train_Dataset(train_paths, failed_branch_list, 0.3, transform=lambda x: x, seg_len=65)
    dataloader = DataLoader(train_dataset, 32)
    for idx, (image, type, stenosis) in enumerate(dataloader):
        image = image[0].squeeze()
        plt.imshow(make_grid(image.unsqueeze(1)).transpose(2, 0))
        plt.show()
        break
