import os
import json
import torch
import random
import numpy as np
import SimpleITK as sitk
from utils import set_seed, Center_Crop, Data_Augmenter, BalancedSampler, digitize_stenosis
from torch.utils.data import Dataset, DataLoader


def split_dataset(case_list, train_ratio):
    random.shuffle(case_list)
    case_num = len(case_list)
    train_num = int(case_num * train_ratio)
    train_paths = case_list[:train_num]
    val_paths = case_list[train_num:]

    print('case num in train_paths: {}'.format(len(train_paths)))
    print('case num in val_paths: {}'.format(len(val_paths)))
    print('--' * 30)

    return train_paths, val_paths

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


def sample_normal(branch_path, avg=15.84, std=11.53, min_len=2, max_len=67):  # 这里的统计值是训练集标注的segment的
    """
    从正常branch中采样出一段正常的segment
    """
    mpr_path = os.path.join(branch_path, 'mpr.nii.gz')
    mpr_itk = sitk.ReadImage(mpr_path)
    mpr_vol = sitk.GetArrayFromImage(mpr_itk)
    img_len = len(mpr_vol)

    max_len = min(max_len, img_len)  # segment的最大值取mask_len和max_len中小的一个
    sample_len = int(np.clip(np.random.normal(avg, std), min_len, max_len))  # 用正态分布采样得到segment长度

    end_idx = (img_len - sample_len) if (img_len - sample_len != 0) else 1
    loc = int(np.random.choice(range(0, end_idx), 1))  # 确定segment起始位置
    sampled_idx = list(range(loc, loc + sample_len))

    label_list = [[0, sampled_idx, [0.0 for _ in range(sample_len)]]]

    return label_list


def get_path2label_dict(case_list, failed_branch_list, sample_normal_prob):
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
                        path2label_dict[branch_path] = process_label(dict['plaques'])
                        abnormal_branch_num += 1
                    else:  # 如果相应branch是正常的，则有概率在branch中sample一段正常的segment
                        if random.uniform(0, 1) < sample_normal_prob:
                            path2label_dict[branch_path] = sample_normal(branch_path)
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
        self.index_stenosis_list = []

        path2label_dict, normal_branch_num = get_path2label_dict(paths, failed_branch_list, sample_normal_prob)
        for branch_path in list(path2label_dict.keys()):
            label = path2label_dict[branch_path]
            for seg in label:
                self.path_list.append(branch_path)
                self.type_list.append(seg[0])
                self.stenosis_list.append(digitize_stenosis(seg[2]))
                self.index_stenosis_list.append(dict(zip(seg[1], seg[2])))

        print('normal seg num in trainset: {}'.format(normal_branch_num))
        print('abnormal seg num in trainset: {}'.format(len(self.path_list) - normal_branch_num))
        print('total seg num in trainset: {}'.format(len(self.path_list)))
        print('--' * 30)

    def __len__(self):
        return len(self.path_list)

    def __getitem__(self, idx):
        path, plaque_type, index_stenosis_dict = self.path_list[idx], self.type_list[idx], self.index_stenosis_list[idx]

        mpr_path = os.path.join(path, 'mpr.nii.gz')
        mpr_itk = sitk.ReadImage(mpr_path)
        image = sitk.GetArrayFromImage(mpr_itk)

        stenosis = digitize_stenosis(index_stenosis_dict.values())
        plaque_idx = list(index_stenosis_dict.keys())
        image_left, image_right = plaque_idx[0], plaque_idx[-1]+1

        if max(index_stenosis_dict) == 0:  # 如果最大狭窄程度为0，说明是正常的血管，center_pos在中间
            center_pos = plaque_idx[len(plaque_idx) // 2]
        else:  # 如果最大狭窄程度不为0，说明是有狭窄程度的血管，center_pos在最严重帧的位置
            center_pos = max(index_stenosis_dict, key=index_stenosis_dict.get)
        
        image = image[image_left: image_right, :, :]  # 标注中最后一帧也有斑块
        image = self._pad_crop_img(image, center_pos, image_left, image_right)
        image = self.transform(image)
    
        return image, plaque_type, stenosis
        
    def _pad_crop_img(self, image, center_pos, image_left, image_right):
        """
        把图像截取或补充成定长
        """
        seg_left = center_pos - (self.seg_len - 1) // 2
        seg_right = center_pos + (self.seg_len - 1) // 2 + 1
        fixed_len_seg = np.zeros((1, self.seg_len, image.shape[1], image.shape[2]))
        
        if seg_left <= image_left and seg_right >= image_right:
            diff = image_left - seg_left
            fixed_len_seg[0, diff: diff + image.shape[0], :, :] = image
        if seg_left >= image_left and seg_right <= image_right:
            diff = seg_left - image_left
            fixed_len_seg[0] = image[diff: diff + self.seg_len, :, :]
        if seg_left >= image_left and seg_right >= image_right:
            diff = seg_left - image_left
            common_part = image[diff:, :, :]
            fixed_len_seg[0, 0: len(common_part), :, :] = common_part
        if seg_left <= image_left and seg_right <= image_right:
            diff = image_left - seg_left
            common_part = image[0: seg_right - image_left, :, :]
            fixed_len_seg[0, diff: diff + len(common_part), :, :] = common_part
        
        return fixed_len_seg


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
    train_dataset = Train_Dataset(train_paths, failed_branch_list, 0.3, transform=Data_Augmenter(prob=1), seg_len=65)
    balanced_sampler = BalancedSampler(train_dataset.type_list, train_dataset.stenosis_list, 480, 120)
    image, type, stenosis = train_dataset[0]
    image = image[0]
    print(image.shape)
    for idx in range(len(image)):
        plt.plot(image[idx])
        plt.show()
        break

    #  调试Eval_Dataset
    # val_dataset = Eval_Dataset(val_paths, failed_branch_list, 0.3, 45, transform=Center_Crop())
    # val_loader = DataLoader(val_dataset, batch_size=None, shuffle=False, collate_fn=lambda x:x)
    
    # for idx, branches_list in enumerate(val_loader):
    #     if len(branches_list) == 0:
    #         print(idx)
