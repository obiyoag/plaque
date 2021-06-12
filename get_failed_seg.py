import os
import json
import shutil
import numpy as np
import SimpleITK as sitk
from collections import deque
from datasets import process_label


def validate_frame(frame, threshold=0.05):
    """
    验证每一帧的斑块面积占非背景区域的面积是否超过阈值
    """
    unique, counts = np.unique(frame, return_counts=True)
    non_zero_area = sum(counts[1:])
    plaque_area = sum(counts[2:])
    if non_zero_area == 0:
        return False
    else:
        return (plaque_area / non_zero_area) > threshold


def validate_v2(mask_vol, label, constraint=None):
    if len(label) == 0:
        range_set = set()
    else:
        total_range = list()
        total_stenosis = list()
        for segment in label:
            seg_range = segment[1]
            stenosis_list = segment[2]
            total_range.extend(seg_range)
            total_stenosis.extend(stenosis_list)

        tmp_arr = np.stack([np.array(total_range), np.array(total_stenosis)], axis=0)
        if constraint is None:
            range_set = set(tmp_arr[0])
        else:
            range_set = set(tmp_arr[0][tmp_arr[1, :] >= 0])
    
    mask_list = list()
    queue = deque()
    for i in range(mask_vol.shape[0]):
        frame = mask_vol[i]

        if validate_frame(frame):
            queue.append(i)
        else:
            if len(queue) > 3:
                mask_list.extend(queue)
            queue.clear()

    mask_set = set(mask_list)
    
    if len(mask_set) != 0:
        # 两种情况：
        # 1：mask有斑块，标注有狭窄，正常斑块。
        # 2：mask有斑块，标注没有狭窄。可能是噪声或错误。要通过mask的斑块面积和长度来确定。
        result = 1 - len(mask_set - range_set) / len(mask_set)

    elif len(mask_set) == 0 and len(range_set) == 0:
        # mask没有斑块，label没有狭窄，正常血管。
        result = 1
    elif len(mask_set) == 0 and len(range_set) != 0:
        # mask没有斑块，label有狭窄。对应无斑块有狭窄的情况。此时label应为软斑块，应通过检测。
        result = 1

    return result

def get_path2label_dict(case_list):
    '''
    得到一个地址到label的映射字典，key为地址，value为label
    '''
    path2label_dict = {}
    for case in case_list:
        case_path = os.path.join(path, case)
        branch_list = os.listdir(case_path)
        for branch_id in branch_list:
            branch_path = os.path.join(case_path, str(branch_id))
            json_path = os.path.join(branch_path, 'plaque.json')

            if os.path.exists(json_path):
                with open(json_path, 'r') as f:
                    dict = json.load(f)
                    path2label_dict[branch_path] = process_label(dict['plaques'])
            else:
                shutil.rmtree(branch_path)

    return path2label_dict


if __name__ == "__main__":
    path = "/Users/gaoyibo/Datasets/plaque_data_whole_new/"
    case_list = sorted(os.listdir(path))
    case_list = [os.path.join(path, case) for case in case_list]
    'total case num: ' + str(len(case_list))

    failed_branch_list = []
    total_dict = get_path2label_dict(case_list)
    for id, branch_name in enumerate(total_dict.keys()):
        # get mask data
        mask_path = os.path.join(branch_name, 'mask.nii.gz')
        mask_itk = sitk.ReadImage(mask_path)
        mask_vol = sitk.GetArrayFromImage(mask_itk)
        # get label
        label = total_dict[branch_name]
        coverage = validate_v2(mask_vol, label)
        if coverage < 0.8:
            failed_branch_list.append(branch_name)
            print('--' * 40)
            print(coverage)
            print(branch_name)
            print(total_dict[branch_name])

    print('--' * 40)
    print(str(len(failed_branch_list)) + ' branches failed to pass the validation')

    # save failed_branches
    failed_branches = {'failed_branches': failed_branch_list}
    with open('/Users/gaoyibo/Coding/python/plaque/failed_branches.json', 'w') as f:
        json.dump(failed_branches, f)