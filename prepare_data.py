import os
import json
import SimpleITK as sitk
from utils import deduplicate_sort_seg


def process_label(label):
    label_list = []
    for seg in label:
        label_list.append(deduplicate_sort_seg(seg))
    label_list = sorted(label_list, key=lambda x: x[1])
    return label_list


data_path = '/Users/gaoyibo/Datasets/plaque_data_whole_new/'
case_list = sorted(os.listdir(data_path))
case_list = [os.path.join(data_path, case) for case in case_list]

path2label_dict = {}

for case_path in case_list:
    branch_list = os.listdir(case_path)
    for branch_id in branch_list:
        branch_path = os.path.join(case_path, str(branch_id))

        json_path = os.path.join(branch_path, 'plaque.json')

        try:
            with open(json_path, 'r') as f:
                dict = json.load(f)
                if len(dict['plaques']) != 0:
                    path2label_dict[branch_path] = process_label(dict['plaques'])
        except IOError:
            print('plaque json file not found')

path_list = []
type_list = []
bound_list = []
seg_idx_list = []
for branch_path, label in path2label_dict.items():
    seg_idx = 0
    for seg in label:
        path_list.append(branch_path)
        type_list.append(seg[0])
        bound_list.append(seg[1])
        seg_idx_list.append(seg_idx)
        seg_idx += 1

save_path = '/Users/gaoyibo/Datasets/data4annote/'
for path, type, bound, seg_idx in zip(path_list, type_list, bound_list, seg_idx_list):
    if type == 1 or type == 3:
        seg_path = os.path.join(save_path, path.split('/')[-2], path.split('/')[-1], str(seg_idx))
        os.makedirs(seg_path, exist_ok=True)
        left_bound, right_bound = bound[0], bound[-1] + 1
        mpr_path = os.path.join(path, 'mpr.nii.gz')
        mpr_itk  = sitk.ReadImage(mpr_path)
        image = sitk.GetArrayFromImage(mpr_itk)
        image = image[left_bound: right_bound, :, :]
        out = sitk.GetImageFromArray(image)
        sitk.WriteImage(out, os.path.join(seg_path, 'segment.nii.gz'))
        with open(os.path.join(seg_path, 'annotation.txt'), 'w') as f:
            f.write(str(type))
