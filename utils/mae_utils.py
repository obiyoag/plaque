import torch
import random
import numpy as np


class RandomMaskingGenerator:
    def __init__(self, seg_len, mask_ratio):
        self.seg_len = seg_len
        self.num_mask = int(mask_ratio * self.seg_len)
        
    def __call__(self):
        mask = np.hstack([np.zeros(self.seg_len - self.num_mask), np.ones(self.num_mask)])
        np.random.shuffle(mask)
        return mask

def get_sinusoid_encoding_table(n_position, d_hid): 
    ''' Sinusoid position encoding table ''' 
    # TODO: make it with torch instead of numpy 
    def get_position_angle_vec(position): 
        return [position / np.power(10000, 2 * (hid_j // 2) / d_hid) for hid_j in range(d_hid)] 

    sinusoid_table = np.array([get_position_angle_vec(pos_i) for pos_i in range(n_position)]) 
    sinusoid_table[:, 0::2] = np.sin(sinusoid_table[:, 0::2]) # dim 2i 
    sinusoid_table[:, 1::2] = np.cos(sinusoid_table[:, 1::2]) # dim 2i+1 

    return torch.FloatTensor(sinusoid_table).unsqueeze(0)


def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)
