import torch
import random
import numpy as np
from sklearn.metrics import multilabel_confusion_matrix
from utils.augments import *
from utils.data_utils import *
from utils.mae_utils import *


def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.cuda.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.deterministic = True


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


class AverageMeter(object):

    def __init__(self):
        self.reset()

    def reset(self):
        self.avg = 0
        self.sum = 0
        self.cnt = 0

    def update(self, val, n=1):
        self.sum += val * n
        self.cnt += n
        self.avg = self.sum / self.cnt
