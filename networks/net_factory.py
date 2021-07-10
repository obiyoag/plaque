from networks.TR_Net import TR_Net
from networks.RCNN import RCNN, RCNN_2D


def net_factory(net_type):
    if net_type == "rcnn":
        net = RCNN()
    elif net_type == 'rcnn_2d':
        net = RCNN_2D()
    elif net_type == 'tr_net':
        net = TR_Net()
    else:
        net = None
    return net
