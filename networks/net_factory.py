from networks.TR_Net import TR_Net
from networks.RCNN import RCNN, RCNN_2D


def net_factory(args):
    if args.model == "rcnn":
        net = RCNN()
    elif args.model == 'rcnn_2d':
        net = RCNN_2D(args.window_size, args.stride)
    elif args.model == 'tr_net':
        net = TR_Net()
    else:
        net = None
    return net
