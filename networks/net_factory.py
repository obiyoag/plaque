from networks.TR_Net import TR_Net_2D, TR_Net_3D
from networks.RCNN import RCNN_3D, RCNN_2D


def net_factory(args):
    if args.model == "rcnn_3d":
        net = RCNN_3D(args.window_size, args.stride, args.sliding_steps)
    elif args.model == 'rcnn_2d':
        net = RCNN_2D(args.window_size, args.stride, args.sliding_steps)
    elif args.model == 'tr_net_3d':
        net = TR_Net_3D(args.window_size, args.stride, args.sliding_steps)
    elif args.model == 'tr_net_2d':
        net = TR_Net_2D(args.window_size, args.stride, args.sliding_steps)
    else:
        net = None
    return net
