from networks.TR_Net import TR_Net_2D, TR_Net_3D
from networks.RCNN import RCNN_3D, RCNN_2D
from networks.miccai_tr import transformer_network


def net_factory(args):
    if args.model == "rcnn_3d":
        net = RCNN_3D(args.window_size, args.stride)
    elif args.model == 'rcnn_2d':
        net = RCNN_2D(args.window_size, args.stride)
    elif args.model == 'tr_net_3d':
        net = TR_Net_3D(args.window_size, args.stride)
    elif args.model == 'tr_net_2d':
        net = TR_Net_2D(args.window_size, args.stride, args.seg_len, pretrain=False)
    elif args.model == 'autoencoder':
        net = TR_Net_2D(args.window_size, args.stride, args.seg_len, pretrain=True, mask_ratio=args.mask_ratio)
    elif args.model == 'miccai_tr':
        net = transformer_network(args.window_size, args.stride)
    else:
        net = None
    return net
