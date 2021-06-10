from networks.RCNN import RCNN


def net_factory(net_type):
    if net_type == "rcnn":
        net = RCNN()
    else:
        net = None
    return net
