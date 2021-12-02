# coding=utf-8
import os
import sys
import json
import time
import torch
import shutil
import logging
import argparse
import numpy as np
import torch.optim as optim
from datetime import timedelta
from torch.nn import CrossEntropyLoss
from torch.utils.data import DataLoader

from learning import train, evaluate
from networks.net_factory import net_factory
from datasets import split_dataset, Train_Dataset
from utils import set_seed, Data_Augmenter, Center_Crop, BalancedSampler


def parse_args():
    parser = argparse.ArgumentParser('main')
    parser.add_argument('--model', default='rcnn', type=str, help="select the model")
    parser.add_argument('--exp_name', default='exp', type=str, help="the name of the experiment")
    parser.add_argument('--machine', default='pc', type=str, help="the machine for training")
    parser.add_argument('--seed', default=57, type=int, help='random seed')
    parser.add_argument('--lr', default=1e-3, type=float, help='learning rate')
    parser.add_argument('--weight_decay', default=1e-3, type=float, help='learning rate')
    parser.add_argument('--arr_columns', default=480, type=int, help='num of cols in balanced matrix')
    parser.add_argument('--num_samples', default=120, type=int, help='num of samples per epoch')  # batch_size=(arr_columns/num_samples)*7
    parser.add_argument('--iteration', default=50000, type=int, help='nums of iteration')
    parser.add_argument('--snapshot_path', default='../', type=str, help="save path")
    parser.add_argument('--train_ratio', default=0.75, type=float, help='the training ratio of all data')
    parser.add_argument('--sample_normal_prob', default=0.3, type=float, help='the prob to sample normal segment in a branch if the branch is normal')
    parser.add_argument('--val_freq', default=1.0, type=float, help='the validation frequency')
    parser.add_argument('--fold_idx', default=None, type=int, help="idx of fold for 4 fold validation")
    parser.add_argument('--num_workers', default=0, type=int, help="num of workers in dataloader")
    parser.add_argument('--pin_memory', default=None, help="use pin_memory or not")
    parser.add_argument('--seg_len', default=17, type=int, help="the length of a segment")
    parser.add_argument('--window_size', default=3, type=int, help="the sliding window size")
    parser.add_argument('--sliding_steps', default=None, type=int, help='the num of sliding cudes along a segment (should be odd)')
    parser.add_argument('--mode', default='2d', type=str, help="mode of the network, 2d or 3d")
    parser.add_argument('--resume', action='store_true')
    
    return parser.parse_args()

def main(args):

    case_list = sorted(os.listdir(args.data_path))  # 病例列表
    case_list = [os.path.join(args.data_path, case) for case in case_list]
    logging.info('total case num: ' + str(len(case_list)))

    train_paths, val_paths = split_dataset(case_list, args.fold_idx, args.train_ratio)

    try:
        with open('failed_branches.json', 'r') as f:
            json_dict = json.load(f)
            failed_branch_list = json_dict['failed_branches']  # 没有通过检测的branch
            print('failed branch num in the dataset: {}'.format(len(failed_branch_list)))
            print('--' * 30)
    except IOError:
        print('failed_branches.json not found.')

    train_dataset = Train_Dataset(train_paths, failed_branch_list, args.sample_normal_prob, args.seg_len, transform=Data_Augmenter())
    balanced_sampler = BalancedSampler(train_dataset.type_list, train_dataset.stenosis_list, args.arr_columns, args.num_samples)
    train_loader = DataLoader(train_dataset, batch_sampler=balanced_sampler, pin_memory=args.pin_memory, num_workers=args.num_workers)

    val_dataset = Train_Dataset(val_paths, failed_branch_list, args.sample_normal_prob, args.seg_len, transform=Center_Crop())
    val_loader = DataLoader(val_dataset, batch_size=21, shuffle=False, pin_memory=args.pin_memory, num_workers=args.num_workers)

    model = net_factory(args).to(args.device)
    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    criterion = CrossEntropyLoss()

    start_epoch = 0
    iter_num = 0
    best_performance = 0
    args.epochs = args.iteration // len(train_loader) + 1
    val_stamps = np.linspace(0, args.epochs, int(args.epochs * args.val_freq)).astype(int)

    if args.resume:
        checkpoint_path = os.path.join(args.snapshot_path, 'checkpoint.pth.tar')
        print(checkpoint_path)
        assert os.path.isfile(checkpoint_path)
        checkpoint = torch.load(checkpoint_path)
        best_performance = checkpoint['best_performance']
        start_epoch = checkpoint['epoch']
        model.load_state_dict(checkpoint['model'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        print('Load checkpoint at epoch %d.' % start_epoch)

    for epoch in range(start_epoch, args.epochs):

        start = time.time()

        iter_num = train(args, model, train_loader, criterion, optimizer, epoch)

        if epoch in val_stamps:
            performance = evaluate(args, model, val_loader, epoch)

            is_best = performance > best_performance
            best_performance = max(best_performance, performance)
            checkpoint = {
                'best_performance': best_performance,
                'epoch': epoch + 1,
                'model': model.state_dict(),
                'optimizer': optimizer.state_dict(),
            }
            checkpoint_path = os.path.join(args.snapshot_path, 'checkpoint.pth.tar')
            torch.save(checkpoint, checkpoint_path)

            if is_best:
                shutil.copy(checkpoint_path, os.path.join(args.snapshot_path, 'best_model.pth.tar'))
                best_epoch = epoch
        
        time_per_epoch = time.time() - start
        seconds_left = int(args.epochs - epoch - 1) * time_per_epoch
        logging.info('Time per epoch: %s, Est. complete in: %s' % (str(timedelta(seconds=time_per_epoch)), str(timedelta(seconds=seconds_left))))
        logging.info('--' * 60)

        if iter_num >= args.iteration:
            break

    logging.info("Best performance: {}, Best epoch: {}".format(best_performance, best_epoch))
    logging.info("Training Finished!")


if __name__ == "__main__":
    args = parse_args()

    assert args.seg_len % 2 != 0, print("segment length should be odd")
    assert args.window_size % 2 != 0, print("window size should be odd")

    if args.mode == '2d':
        args.stride = (args.window_size - 1) // 2
        if args.stride == 0: args.stride = 1

        if args.model == 'rcnn':
            args.model = 'rcnn_2d'
        elif args.model == 'tr_net':
            args.model = 'tr_net_2d'
        elif args.model == 'miccai_tr':
            raise ValueError('miccai_tr does not support 2d mode.')
        elif args.model == 'vit':
            pass
        else:
            raise NotImplementedError

    elif args.mode == '3d':
        args.seg_len = 65
        assert args.seg_len % 5 == 0, print("segment length should be a multiple of 5")
        args.window_size = 25
        args.stride = 5

        if args.model == 'rcnn':
            args.model = 'rcnn_3d'
        elif args.model == 'tr_net':
            args.model = 'tr_net_3d'
        elif args.model == 'miccai_tr':
            pass
        elif args.model == 'vit':
            raise ValueError('vit does not support 3d mode')
        else:
            raise NotImplementedError

    args.sliding_steps = (args.seg_len - args.window_size + args.stride) // args.stride

    if args.machine == 'server':
        args.data_path = '/mnt/lustre/gaoyibo.vendor/Datasets/plaque_data_whole_new/'
        args.pin_memory = False
        args.num_workers = 4
    elif args.machine == 'pc':
        args.data_path = '/home/gaoyibo/Datasets/plaque_data_whole_new/'
        args.pin_menory = False
        args.num_workers = 8
    elif args.machine == 'laptop':
        args.data_path = '/Users/gaoyibo/Datasets/plaque_data_whole_new/'
        args.pin_menory = False
        args.num_workers = 2
    else:
        raise NotImplementedError

    set_seed(args.seed)
    args.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    args.snapshot_path = "./snapshot/{}/".format(args.exp_name)

    if not os.path.exists(args.snapshot_path):
        os.makedirs(args.snapshot_path)

    logging.basicConfig(filename=args.snapshot_path + "/log.txt", level=logging.INFO, format='%(message)s')
    logging.getLogger().addHandler(logging.StreamHandler(sys.stdout))
    logging.info(str(args))
    print('--' * 30)

    main(args)
