# coding=utf-8
import os
import sys
import json
import torch
import logging
import argparse
import numpy as np
import torch.optim as optim
from torch.nn import MSELoss
from torch.utils.data import DataLoader

from learning import pretrain_one_epoch
from networks.net_factory import net_factory
from datasets import split_dataset, Train_Dataset
from utils import set_seed, MAE_Augmenter, seed_worker


def parse_args():
    parser = argparse.ArgumentParser('main')
    parser.add_argument('--model', default='mae_pretrain', type=str, help="select the model")
    parser.add_argument('--exp_name', default='mae_pretrain', type=str, help="the name of the experiment")
    parser.add_argument('--machine', default='pc', type=str, help="the machine for training")
    parser.add_argument('--seed', default=57, type=int, help='random seed')
    parser.add_argument('--batch_size', default=28, type=int, help="size of minibatch")
    parser.add_argument('--lr', default=1e-3, type=float, help='learning rate')
    parser.add_argument('--weight_decay', default=1e-3, type=float, help='learning rate')
    parser.add_argument('--arr_columns', default=480, type=int, help='num of cols in balanced matrix')
    parser.add_argument('--num_samples', default=120, type=int, help='num of samples per epoch')  # batch_size=(arr_columns/num_samples)*7
    parser.add_argument('--epochs', default=200, type=int, help='nums of epochs')
    parser.add_argument('--snapshot_path', default='../', type=str, help="save path")
    parser.add_argument('--train_ratio', default=0.75, type=float, help='the training ratio of all data')
    parser.add_argument('--sample_normal_prob', default=0.3, type=float, help='the prob to sample normal segment in a branch if the branch is normal')
    parser.add_argument('--val_freq', default=1.0, type=float, help='the validation frequency')
    parser.add_argument('--fold_idx', default=None, type=int, help="idx of fold for 4 fold validation")
    parser.add_argument('--num_workers', default=0, type=int, help="num of workers in dataloader")
    parser.add_argument('--pin_memory', default=None, help="use pin_memory or not")
    parser.add_argument('--seg_len', default=17, type=int, help="the length of a segment")
    parser.add_argument('--mode', default='2d', type=str, help="mode of the network, 2d or 3d")
    parser.add_argument('--resume', action='store_true')
    parser.add_argument('--mask_ratio', default=0.75, type=float, help="mask ratio of mae")

    return parser.parse_args()

def main(args):

    case_list = sorted(os.listdir(args.data_path))  # 病例列表
    case_list = [os.path.join(args.data_path, case) for case in case_list]
    logging.info('total case num: ' + str(len(case_list)))

    train_paths, _ = split_dataset(case_list, args.fold_idx, args.train_ratio)

    try:
        with open('failed_branches.json', 'r') as f:
            json_dict = json.load(f)
            failed_branch_list = json_dict['failed_branches']  # 没有通过检测的branch
            print('failed branch num in the dataset: {}'.format(len(failed_branch_list)))
            print('--' * 30)
    except IOError:
        print('failed_branches.json not found.')

    train_dataset = Train_Dataset(train_paths, failed_branch_list, args.sample_normal_prob, args.seg_len, transform=MAE_Augmenter(args.seg_len, args.mask_ratio))
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, pin_memory=args.pin_memory, num_workers=args.num_workers, worker_init_fn=seed_worker)

    model = net_factory(args).to(args.device)
    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    criterion = MSELoss()

    start_epoch = 0
    args.iteration = args.epochs * len(train_loader)
    val_stamps = np.linspace(0, args.epochs, int(args.epochs * args.val_freq)).astype(int)

    if args.resume:
        checkpoint_path = os.path.join(args.snapshot_path, 'checkpoint.pth.tar')
        print(checkpoint_path)
        assert os.path.isfile(checkpoint_path)
        checkpoint = torch.load(checkpoint_path)
        start_epoch = checkpoint['epoch']
        model.load_state_dict(checkpoint['model'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        print('Load checkpoint at epoch %d.' % start_epoch)

    for epoch in range(start_epoch, args.epochs):

        loss = pretrain_one_epoch(args, model, train_loader, criterion, optimizer)
        logging.info('epoch: %d, loss: %e', epoch, loss)

        if epoch in val_stamps:

            checkpoint = {
                'epoch': epoch + 1,
                'model': model.state_dict(),
                'optimizer': optimizer.state_dict(),
            }
            checkpoint_path = os.path.join(args.snapshot_path, 'checkpoint.pth.tar')
            torch.save(checkpoint, checkpoint_path)

    logging.info("MAE Pretraining Finished!")


if __name__ == "__main__":
    args = parse_args()

    assert args.seg_len % 2 != 0, print("segment length should be odd")

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
