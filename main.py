import os
import sys
import json
import torch
import shutil
import logging
import argparse
import numpy as np
from tqdm import tqdm
import torch.optim as optim
from torch.nn import CrossEntropyLoss
from tensorboardX import SummaryWriter
from torch.utils.data import DataLoader

from networks.net_factory import net_factory
from utils import set_seed, Data_Augmenter, Center_Crop, BalancedSampler
from datasets import split_dataset, Train_Dataset, Eval_Dataset
from learning import train, evaluate


def parse_args():
    parser = argparse.ArgumentParser('main')
    parser.add_argument('--model', default='rcnn', type=str, help="select the model")
    parser.add_argument('--exp_name', default='exp', type=str, help="the name of the experiment")
    parser.add_argument('--machine', default='server', type=str, help="the machine for training")
    parser.add_argument('--seed', default=57, type=int, help='random seed')
    parser.add_argument('--lr', default=1e-3, type=float, help='learning rate')
    parser.add_argument('--weight_decay', default=1e-3, type=float, help='learning rate')
    parser.add_argument('--arr_columns', default=360, type=int, help='num of cols in balanced matrix')
    parser.add_argument('--num_samples', default=120, type=int, help='num of samples per epoch')  # batch_size=(arr_columns/num_samples)*7
    parser.add_argument('--iteration', default=50000, type=int, help='nums of iteration')
    parser.add_argument('--snapshot_path', default='../', type=str, help="save path")
    parser.add_argument('--pred_unit', default=45, type=int, help='the windowing size of prediciton, default=45')
    parser.add_argument('--train_ratio', default=0.70, type=float, help='the training ratio of all data')
    parser.add_argument('--sample_normal_prob', default=0.3, type=float, help='the prob to sample normal segment in a branch if the branch is normal')
    parser.add_argument('--val_time', default=100, type=int, help='the validation times in training')
    parser.add_argument('--sliding_steps', default=9, type=int, help='the num of sliding cudes along a segment (should be odd)')
    
    return parser.parse_args()

def main(args):

    case_list = sorted(os.listdir(args.data_path))  # 病例列表
    case_list = [os.path.join(args.data_path, case) for case in case_list]
    logging.info('total case num: ' + str(len(case_list)))

    train_paths, val_paths = split_dataset(case_list, args.train_ratio)

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
    train_loader = DataLoader(train_dataset, batch_sampler=balanced_sampler)

    val_dataset = Eval_Dataset(val_paths, failed_branch_list, args.sample_normal_prob, args.pred_unit, transform=Center_Crop())
    val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False, collate_fn=lambda x:x)

    model = net_factory(args.model).to(args.device)
    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    criterion = CrossEntropyLoss()
    writer = SummaryWriter(os.path.join(args.snapshot_path, 'log'))

    iter_num = 0
    best_performance = 0
    epoch = args.iteration // len(train_loader) + 1
    val_stamps = np.linspace(0, epoch, int(epoch * 0.1)).astype(int)
    iterator = tqdm(range(epoch), unit='epoch')

    for epoch_num in iterator:
        iterator.set_description(f'Epoch [{epoch_num}/{epoch}]')

        # train
        iter_num = train(args, model, train_loader, criterion, optimizer, iter_num, writer)

        # validation
        if epoch_num in val_stamps:
            performance = evaluate(args, model, val_loader, epoch_num, writer)

            if performance > best_performance:
                best_performance = performance
                save_mode_path = os.path.join(args.snapshot_path, 'epoch_{}_acc_{}.pth'.format(epoch_num, round(best_performance, 4)))
                save_best = os.path.join(args.snapshot_path, '{}_best_model.pth'.format(args.exp_name))
                torch.save(model.state_dict(), save_mode_path)
                torch.save(model.state_dict(), save_best)

        if iter_num >= args.iteration:
            iterator.close()
            break

    writer.close()
    logging.info("Best performance: {}".format(best_performance))
    logging.info("Training Finished!")


if __name__ == "__main__":
    args = parse_args()

    assert args.sliding_steps % 2 != 0, print("sliding steps should be odd")
    args.seg_len = args.sliding_steps * 5 + 20

    if args.machine == 'server':
        args.data_path = '/mnt/lustre/wanghuan3/gaoyibo/Datasets/plaque_data_whole_new/'
    elif args.machine == 'pc':
        args.data_path = '/home/gyb/Datasets/plaque_data_whole_new/'
    elif args.machine == 'laptop':
        args.data_path = '/Users/gaoyibo/Datasets/plaque_data_whole_new/'
    else:
        raise NotImplementedError

    set_seed(args.seed)
    args.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    args.snapshot_path = "./snapshot/{}/".format(args.exp_name)

    if os.path.exists(args.snapshot_path):
        shutil.rmtree(args.snapshot_path)
    os.makedirs(args.snapshot_path)
    
    logging.basicConfig(filename=args.snapshot_path+"/log.txt", level=logging.INFO, format='[%(asctime)s] %(message)s', datefmt='%m/%d, %H:%M')
    logging.getLogger().addHandler(logging.StreamHandler(sys.stdout))
    logging.info(str(args))
    print('--' * 30)

    main(args)
