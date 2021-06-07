import os
import sys
import torch
import logging
import argparse
from tqdm import tqdm
import torch.optim as optim
from torchvision import transforms
from torch.nn import CrossEntropyLoss
from tensorboardX import SummaryWriter
from torch.utils.data import DataLoader

from networks.net_factory import net_factory
from utils import set_seed, Data_Augmenter, Center_Crop
from datasets import Branch_Dataset, Segment_Dataset, split_dataset, BalancedSampler, Patient_Dataset
from learning import train, evaluate, segment_evaluate, branch_evaluate, patient_evaluate


def parse_args():
    parser = argparse.ArgumentParser('main')
    parser.add_argument('--data_path', default='/Users/gaoyibo/Datasets/plaque_data_whole/', type=str, help='data path')
    parser.add_argument('--model', default='rcnn', type=str, help="select the model")
    parser.add_argument('--seed', default=57, type=int, help='random seed')
    parser.add_argument('--lr', default=1e-3, type=float, help='learning rate')
    parser.add_argument('--weight_decay', default=1e-3, type=float, help='learning rate')
    parser.add_argument('--arr_columns', default=240, type=int, help='num of cols in balanced matrix')
    parser.add_argument('--num_samples', default=80, type=int, help='num of samples per epoch')  # batch_size=(arr_columns/num_samples)*7
    parser.add_argument('--iteration', default=50000, type=int, help='nums of iteration')
    parser.add_argument('--snapshot_path', default='../', type=str, help="save path")
    parser.add_argument('--pred_unit', default=45, type=int, help='the windowing size of prediciton, default=45')
    parser.add_argument('--eval_level', default='branch', type=str, help='choose the level to eval, [segment, branch, patient]')
    
    return parser.parse_args()

def main(args, train_paths, val_paths):

    train_dataset = Segment_Dataset(train_paths, transform=transforms.Compose([Data_Augmenter()]))
    balanced_sampler = BalancedSampler(train_dataset.type_list, train_dataset.stenosis_list, args.arr_columns, args.num_samples)
    train_loader = DataLoader(train_dataset, batch_sampler=balanced_sampler)

    if args.eval_level == 'segment':
        val_dataset = Segment_Dataset(val_paths, transform=transforms.Compose([Center_Crop()]))
    elif args.eval_level == 'branch':
        val_dataset = Branch_Dataset(val_paths, args.pred_unit)
    elif args.eval_level == 'patient':
        val_dataset = Patient_Dataset(val_paths)
    else:
        raise NotImplementedError
    val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False)

    model = net_factory(args.model).to(args.device)
    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    criterion = CrossEntropyLoss()
    writer = SummaryWriter(os.path.join(args.snapshot_path, 'log'))

    iter_num = 0
    best_performance = 0
    epoch = args.iteration // len(train_loader) + 1
    iterator = tqdm(range(epoch), ncols=70)

    for epoch_num in iterator:

        # train
        # iter_num = train(args, model, train_loader, criterion, optimizer, iter_num, writer)

        # validation
        if iter_num == 0 and iter_num % 1 == 0:

            if args.eval_level == 'segment':
                performance = segment_evaluate(args, model, val_loader, iter_num, writer)
            elif args.eval_level == 'branch':
                performance = branch_evaluate(args, model, val_loader, iter_num, writer)
            elif args.eval_level == 'patient':
                performance = patient_evaluate(args, model, val_loader, iter_num, writer)
            else:
                raise NotImplementedError

            if performance > best_performance:
                best_performance = performance
                save_mode_path = os.path.join(args.snapshot_path, 'iter_{}_dice_{}.pth'.format(iter_num, round(best_performance, 4)))
                save_best = os.path.join(args.snapshot_path, '{}_best_model.pth'.format(args.model))
                torch.save(model.state_dict(), save_mode_path)
                torch.save(model.state_dict(), save_best)

            if iter_num >= args.iteration:
                break
        if iter_num >= args.iteration:
            iterator.close()
            break

    writer.close()
    logging.info("Best performance: {}".format(best_performance))
    logging.info("Training Finished!")


if __name__ == "__main__":
    args = parse_args()
    set_seed(args.seed)
    args.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    args.snapshot_path = "./snapshot/{}/".format(args.model)

    if not os.path.exists(args.snapshot_path):
        os.makedirs(args.snapshot_path)
    
    logging.basicConfig(filename=args.snapshot_path+"/log.txt", level=logging.INFO, format='[%(asctime)s.%(msecs)03d] %(message)s', datefmt='%H:%M:%S')
    logging.getLogger().addHandler(logging.StreamHandler(sys.stdout))
    logging.info(str(args))

    case_list = sorted(os.listdir(args.data_path))  # 病例列表
    case_list = [os.path.join(args.data_path, case) for case in case_list]
    logging.info('total case num: ' + str(len(case_list)))

    train_paths, val_paths, test_paths = split_dataset(case_list)

    main(args, train_paths, val_paths)
