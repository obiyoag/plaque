import os
import sys
import torch
import logging
import argparse
import numpy as np
from tqdm import tqdm
import torch.optim as optim
from torchvision import transforms
from torch.nn import CrossEntropyLoss
from tensorboardX import SummaryWriter
from torch.utils.data import DataLoader
from sklearn.metrics import f1_score, accuracy_score

from networks.net_factory import net_factory
from utils import set_seed, Data_Augmenter, seg_digitize
from datasets import Eval_Dataset, Train_Dataset, split_dataset, BalancedSampler


def parse_args():
    parser = argparse.ArgumentParser('main')
    parser.add_argument('--data_path', default='/Users/gaoyibo/Datasets/plaque_data_whole', type=str, help='data path')
    parser.add_argument('--model', default='rcnn', type=str, help="select the model")
    parser.add_argument('--seed', default=57, type=int, help='random seed')
    parser.add_argument('--lr', default=1e-3, type=float, help='learning rate')
    parser.add_argument('--weight_decay', default=1e-3, type=float, help='learning rate')
    parser.add_argument('--arr_columns', default=240, type=int, help='num of cols in balanced matrix')
    parser.add_argument('--num_samples', default=40, type=int, help='num of samples per epoch')  # batch_size=(arr_columns/num_samples)*7
    parser.add_argument('--iteration', default=50000, type=int, help='nums of iteration')
    parser.add_argument('--snapshot_path', default='../', type=str, help="select the model")
    
    return parser.parse_args()

def train(args, train_paths, val_paths):

    train_dataset = Train_Dataset(train_paths, transform=transforms.Compose([Data_Augmenter()]))
    balanced_sampler = BalancedSampler(train_dataset.type_list, train_dataset.stenosis_list, args.arr_columns, args.num_samples)
    train_loader = DataLoader(train_dataset, batch_sampler=balanced_sampler)

    val_dataset = Eval_Dataset(val_paths)
    val_loader = DataLoader(val_dataset, batch_size=1, shuffle=True)

    model = net_factory(args.model).to(args.device)
    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    criterion = CrossEntropyLoss()
    writer = SummaryWriter(os.path.join(args.snapshot_path, 'log'))

    iter_num = 0
    best_performance = 0
    epoch = args.iteration // len(train_loader) + 1
    iterator = tqdm(range(epoch), ncols=70)
    for epoch_num in iterator:
        for i_batch, (image, plaque_type, stenosis) in enumerate(train_loader):

            image, plaque_type, stenosis = image.to(args.device).float(), plaque_type.to(args.device), stenosis.to(args.device)
            type_output, stenosis_output = model(image, steps=10)

            type_loss = criterion(type_output, plaque_type)
            stenosis_loss = criterion(stenosis_output, stenosis)
            loss = 0.5 * (type_loss + stenosis_loss)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            iter_num = iter_num + 1
            writer.add_scalar('info/total_loss', loss, iter_num)
            writer.add_scalar('info/type_loss', type_loss, iter_num)
            writer.add_scalar('info/stenosis_loss', stenosis_loss, iter_num)
            logging.info('iteration %d : loss : %f, type_loss: %f, stenosis_loss: %f' %(iter_num, loss.item(), type_loss.item(), stenosis_loss.item()))

            if iter_num > 0 and iter_num % 500 == 0:
                model.eval()
                acc_type_list = []
                f1_type_list = []
                acc_stenosis_list = []
                f1_stenosis_list = []
                for i_batch, (image, plaque_type, stenosis) in enumerate(val_loader):
                    length = image.size(2)
                    type_pred_list = []
                    stenosis_pred_list = []
                    type_seg_label = []
                    stenosis_seg_label = []

                    for i in range(length // 45):
                        input = image[:, :, i * 45: (i + 1) * 45, :, :].float()
                        type_seg = plaque_type[i * 45: (i + 1) * 45]
                        stenosis_seg = stenosis[i * 45: (i + 1) * 45]

                        type_output, stenosis_output = model(input, steps=5)
                        type_pred_list.append(torch.max(torch.softmax(type_output, dim=1), dim=1)[1].item())
                        stenosis_pred_list.append(torch.max(torch.softmax(stenosis_output, dim=1), dim=1)[1].item())
                        type_seg_label.append(seg_digitize(type_seg))
                        stenosis_seg_label.append(seg_digitize(stenosis_seg))
                    
                    acc_type_list.append(accuracy_score(type_seg_label, type_pred_list))
                    f1_type_list.append(f1_score(type_seg_label, type_pred_list, average="macro"))
                    acc_stenosis_list.append(accuracy_score(stenosis_seg_label, stenosis_pred_list))
                    f1_stenosis_list.append(f1_score(stenosis_seg_label, stenosis_pred_list, average="macro"))

                acc_type = np.mean(acc_type_list)
                f1_type = np.mean(f1_type_list)
                acc_stenosis = np.mean(acc_stenosis_list)
                f1_stenosis = np.mean(f1_stenosis_list)
                performance = (acc_type + acc_stenosis) / 2

                writer.add_scalar('val/acc_type', acc_type, iter_num)
                writer.add_scalar('val/f1_type', f1_type, iter_num)
                writer.add_scalar('val/acc_stenosis', acc_stenosis, iter_num)
                writer.add_scalar('val/f1_stenosis', f1_stenosis, iter_num)
                logging.info('iteration %d : performance: %f, acc_type: %f, f1_type: %f, acc_stenosis: %f, f1_stenosis: %f' %(iter_num, performance, acc_type, f1_type, acc_stenosis, f1_stenosis))
                
                if performance > best_performance:
                    best_performance = performance
                    save_mode_path = os.path.join(args.snapshot_path, 'iter_{}_dice_{}.pth'.format(iter_num, round(best_performance, 4)))
                    save_best = os.path.join(args.snapshot_path, '{}_best_model.pth'.format(args.model))
                    torch.save(model.state_dict(), save_mode_path)
                    torch.save(model.state_dict(), save_best)

                model.train()

            if iter_num % 3000 == 0:
                save_mode_path = os.path.join(args.snapshot_path, 'iter_' + str(iter_num) + '.pth')
                torch.save(model.state_dict(), save_mode_path)
                logging.info("save model to {}".format(save_mode_path))

            if iter_num >= args.iterations:
                break
        if iter_num >= args.iterations:
            iterator.close()
            break
    writer.close()
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

    train(args, train_paths, val_paths)
