import torch
import logging
from utils import get_metrics

def train(args, model, train_loader, criterion, optimizer, epoch):
    model.train()
    type_pred_list = []
    type_label_list = []
    stenosis_pred_list = []
    stenosis_label_list = []
    for i_batch, (image, plaque_type, stenosis) in enumerate(train_loader):

        image, plaque_type, stenosis = image.to(args.device).float(), plaque_type.to(args.device), stenosis.to(args.device)
        type_output, stenosis_output = model(image, device=args.device)

        type_pred_list.extend(torch.argmax(torch.softmax(type_output, dim=1), dim=1).tolist())
        stenosis_pred_list.extend(torch.argmax(torch.softmax(stenosis_output, dim=1), dim=1).tolist())
        type_label_list.extend(plaque_type.tolist())
        stenosis_label_list.extend(stenosis.tolist())

        type_loss = criterion(type_output, plaque_type)
        stenosis_loss = criterion(stenosis_output, stenosis)
        loss = 0.5 * (type_loss + stenosis_loss)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        iter_num = i_batch + epoch * len(train_loader)

        if iter_num < args.iteration:
            lr_ = max(args.lr * (1.0 - iter_num / args.iteration) ** 0.9, 1e-4)

        for param_group in optimizer.param_groups:
            param_group['lr'] = lr_

        type_loss, stenosis_loss, loss = round(type_loss.item(), 4), round(stenosis_loss.item(), 4), round(loss.item(), 4)

        if iter_num % 10 == 0:
            batch_log = 'Train [{0}][{1}/{2}]\t' \
                        'total_loss {3:.4f}\t' \
                        'type_loss {4:.4f}\t' \
                        'stenosis_loss {5:.4f}\t' \
                        'lr {6:.5f}\t'. \
                format(str(epoch + 1).zfill(3), str(i_batch).zfill(3), len(train_loader), loss, type_loss, stenosis_loss, lr_)
            logging.info(batch_log)

        if i_batch == len(train_loader) or iter_num >= args.iteration:
            break
    
    type_acc, type_f1 = get_metrics(type_label_list, type_pred_list)
    stenosis_acc, stenosis_f1 = get_metrics(stenosis_label_list, stenosis_pred_list)

    val_log = 'Train_val [{0}/{1}]\tperformance {2:.2f}\t' \
              'type_acc {3:.2f}\ttype_f1 {4:.2f}\t' \
              'stenosis_acc {5:.2f}\tstenosis_f1 {6:.2f}\t' \
              'no_stenosis {7:.2f}/{8:.2f}\t' \
              'non_sig {9:.2f}/{10:.2f}\t' \
              'sig {11:.2f}/{12:.2f}\t' \
              'no_plaque {13:.2f}/{14:.2f}\t' \
              'cal {15:.2f}/{16:.2f}\t' \
              'non_cal {17:.2f}/{18:.2f}\t' \
              'mixed {19:.2f}/{20:.2f}'. \
        format(str(epoch + 1).zfill(3), args.epochs, (type_acc.mean() + stenosis_acc.mean()) / 2, type_acc.mean(), type_f1.mean(), \
            stenosis_acc.mean(), stenosis_f1.mean(), stenosis_acc[0], stenosis_f1[0], stenosis_acc[1], stenosis_f1[1], stenosis_acc[2], \
            stenosis_f1[2], type_acc[0], type_f1[0], type_acc[1], type_f1[1], type_acc[2], type_f1[2], type_acc[3], type_f1[3])
    
    print('\n')
    logging.info(val_log)

    return iter_num


def evaluate(args, model, val_loader, epoch):
    with torch.no_grad():
        model.eval()
        type_pred_list = []
        type_label_list = []
        stenosis_pred_list = []
        stenosis_label_list = []
        for i_batch, (image, plaque_type, stenosis) in enumerate(val_loader):


            image, plaque_type, stenosis = image.to(args.device).float(), plaque_type.to(args.device), stenosis.to(args.device)
            type_output, stenosis_output = model(image, device=args.device)

            type_pred_list.extend(torch.argmax(torch.softmax(type_output, dim=1), dim=1).tolist())
            stenosis_pred_list.extend(torch.argmax(torch.softmax(stenosis_output, dim=1), dim=1).tolist())
            type_label_list.extend(plaque_type.tolist())
            stenosis_label_list.extend(stenosis.tolist())

        type_acc, type_f1 = get_metrics(type_label_list, type_pred_list)
        stenosis_acc, stenosis_f1 = get_metrics(stenosis_label_list, stenosis_pred_list)

        performance = (type_acc.mean() + stenosis_acc.mean()) / 2

        val_log = 'Valid [{0}/{1}]\tperformance {2:.2f}\t' \
                'type_acc {3:.2f}\ttype_f1 {4:.2f}\t' \
                'stenosis_acc {5:.2f}\tstenosis_f1 {6:.2f}\t' \
                'no_stenosis {7:.2f}/{8:.2f}\t' \
                'non_sig {9:.2f}/{10:.2f}\t' \
                'sig {11:.2f}/{12:.2f}\t' \
                'no_plaque {13:.2f}/{14:.2f}\t' \
                'cal {15:.2f}/{16:.2f}\t' \
                'non_cal {17:.2f}/{18:.2f}\t' \
                'mixed {19:.2f}/{20:.2f}'. \
            format(str(epoch + 1).zfill(3), args.epochs, performance, type_acc.mean(), type_f1.mean(), stenosis_acc.mean(), \
                stenosis_f1.mean(), stenosis_acc[0], stenosis_f1[0], stenosis_acc[1], stenosis_f1[1], stenosis_acc[2], \
                stenosis_f1[2], type_acc[0], type_f1[0], type_acc[1], type_f1[1], type_acc[2], type_f1[2], type_acc[3], type_f1[3])
        
        print('\n')
        logging.info(val_log)
        print('\n')

        return performance
