import torch
import logging
from tqdm import tqdm
from utils import get_metrics

def train(args, model, train_loader, criterion, optimizer, iter_num, writer):
    model.train()
    type_pred_list = []
    type_label_list = []
    stenosis_pred_list = []
    stenosis_label_list = []
    loop = tqdm(enumerate(train_loader), total=len(train_loader), leave=False, unit='iter')
    for i_batch, (image, plaque_type, stenosis) in loop:

        loop.set_description(f'Iter {iter_num}')

        image, plaque_type, stenosis = image.to(args.device).float(), plaque_type.to(args.device), stenosis.to(args.device)
        type_output, stenosis_output = model(image, steps=args.sliding_steps, device=args.device)

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

        if iter_num < args.iteration:
            lr_ = max(args.lr * (1.0 - iter_num / args.iteration) ** 0.9, 1e-4)

        for param_group in optimizer.param_groups:
            param_group['lr'] = lr_
        writer.add_scalar('train/lr', lr_, iter_num + 1)

        iter_num  = iter_num + 1
        type_loss, stenosis_loss, loss = round(type_loss.item(), 4), round(stenosis_loss.item(), 4), round(loss.item(), 4)
        writer.add_scalar('train/total_loss', loss, iter_num)
        writer.add_scalar('train/type_loss', type_loss, iter_num)
        writer.add_scalar('train/stenosis_loss', stenosis_loss, iter_num)

        loop.set_postfix(total_loss=loss, type_loss=type_loss, stenosis_loss=stenosis_loss)

        if i_batch == len(train_loader) or iter_num >= args.iteration:
            break
    loop.close()
    
    type_acc, type_f1 = get_metrics(type_label_list, type_pred_list)
    stenosis_acc, stenosis_f1 = get_metrics(stenosis_label_list, stenosis_pred_list)
    for i in range(len(type_acc)):
        writer.add_scalar('train/type_{}_acc'.format(i), type_acc[i], iter_num)
        writer.add_scalar('train/type_{}_f1'.format(i), type_f1[i], iter_num)
    for i in range(len(stenosis_acc)):
        writer.add_scalar('train/stenosis_{}_acc'.format(i), stenosis_acc[i], iter_num)
        writer.add_scalar('train/stenosis_{}_f1'.format(i), stenosis_f1[i], iter_num)
    performance = (type_acc.mean() + stenosis_acc.mean()) / 2
    writer.add_scalar('train/performance', performance, iter_num)
    writer.add_scalar('train/type_mean_acc', type_acc.mean(), iter_num)
    writer.add_scalar('train/type_mean_f1', type_f1.mean(), iter_num)
    writer.add_scalar('train/stenosis_mean_acc', stenosis_acc.mean(), iter_num)
    writer.add_scalar('train/stenosis_mean_f1', stenosis_f1.mean(), iter_num)
    print('\n')
    logging.info('training: performance: %.2f, acc_type: %.2f, f1_type: %.2f, acc_stenosis: %.2f, f1_stenosis: %.2f' %(performance, type_acc.mean(), type_f1.mean(), stenosis_acc.mean(), stenosis_f1.mean()))

    return iter_num


def evaluate(args, model, val_loader, epoch_num, writer):
    with torch.no_grad():
        model.eval()
        type_pred_list = []
        type_label_list = []
        stenosis_pred_list = []
        stenosis_label_list = []
        loop = tqdm(enumerate(val_loader), total=len(val_loader), leave=False, unit='iter')
        for i_batch, (image, plaque_type, stenosis) in loop:

            loop.set_description(f'Epoch {epoch_num}')

            image, plaque_type, stenosis = image.to(args.device).float(), plaque_type.to(args.device), stenosis.to(args.device)
            type_output, stenosis_output = model(image, steps=args.sliding_steps, device=args.device)

            type_pred_list.extend(torch.argmax(torch.softmax(type_output, dim=1), dim=1).tolist())
            stenosis_pred_list.extend(torch.argmax(torch.softmax(stenosis_output, dim=1), dim=1).tolist())
            type_label_list.extend(plaque_type.tolist())
            stenosis_label_list.extend(stenosis.tolist())

        #  评价指标计算和记录
        print('\n')
        logging.info(f'Epoch {epoch_num} Evaluation')

        # segment-level
        type_acc, type_f1 = get_metrics(type_label_list, type_pred_list)
        stenosis_acc, stenosis_f1 = get_metrics(stenosis_label_list, stenosis_pred_list)
        for i in range(len(type_acc)):
            writer.add_scalar('segment_eval/type_{}_acc'.format(i), type_acc[i], epoch_num)
            writer.add_scalar('segment_eval/type_{}_f1'.format(i), type_f1[i], epoch_num)
        for i in range(len(stenosis_acc)):
            writer.add_scalar('segment_eval/stenosis_{}_acc'.format(i), stenosis_acc[i], epoch_num)
            writer.add_scalar('segment_eval/stenosis_{}_f1'.format(i), stenosis_f1[i], epoch_num)

        performance = (type_acc.mean() + stenosis_acc.mean()) / 2
        writer.add_scalar('segment_eval/performance', performance, epoch_num)
        writer.add_scalar('segment_eval/type_mean_acc', type_acc.mean(), epoch_num)
        writer.add_scalar('segment_eval/type_mean_f1', type_f1.mean(), epoch_num)
        writer.add_scalar('segment_eval/stenosis_mean_acc', stenosis_acc.mean(), epoch_num)
        writer.add_scalar('segment_eval/stenosis_mean_f1', stenosis_f1.mean(), epoch_num)
        logging.info('segment-level: performance: %.2f, acc_type: %.2f, f1_type: %.2f, acc_stenosis: %.2f, f1_stenosis: %.2f' %(performance, type_acc.mean(), type_f1.mean(), stenosis_acc.mean(), stenosis_f1.mean()))
        print('\n')

        return performance
