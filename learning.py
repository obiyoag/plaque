import logging
import torch
import numpy as np
from sklearn.metrics import accuracy_score, f1_score
from utils import seg_digitize, get_metrics

def train(args, model, train_loader, criterion, optimizer, iter_num, writer):
    model.train()
    for i_batch, (image, plaque_type, stenosis) in enumerate(train_loader):

        image, plaque_type, stenosis = image.to(args.device).float(), plaque_type.to(args.device), stenosis.to(args.device)
        type_output, stenosis_output = model(image, steps=10, device=args.device)

        type_loss = criterion(type_output, plaque_type)
        stenosis_loss = criterion(stenosis_output, stenosis)
        loss = 0.5 * (type_loss + stenosis_loss)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        iter_num  = iter_num + 1
        writer.add_scalar('train/total_loss', loss, iter_num)
        writer.add_scalar('train/type_loss', type_loss, iter_num)
        writer.add_scalar('train/stenosis_loss', stenosis_loss, iter_num)
        logging.info('iteration %d : loss : %f, type_loss: %f, stenosis_loss: %f' %(iter_num, loss.item(), type_loss.item(), stenosis_loss.item()))
        return iter_num


def evaluate(args, model, val_loader, iter_num, writer):
    with torch.no_grad():
        model.eval()
        acc_type_list = []
        f1_type_list = []
        acc_stenosis_list = []
        f1_stenosis_list = []
        for i_batch, (image, plaque_type, stenosis) in enumerate(val_loader):

            length = image.size(2)
            plaque_type = plaque_type[0]
            stenosis = stenosis[0]

            type_pred_list = []
            stenosis_pred_list = []
            type_seg_label = []
            stenosis_seg_label = []

            for i in range(length // 45):
                input = image[:, :, i * 45: (i + 1) * 45, :, :].float().to(args.device)
                type_seg = plaque_type[i * 45: (i + 1) * 45]
                stenosis_seg = stenosis[i * 45: (i + 1) * 45]

                type_output, stenosis_output = model(input, steps=5, device=args.device)
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

        writer.add_scalar('eval/acc_type', acc_type, iter_num)
        writer.add_scalar('eval/f1_type', f1_type, iter_num)
        writer.add_scalar('eval/acc_stenosis', acc_stenosis, iter_num)
        writer.add_scalar('eval/f1_stenosis', f1_stenosis, iter_num)
        logging.info('iteration %d : performance: %f, acc_type: %f, f1_type: %f, acc_stenosis: %f, f1_stenosis: %f' %(iter_num, performance, acc_type, f1_type, acc_stenosis, f1_stenosis))

        return performance


def segment_evaluate(args, model, val_loader, iter_num, writer):
    with torch.no_grad():
        model.eval()
        type_pred_list = []
        stenosis_pred_list = []
        type_label_list = []
        stenosis_label_list = []
        for i_batch, (image, plaque_type, stenosis) in enumerate(val_loader):

            image, plaque_type, stenosis = image.to(args.device).float(), plaque_type, stenosis
            type_output, stenosis_output = model(image, steps=10, device=args.device)
            type_pred_list.append(torch.max(torch.softmax(type_output, dim=1), dim=1)[1].item())
            stenosis_pred_list.append(torch.max(torch.softmax(stenosis_output, dim=1), dim=1)[1].item())
            type_label_list.append(plaque_type.item())
            stenosis_label_list.append(stenosis.item())
        
        type_acc, type_f1 = get_metrics(type_label_list, type_pred_list)
        stenosis_acc, stenosis_f1 = get_metrics(stenosis_label_list, stenosis_pred_list)

        for i in range(len(type_acc)):
            writer.add_scalar('segment_eval/type_{}_acc'.format(i), type_acc[i], iter_num)
            writer.add_scalar('segment_eval/type_{}_f1'.format(i), type_f1[i], iter_num)
        
        for i in range(len(stenosis_acc)):
            writer.add_scalar('segment_eval/stenosis_{}_acc'.format(i), stenosis_acc[i], iter_num)
            writer.add_scalar('segment_eval/stenosis_{}_f1'.format(i), stenosis_f1[i], iter_num)
        
        performance = (type_acc.mean() + stenosis_acc.mean()) / 2
        writer.add_scalar('segment_eval/performance', performance, iter_num)
        writer.add_scalar('segment_eval/type_mean_acc', type_acc.mean(), iter_num)
        writer.add_scalar('segment_eval/type_mean_f1', type_f1.mean(), iter_num)
        writer.add_scalar('segment_eval/stenosis_mean_acc', stenosis_acc.mean(), iter_num)
        writer.add_scalar('segment_eval/stenosis_mean_f1', stenosis_f1.mean(), iter_num)

        logging.info('iteration %d : performance: %f, acc_type: %f, f1_type: %f, acc_stenosis: %f, f1_stenosis: %f' %(iter_num, performance, type_acc.mean(), type_f1.mean(), stenosis_acc.mean(), stenosis_f1.mean()))

    return performance

def branch_evaluate(args, model, val_loader, iter_num, writer):  # 在branch-level不评估种类，只评估狭窄程度
    with torch.no_grad():
        model.eval()
        pad_len = (args.pred_unit - 1) // 2

        set_pred_list = []
        set_label_list = []

        for i_batch, (image, plaque_type, stenosis) in enumerate(val_loader):

            length = image.size(2) - 2 * pad_len
            stenosis = stenosis[0]
            branch_pred_list = []

            for i in range(length):
                input = image[:, :, i: i + args.pred_unit, :, :].float().to(args.device)
                type_output, stenosis_output = model(input, steps=5, device=args.device)
                branch_pred_list.append(torch.max(torch.softmax(stenosis_output, dim=1), dim=1)[1].item())
            
            set_pred_list.append(max(branch_pred_list))
            set_label_list.append(max(stenosis))
            
        stenosis_acc, stenosis_f1 = get_metrics(set_label_list, set_pred_list)

        for i in range(len(stenosis_acc)):
            writer.add_scalar('branch_eval/stenosis_{}_acc'.format(i), stenosis_acc[i], iter_num)
            writer.add_scalar('branch_eval/stenosis_{}_f1'.format(i), stenosis_f1[i], iter_num)
        
        performance = stenosis_acc.mean()
        writer.add_scalar('branch_eval/performance', performance, iter_num)
        writer.add_scalar('branch_eval/stenosis_mean_acc', stenosis_acc.mean(), iter_num)
        writer.add_scalar('branch_eval/stenosis_mean_f1', stenosis_f1.mean(), iter_num)

        logging.info('iteration %d : performance: %f, acc_stenosis: %f, f1_stenosis: %f' %(iter_num, performance, stenosis_acc.mean(), stenosis_f1.mean()))

    return performance

def patient_evaluate(args, model, val_loader, iter_num, writer):
    pass