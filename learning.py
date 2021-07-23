import torch
import logging
from tqdm import tqdm
from utils import get_metrics, merge_plaque, get_branch_stenosis, digitize_stenosis

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
        pad_len = (args.pred_unit - 1) // 2

        #  得到验证集中每个病例，每个branch的预测
        type_record = []
        stenosis_record = []
        label_record = []
        loop = tqdm(enumerate(val_loader), total=len(val_loader), leave=False, unit='patient')
        for i_batch, branches_list in loop:

            branches_type_list = []
            branches_stenosis_list = []
            branches_label_list = []

            for branch in branches_list[0]:  # 对于每一个branch

                image, label = branch['image'], branch['label']
                length = image.size(2) - 2 * pad_len
                tmp_type_list = []
                tmp_stenosis_list = []

                for i in range(length):

                    input = image[:, :, i: i + args.pred_unit, :, :].float().to(args.device)
                    type_output, stenosis_output = model(input, steps=5, device=args.device)
                    tmp_type_list.append(torch.argmax(torch.softmax(type_output, dim=1)).item())
                    tmp_stenosis_list.append(torch.argmax(torch.softmax(stenosis_output, dim=1)).item())

                tmp_type_list = merge_plaque(tmp_type_list)
                branches_type_list.append(tmp_type_list)
                branches_stenosis_list.append(tmp_stenosis_list)
                branches_label_list.append(label)

            # type_record&stenosis_record size: (patient_num, branch_num, branch_length)
            # label_record size: (patient_num, branch_num, seg_num)
            type_record.append(branches_type_list)
            stenosis_record.append(branches_stenosis_list)
            label_record.append(branches_label_list)

            loop.set_description(f'Patient [{i_batch}/{len(val_loader)}]')
        loop.close()

        # 得到预测结果后进行三个level的评估和记录
        seg_type_label = []
        seg_type_pred = []
        seg_stenosis_label = []
        seg_stenosis_pred = []
        branch_label = []
        branch_pred = []
        patient_label = []
        patient_pred = []
        for patient_id in range(len(val_loader)):
            patient_tmp_label = []
            patient_tmp_pred = []
            for branch_id in range(len(stenosis_record[patient_id])):
                label = label_record[patient_id][branch_id]
                stenosis_pred = stenosis_record[patient_id][branch_id]

                for seg in label:
                    seg_type_label.append(seg[0])
                    seg_stenosis_label.append(digitize_stenosis(seg[2]))
                    s_idx, e_idx = seg[1][0], seg[1][-1]
                    type_pred = type_record[patient_id][branch_id][s_idx: e_idx]
                    seg_type_pred.append(max(type_pred, key=type_pred.count))
                    seg_stenosis_pred.append(max(stenosis_pred[s_idx: e_idx]))

                branch_stenosis_label = get_branch_stenosis(label)
                branch_stenosis_pred = max(stenosis_pred)
                branch_label.append(branch_stenosis_label)
                branch_pred.append(branch_stenosis_pred)

                patient_tmp_label.append(branch_stenosis_label)
                patient_tmp_pred.append(branch_stenosis_pred)

            patient_label.append(max(patient_tmp_label))
            patient_pred.append(max(patient_tmp_pred))

        #  评价指标计算和记录
        print('\n')
        logging.info(f'Epoch {epoch_num} Evaluation')

        # segment-level
        type_acc, type_f1 = get_metrics(seg_type_label, seg_type_pred)
        stenosis_acc, stenosis_f1 = get_metrics(seg_stenosis_label, seg_stenosis_pred)
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

        #  branch-level
        branch_acc, branch_f1 = get_metrics(branch_label, branch_pred)
        for i in range(len(branch_acc)):
            writer.add_scalar('branch_eval/stenosis_{}_acc'.format(i), branch_acc[i], epoch_num)
            writer.add_scalar('branch_eval/stenosis_{}_f1'.format(i), branch_f1[i], epoch_num)
        writer.add_scalar('branch_eval/stenosis_mean_acc', branch_acc.mean(), epoch_num)
        writer.add_scalar('branch_eval/stenosis_mean_f1', branch_f1.mean(), epoch_num)
        logging.info('branch-level: acc_stenosis: %.2f, f1_stenosis: %.2f' %(branch_acc.mean(), branch_f1.mean()))

        #  patient-level
        patient_acc, patient_f1 = get_metrics(patient_label, patient_pred)
        for i in range(len(patient_acc)):
            writer.add_scalar('patient_eval/stenosis_{}_acc'.format(i), patient_acc[i], epoch_num)
            writer.add_scalar('patient_eval/stenosis_{}_f1'.format(i), patient_f1[i], epoch_num)
        writer.add_scalar('patient_eval/stenosis_mean_acc', patient_acc.mean(), epoch_num)
        writer.add_scalar('patient_eval/stenosis_mean_f1', patient_f1.mean(), epoch_num)
        logging.info('patient-level: acc_stenosis: %.2f, f1_stenosis: %.2f' %(patient_acc.mean(), patient_f1.mean()))
        print('\n')

        return performance
