import os
import numpy as np
import matplotlib.pyplot as plt
import prettytable as pt


name_list = ['performance', 'type_acc', 'type_f1', 'stenosis_acc', 'stenosis_f1', 'no_stenosis_acc', 'no_stenosis_f1', 'non_significant_acc', \
             'non_significant_f1', 'significant_acc', 'significant_f1', 'no_plaque_acc', 'no_plaque_f1', 'calcified_acc', 'calcified_f1', \
             'non_calcified_acc', 'non_calcified_f1', 'mixed_acc', 'mixed_f1']

def get_one_fold_result(path, epoch):
    performance_list = []
    type_acc_list = []
    type_f1_list = []
    stenosis_acc_list = []
    stenosis_f1_list = []
    no_stenosis_acc = []
    no_stenosis_f1 = []
    non_sig_acc = []
    non_sig_f1 = []
    sig_acc = []
    sig_f1 = []
    no_plaque_acc = []
    no_plaque_f1 = []
    cal_acc = []
    cal_f1 = []
    non_cal_acc = []
    non_cal_f1 = []
    mixed_acc = []
    mixed_f1 = []
    
    with open(path, 'r') as f:
        lines = f.readlines()
        for line in lines:
            cols = line.split()
            if cols[0] == 'Valid' and int(cols[1].strip('[').split('/')[0]) < epoch:
                performance_list.append(float(cols[3]))
                type_acc_list.append(float(cols[5]))
                type_f1_list.append(float(cols[7]))
                stenosis_acc_list.append(float(cols[9]))
                stenosis_f1_list.append(float(cols[11]))
                no_stenosis_acc.append(float(cols[13].split('/')[0]))
                no_stenosis_f1.append(float(cols[13].split('/')[1]))
                non_sig_acc.append(float(cols[15].split('/')[0]))
                non_sig_f1.append(float(cols[15].split('/')[1]))
                sig_acc.append(float(cols[17].split('/')[0]))
                sig_f1.append(float(cols[17].split('/')[1]))
                no_plaque_acc.append(float(cols[19].split('/')[0]))
                no_plaque_f1.append(float(cols[19].split('/')[1]))
                cal_acc.append(float(cols[21].split('/')[0]))
                cal_f1.append(float(cols[21].split('/')[1]))
                non_cal_acc.append(float(cols[23].split('/')[0]))
                non_cal_f1.append(float(cols[23].split('/')[1]))
                mixed_acc.append(float(cols[25].split('/')[0]))
                mixed_f1.append(float(cols[25].split('/')[1]))
    return [(np.array(type_acc_list)+np.array(type_f1_list)+np.array(stenosis_acc_list)+np.array(stenosis_f1_list)) / 4, np.array(type_acc_list), np.array(type_f1_list), np.array(stenosis_acc_list), np.array(stenosis_f1_list), \
            np.array(no_stenosis_acc), np.array(no_stenosis_f1), np.array(non_sig_acc), np.array(non_sig_f1), np.array(sig_acc), np.array(sig_f1), \
            np.array(no_plaque_acc), np.array(no_plaque_f1),np.array(cal_acc),np.array(cal_f1), np.array(non_cal_acc),np.array(non_cal_f1), \
            np.array(mixed_acc), np.array(mixed_f1)]

def cross_evaluate_4fold(root_path, exp_name, epoch):
    result_list = [0 for i in range(len(name_list))]
    folds = [0, 1, 2, 3]
    for fold_idx in folds:
        path = os.path.join(root_path, 'snapshot', exp_name + '_fold' + str(fold_idx), 'log.txt')
        fold_list = get_one_fold_result(path, epoch)
        for idx in range(len(result_list)):
            result_list[idx] += fold_list[idx]

    for idx in range(len(result_list)):
        result_list[idx] /= len(folds)
    return result_list

def plot_results(exp_name_list, total_list):

    if len(exp_name_list) == 2:
        name = exp_name_list[0] + '-' + exp_name_list[1]
        plt.figure(figsize=(15, 3))
        plt.subplots_adjust(left=0.03, right=0.99, top=0.83, wspace=0.165)
    else:
        name = 'plot'
        plt.figure(figsize=(10, 5))
        plt.subplots_adjust(wspace=0.3, hspace=0.3)

    for i in range(5):

        if len(exp_name_list) == 2:
            ax = plt.subplot(151 + i)
        else:
            ax = plt.subplot(231 + i)

        ax.set_title(name_list[i])
        for idx, result_list in enumerate(total_list):
            plt.plot(range(len(result_list[i])), result_list[i], label=exp_name_list[idx])
        plt.legend()
    plt.suptitle(name)
    plt.savefig('/Users/gaoyibo/Desktop/{}.png'.format(name), dpi=300)
    plt.show()

def get_best_result(exp_name_list, total_list):
    for id, result_list in enumerate(total_list):
        for idx in range(len(result_list)):
            result_list[idx] = list(result_list[idx])
            for j in range(len(result_list[idx])):
                result_list[idx][j] = '{:.2f}'.format(result_list[idx][j])
        best_epoch = np.argmax(result_list[0])
        print('--'*30)
        print(exp_name_list[id])
        print('best_epoch {}'.format(best_epoch))
        print('performance {}'.format(result_list[0][best_epoch]))
        pt.float_format = "2.2"
        stenosis_table = pt.PrettyTable(['stenosis', 'no_stenosis', 'non_signigicant', 'significant', 'average'])
        type_table = pt.PrettyTable(['type', 'no_plaque', 'calcified', 'non_calcified', 'mixed', 'average'])

        stenosis_table.add_row(['Acc', result_list[5][best_epoch], result_list[7][best_epoch], result_list[9][best_epoch], result_list[3][best_epoch]])
        stenosis_table.add_row(['F1', result_list[6][best_epoch], result_list[8][best_epoch], result_list[10][best_epoch], result_list[4][best_epoch]])
        
        type_table.add_row(['Acc', result_list[11][best_epoch], result_list[13][best_epoch], result_list[15][best_epoch], result_list[17][best_epoch], result_list[1][best_epoch]])
        type_table.add_row(['F1', result_list[12][best_epoch], result_list[14][best_epoch], result_list[16][best_epoch], result_list[18][best_epoch], result_list[2][best_epoch]])

        print(stenosis_table)
        print(type_table)


if __name__ == '__main__':
    root_path = '/Users/gaoyibo/experimental_results/plaque/'
    exp_name_list = ['2d_rcnn', '2d_rcnn_len25', '2d_tr_net_len25', '3d_rcnn', '3d_tr_net', '2d_tr_net', '3d_miccai-tr', 'vit_len17']
    # exp_name_list = ['2d_tr_net_len25', '2d_rcnn_len25']
    total_list = []
    for exp_name in exp_name_list:
        result_list = cross_evaluate_4fold(root_path, exp_name, epoch=200)
        total_list.append(result_list)
    plot_results(exp_name_list, total_list)
    get_best_result(exp_name_list, total_list)
