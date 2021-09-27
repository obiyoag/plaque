import os
import numpy as np
import matplotlib.pyplot as plt


name_list = ['performance', 'type_acc', 'type_f1', 'stenosis_acc', 'stenosis_f1', 'no_stenosis_acc', 'no_stenosis_f1', 'non_significant_acc', \
             'non_significant_f1', 'significant_acc', 'significant_f1', 'no_plaque_acc', 'no_plaque_f1', 'calcified_acc', 'calcified_f1', \
             'non_calcified_acc', 'non_calcified_f1', 'mixed_acc', 'mixed_f1']

def get_one_fold_result(path):
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
            if cols[0] == 'Valid' and int(cols[1].strip('[').split('/')[0]):
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

def cross_evaluate_4fold(root_path, exp_name):
    result_list = [0 for i in range(len(name_list))]
    folds = [0, 1, 2, 3]
    for fold_idx in folds:
        path = os.path.join(root_path, 'snapshot', exp_name + '_fold' + str(fold_idx), 'log.txt')
        fold_list = get_one_fold_result(path)
        for idx in range(len(result_list)):
            result_list[idx] += fold_list[idx]

    for idx in range(len(result_list)):
        result_list[idx] /= len(folds)
    return result_list

def plot_results(exp_name_list, total_list):
    name = exp_name_list[0] + '-' + exp_name_list[1]
    plt.figure(figsize=(15, 3))
    for i in range(5):
        ax = plt.subplot(151 + i)
        ax.set_title(name_list[i])
        for idx, result_list in enumerate(total_list):
            plt.plot(range(len(result_list[i])), result_list[i], label=exp_name_list[idx])
        plt.legend()
    plt.subplots_adjust(left=0.03, right=0.99, top=0.83, wspace=0.165)
    plt.suptitle(name)
    plt.savefig('/Users/gaoyibo/Desktop/{}.png'.format(name), dpi=300)
    plt.show()

def get_best_result(exp_name_list, total_list):
    for idx, result_list in enumerate(total_list):
        print('--'*30)
        print(exp_name_list[idx])
        best_epoch = np.argmax(result_list[0])
        print('best_epoch {}'.format(best_epoch))
        for i in range(len(name_list)):
            print("{}: {:.2f}".format(name_list[i], result_list[i][best_epoch]))


if __name__ == '__main__':
    root_path = '/Users/gaoyibo/experimental_results/plaque/'
    exp_name_list = ['2d_rcnn', '2d_rcnn_len25', '2d_tr_net_len25', '3d_rcnn', '3d_tr_net', '2d_tr_net', '3d_miccai-tr']
    total_list = []
    for exp_name in exp_name_list:
        result_list = cross_evaluate_4fold(root_path, exp_name)
        total_list.append(result_list)
    plot_results(exp_name_list, total_list)
    get_best_result(exp_name_list, total_list)
