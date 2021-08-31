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
            if cols[0] == 'Valid':
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
    return [np.array(performance_list), np.array(type_acc_list), np.array(type_f1_list), np.array(stenosis_acc_list), np.array(stenosis_f1_list), \
            np.array(no_stenosis_acc), np.array(no_stenosis_f1), np.array(non_sig_acc), np.array(non_sig_f1), np.array(sig_acc), np.array(sig_f1), \
            np.array(no_plaque_acc), np.array(no_plaque_f1),np.array(cal_acc),np.array(cal_f1), np.array(non_cal_acc),np.array(non_cal_f1), \
            np.array(mixed_acc), np.array(mixed_f1)]

def cross_evaluate_4fold(exp_name):
    result_list = [0 for i in range(len(name_list))]
    folds = [0, 1, 2, 3]

    for fold_idx in folds:
        path = os.path.join('snapshot', exp_name + '_fold' + str(fold_idx), 'log.txt')
        fold_list = get_one_fold_result(path)
        for idx in range(len(result_list)):
            result_list[idx] += fold_list[idx]

    for idx in range(len(result_list)):
        result_list[idx] /= len(folds)
    return result_list

def plot_results(result_list):
    plt.figure(figsize=(10, 5))
    for i in range(len(result_list)):
        ax = plt.subplot(231 + i)
        ax.set_title(name_list[i])
        plt.plot(range(len(result_list[i])), result_list[i], label='result')
        plt.legend()
    plt.subplots_adjust(wspace=0.3, hspace=0.3)
    plt.show()

def get_best_result(result_list):
    best_epoch = np.argmax(result_list[0])
    print('best_epoch {}'.format(best_epoch))
    for i in range(len(name_list)):
        print(name_list[i] + " " +str(result_list[i][best_epoch]))


if __name__ == '__main__':
    result_list = cross_evaluate_4fold()
    plot_results(result_list)
    get_best_result(result_list)
