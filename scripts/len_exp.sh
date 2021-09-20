MODEL="rcnn tr_net"
FOLD="0 1 2 3"

source /mnt/lustre/share/platform/env/pat_latest
cd ..

for fold_idx in $FOLD
do
    for model in $MODEL
    do
        srun -p pat_uranus -n1 --gres=gpu:1 --mpi=pmi2 --job-name=len_exp --kill-on-bad-exit=1 \
        -x SH-IDC1-10-198-8-41,SH-IDC1-10-198-8-42 \
        python -u main.py \
        --exp_name "2d_${model}_len25_fold${fold_idx}" \
        --machine server --mode 2d --model ${model} --fold_idx ${fold_idx} --seg_len 25 &
    done
done

