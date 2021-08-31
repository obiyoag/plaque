FOLD="0 1 2 3"
MODE="2d 3d"
MODEL="rcnn tr_net"

source /mnt/lustre/share/platform/env/pat_latest
cd ..

for fold_idx in $FOLD
do
    for model in $MODEL
    do
        for mode in $MODE
        do
            srun -p pat_uranus -n1 --gres=gpu:1 --mpi=pmi2 --job-name=plaque_net --kill-on-bad-exit=1 \
            -x SH-IDC1-10-198-8-41,SH-IDC1-10-198-8-42 \
            python -u main.py --exp_name "${mode}_${model}_fold${fold_idx}" \
            --machine server --mode ${mode} --model ${model} --fold_idx ${fold_idx}
        done
    done &
done
