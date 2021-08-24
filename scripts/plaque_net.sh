FOLD="0 1 2 3"
MODE = "2d 3d"
MODEL = "rcnn tr_net"
NODE = "SH-IDC1-10-5-40-220"

cd ..
source ../../../../../cache/share/spring/s0.3.3

for mode in $MODE
    for model in $MODEL
        for fold_idx in $FOLD
        do
        srun -p MIA -n1 --gres=gpu:1 --mpi=pmi2 --job-name=plaque_net --kill-on-bad-exit=1 -w ${NODE} python -u main.py --exp_name "${mode}_${model}_fold${fold_idx}" \
        --mahine server --mode ${mode} --model --${model} --fold_idx ${fold_idx}
        done
    done &
done
