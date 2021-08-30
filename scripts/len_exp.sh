DEL="rcnn tr_net"
NODE="SH-IDC1-10-5-40-219"

source ../../../cache/share/spring/s0.3.3
cd ..

for fold_idx in $FOLD
do
    for model in $MODEL
    do
        srun -p MIA -n1 --gres=gpu:1 --mpi=pmi2 --job-name=plaque_net --kill-on-bad-exit=1 \
        -w ${NODE} python -u main.py --exp_name "2d_${model}_len25_fold${fold_idx}" \
        --machine server --mode 2d --model ${model} --fold_idx ${fold_idx} --seg_len 25 &
    done
done

