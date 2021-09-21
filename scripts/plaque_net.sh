PARROTS=false

FOLD="0 1 2 3"
MODE="2d 3d"
MODEL="tr_net"

if $PARROTS
then
    source /mnt/lustre/share/platform/env/pat_latest
    PARTITION="pat_uranus"
    EXCLUDE_NODE="SH-IDC1-10-198-8-41,SH-IDC1-10-198-8-42"
else
    source /mnt/lustre/share/spring/s0.3.3
    PARTITION="MIA"
fi

cd ..

for fold_idx in $FOLD
do
    for model in $MODEL
    do
        for mode in $MODE
        do
            srun -p ${PARTITION} -n1 --gres=gpu:1 --mpi=pmi2 --job-name=plaque_net --kill-on-bad-exit=1 \
            python -u main.py --exp_name "${mode}_${model}_fold${fold_idx}" \
            --machine server --mode ${mode} --model ${model} --fold_idx ${fold_idx}
        done
    done &
done
