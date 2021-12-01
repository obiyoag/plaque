
MODEL="vit"
FOLD="0 1 2 3"

source /mnt/lustre/share/spring/s0.3.3
PARTITION="MIA"

cd ..

for fold_idx in $FOLD
do
    for model in $MODEL
    do
        srun -p ${PARTITION} -n1 --gres=gpu:1 --mpi=pmi2 --job-name=vit --kill-on-bad-exit=1 \
        python -u main.py \
        --exp_name "${model}_fold${fold_idx}" \
        --machine server --mode 2d --model ${model} --fold_idx ${fold_idx} &
    done
done

