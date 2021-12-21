source /mnt/lustre/share/spring/s0.3.4
PARTITION="pat_mercury"
FOLD_IDX="0 1 2 3"

cd ../..

for fold_idx in $FOLD_IDX
do
    exp_name="tfs_fold${fold_idx}"
    srun -p ${PARTITION} -n1 --gres=gpu:1 --mpi=pmi2 --job-name=${exp_name} --kill-on-bad-exit=1 -w SH-IDC1-10-198-4-80 \
    python -u mae_finetune.py --machine server \
    --exp_name ${exp_name} --fold_idx ${fold_idx} &
done
