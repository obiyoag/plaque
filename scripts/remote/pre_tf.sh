source /mnt/lustre/share/spring/s0.3.4
PARTITION="pat_mercury"

cd ../..

FOLD_IDX="0 1 2 3"
MASK_RATIO="0.4 0.5 0.6 0.7"


for mask_ratio in $MASK_RATIO
do
    srun -p ${PARTITION} -n1 --gres=gpu:1 --mpi=pmi2 --job-name=${exp_name} --kill-on-bad-exit=1 -w SH-IDC1-10-198-4-80 \
    python mae_pretrain.py --exp_name "mask_ratio_${mask_ratio}_mae" --mask_ratio ${mask_ratio} --batch_size 28 --machine server
    for fold_idx in $FOLD_IDX
    do
        finetune="/mnt/lustre/gaoyibo.vendor/plaque/snapshot/mask_ratio_${mask_ratio}_mae/checkpoint.pth.tar"
        srun -p ${PARTITION} -n1 --gres=gpu:1 --mpi=pmi2 --job-name=${exp_name} --kill-on-bad-exit=1 -w SH-IDC1-10-198-4-80 \
        python mae_finetune.py --finetune ${finetune} --exp_name "ft_${mask_ratio}_mae_fold${fold_idx}" --fold_idx ${fold_idx} --machine server
    done
done