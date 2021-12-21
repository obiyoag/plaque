source /mnt/lustre/share/spring/s0.3.4
PARTITION="pat_mercury"
FOLD_IDX="0 1 2 3"

cd ../..


mask_ratio=0.6

for fold_idx in $FOLD_IDX
do
    exp_name="ft_${mask_ratio}_fold${fold_idx}"
    finetune="/mnt/lustre/gaoyibo.vendor/plaque/snapshot/mask_ratio_${mask_ratio}/checkpoint.pth.tar"
    srun -p ${PARTITION} -n1 --gres=gpu:1 --mpi=pmi2 --job-name=${exp_name} --kill-on-bad-exit=1 -w SH-IDC1-10-198-4-80 \
    python -u mae_finetune.py --machine server \
    --exp_name ${exp_name}  --mask_ratio ${mask_ratio} --finetune ${finetune} --fold_idx ${fold_idx} &
done




