cd ../..

FOLD_IDX="0 1 2 3"
MASK_RATIO="0.4 0.5 0.7"
snapshot_path="/media/gaoyibo/SSD/snapshot/"


for mask_ratio in $MASK_RATIO
do
    python mae_pretrain.py --exp_name "mask_ratio_${mask_ratio}_mae" --mask_ratio ${mask_ratio} --snapshot_path ${snapshot_path} --batch_size 28
    for fold_idx in $FOLD_IDX
    do
        finetune="/media/gaoyibo/SSD/snapshot/mask_ratio_${mask_ratio}_mae/checkpoint.pth.tar"
        python mae_finetune.py --finetune ${finetune} --exp_name "ft_${mask_ratio}_mae_fold${fold_idx}" --fold_idx ${fold_idx}
    done
done