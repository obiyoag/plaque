cd ../..

FOLD_IDX="0 1 2 3"
mask_ratio=0.6
epochs=200
lr1=0.005
lr2=0.01

for fold_idx in $FOLD_IDX
do
    finetune="/media/gaoyibo/SSD/snapshot/mask_ratio_${mask_ratio}_mae/checkpoint.pth.tar"
    python mae_finetune.py --finetune ${finetune} --exp_name "ft_${mask_ratio}_lr_${lr1}_fold${fold_idx}" --fold_idx ${fold_idx} --lr ${lr1}
    python mae_finetune.py --finetune ${finetune} --exp_name "ft_${mask_ratio}_lr_${lr2}_fold${fold_idx}" --fold_idx ${fold_idx} --lr ${lr2}
done
