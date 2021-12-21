cd ..

MASK_RATIO="0.1 0.2 0.3 0.4 0.5 0.6 0.7"
snapshot_path="/media/gaoyibo/SSD/snapshot/"

for mask_ratio in $MASK_RATIO
do
    python mae_pretrain.py --exp_name "mask_ratio_${mask_ratio}" --mask_ratio ${mask_ratio} --snapshot_path ${snapshot_path}
done

python mae_finetune.py --exp_name 'train_from_scratch'

for mask_ratio in $MASK_RATIO
do
    finetune="/media/gaoyibo/SSD/snapshot/mask_ratio_${mask_ratio}/checkpoint.pth.tar"
    python mae_finetune.py --finetune ${finetune} --exp_name "finetune_${mask_ratio}"
done
