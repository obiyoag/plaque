FOLD_IDX="0 1 2 3"
MASK_RATIO="0.4 0.5 0.7"

for fold_idx in $FOLD_IDX
do
    for mask_ratio in $MASK_RATIO
    do
        echo "mask_ratio_${mask_ratio}_fold${fold_idx}"
        sleep 15
    done &
done