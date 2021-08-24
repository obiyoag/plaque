FOLD="0 1 2 3"
WINDOW_SIZE="1 3 5"

cd ..
for window_size in $WINDOW_SIZE
do
    for fold_idx in $FOLD
    do
        python main.py --exp_name "size${window_size}_fold${fold_idx}" --machine pc --mode 2d --window_size ${window_size} --fold_idx ${fold_idx}
    done &
done
