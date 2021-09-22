source /mnt/lustre/share/spring/r0.3.3

cd ..

srun -p MIA -n1 --gres=gpu:1 --mpi=pmi2 --job-name=2d_tr --kill-on-bad-exit=1 python -u main.py —exp_name 2d_tr_net_fold2 —machine server —mode 2d —model tr_net —fold_idx 2 &

FOLD="0 1 2 3"
for fold_idx in $FOLD
do
    srun -p MIA -n1 --gres=gpu:1 --mpi=pmi2 --job-name=3d_rcnn --kill-on-bad-exit=1 python -u main.py —exp_name "3d_rcnn_fold${fold_idx}" —machine server —mode 3d —model rcnn —fold_idx ${fold_idx} &
done

FOLD="1 3"
for fold_idx in $FOLD
do
    srun -p MIA -n1 --gres=gpu:1 --mpi=pmi2 --job-name=3d_tr --kill-on-bad-exit=1 python -u main.py —exp_name "3d_tr_net_fold${fold_idx}" —machine server —mode 3d —model tr_net —fold_idx ${fold_idx} &
done
