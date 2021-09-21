PARROTS=false
source activate pytorch

cd ..

srun -p MIA -n1 --gres=gpu:1 --mpi=pmi2 --job-name=plaque_net --kill-on-bad-exit=1 python -u main.py —exp_name 2d_tr_net_fold2 —machine server —mode 2d —model tr_net —fold_idx 2 &

srun -p MIA -n1 --gres=gpu:1 --mpi=pmi2 --job-name=len_exp --kill-on-bad-exit=1 python -u main.py —exp_name 2d_tr_net_len25_fold0 —machine server —mode 2d —model tr_net —fold_idx 0 —seg_len 25 &


FOLD="1 2 3"
for fold_idx in $FOLD
do
    srun -p MIA -n1 --gres=gpu:1 --mpi=pmi2 --job-name=plaque_net --kill-on-bad-exit=1 python -u main.py —exp_name "3d_tr_net_fold${fold_idx}" —machine server —mode 3d —model tr_net —fold_idx ${fold_idx} &
done
