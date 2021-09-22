MODEL="miccai_tr"
FOLD="0 1 2 3"

source /mnt/lustre/share/spring/r0.3.3
cd ..

for fold_idx in $FOLD
do
    srun -p MIA -n1 --gres=gpu:1 --mpi=pmi2 --job-name=len_exp --kill-on-bad-exit=1 \
    python -u main.py \
    --exp_name "3d_${model}_fold${fold_idx}" \
    --machine server --mode 3d --model ${model} --fold_idx ${fold_idx} &
done