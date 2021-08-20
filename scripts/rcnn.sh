cd ..
srun -p MIA -n1 --gres=gpu:1 --mpi=pmi2 --job-name=plaque --kill-on-bad-exit=1 \
-w SH-IDC1-10-5-40-214 \
python -u main.py \
--machine server \
--exp_name baseline \
