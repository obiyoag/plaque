source ../../../../../cache/share/spring/s0.3.3
cd ..
srun -p MIA -n1 --gres=gpu:1 --mpi=pmi2 --job-name=plaque --kill-on-bad-exit=1 \
-w SH-IDC1-10-5-40-220 \
python -u main.py \
--machine server \
--exp_name step_3 \
--sliding_steps 3
