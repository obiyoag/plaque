source /mnt/lustre/share/platform/env/pat_latest
cd ..
srun -p pat_mercury -n1 --gres=gpu:1 --mpi=pmi2 --job-name=plaque --kill-on-bad-exit=1 \
-x SH-IDC1-10-198-8-41,SH-IDC1-10-198-8-42 \
python -u datasets.py
