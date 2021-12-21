source /mnt/lustre/share/spring/s0.3.4
PARTITION="pat_mercury"

cd ../..

mask_ratio=0.6
exp_name="mask_ratio_${mask_ratio}"

srun -p ${PARTITION} -n1 --gres=gpu:1 --mpi=pmi2 --job-name=${exp_name} --kill-on-bad-exit=1 -w SH-IDC1-10-198-4-80 \
python -u mae_pretrain.py --machine server \
--exp_name ${exp_name}  --mask_ratio ${mask_ratio}
