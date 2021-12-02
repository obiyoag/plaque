cd ..

finetune="./snapshot/mae_pretrain/checkpoint.pth.tar"

python mae_finetune.py --finetune ${finetune}