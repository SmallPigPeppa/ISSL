python3 main_linear_reparam_all.py \
    --dataset imagenet100 \
    --encoder resnet18_reparam_3x3_1x3 \
    --data_dir $DATA_DIR \
    --train_dir imagenet100/train \
    --val_dir imagenet100/val \
    --split_strategy class \
    --num_tasks 5 \
    --max_epochs 100 \
    --gpus 0 \
    --precision 16 \
    --optimizer sgd \
    --scheduler step \
    --lr 1.0 \
    --lr_decay_steps 60 80 \
    --weight_decay 0 \
    --batch_size 256 \
    --num_workers 7 \
    --dali \
    --name reparam-3x3-1x3-fixbn \
    --project ISSL-simclr-imagenet-eval \
    --entity pigpeppa \
    --wandb \
    --linear_eval_dir 2023_04_16_14_59_26-reparam-3x3-1x3-fixbn