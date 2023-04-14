python3 main_continual_reparam.py \
    --dataset cifar100 \
    --encoder resnet18_cifar_reparam_1x3 \
    --data_dir $DATA_DIR \
    --split_strategy class \
    --task_idx 0 \
    --max_epochs 500 \
    --num_tasks 5 \
    --gpus 0 \
    --num_workers 4 \
    --precision 16 \
    --optimizer sgd \
    --lars \
    --grad_clip_lars \
    --eta_lars 0.02 \
    --exclude_bias_n_norm \
    --scheduler warmup_cosine \
    --lr 0.3 \
    --classifier_lr 0.1 \
    --weight_decay 1e-4 \
    --batch_size 256 \
    --brightness 0.4 \
    --contrast 0.4 \
    --saturation 0.2 \
    --hue 0.1 \
    --gaussian_prob 0.0 0.0 \
    --solarization_prob 0.0 0.2 \
    --name reparam-1x3 \
    --project ISSL-barlow \
    --entity pigpeppa \
    --wandb \
    --save_checkpoint \
    --method barlow_twins \
    --proj_hidden_dim 2048 \
    --output_dim 2048 \
    --scale_loss 0.1