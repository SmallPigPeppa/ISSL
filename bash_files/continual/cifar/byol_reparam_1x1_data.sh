python3 main_continual_reparam.py \
    --dataset cifar100 \
    --encoder resnet18_cifar_reparam \
    --data_dir $DATA_DIR \
    --split_strategy data \
    --task_idx 0 \
    --max_epochs 500 \
    --num_tasks 5 \
    --gpus 0 \
    --precision 16 \
    --optimizer sgd \
    --lars \
    --grad_clip_lars \
    --eta_lars 0.02 \
    --exclude_bias_n_norm \
    --scheduler warmup_cosine \
    --lr 1.0 \
    --classifier_lr 0.1 \
    --weight_decay 1e-5 \
    --batch_size 256 \
    --num_workers 5 \
    --brightness 0.4 \
    --contrast 0.4 \
    --saturation 0.2 \
    --hue 0.1 \
    --gaussian_prob 0.0 0.0 \
    --solarization_prob 0.0 0.2 \
    --name reparam-1x1-data \
    --project ISSL-byol \
    --entity pigpeppa \
    --wandb \
    --save_checkpoint \
    --method byol \
    --output_dim 256 \
    --proj_hidden_dim 4096 \
    --pred_hidden_dim 4096 \
    --base_tau_momentum 0.99 \
    --final_tau_momentum 1.0 \
    --momentum_classifier