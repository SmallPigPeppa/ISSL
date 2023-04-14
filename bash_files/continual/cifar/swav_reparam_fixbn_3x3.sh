python3 main_continual_reparam_fixbn.py \
    --dataset cifar100 \
    --encoder resnet18_cifar_reparam_3x3 \
    --data_dir $DATA_DIR \
    --split_strategy class \
    --task_idx 0 \
    --max_epochs 500 \
    --num_tasks 5 \
    --gpus 0 \
    --precision 16 \
    --optimizer sgd \
    --lars \
    --grad_clip_lars \
    --eta_lars 0.02 \
    --scheduler warmup_cosine \
    --lr 0.6 \
    --min_lr 0.0006 \
    --classifier_lr 0.1 \
    --weight_decay 1e-6 \
    --batch_size 256 \
    --num_workers 3 \
    --brightness 0.8 \
    --contrast 0.8 \
    --saturation 0.8 \
    --hue 0.2 \
    --gaussian_prob 0.0 0.0 \
    --name reparam-fixbn-3x3 \
    --project ISSL-swav \
    --entity pigpeppa \
    --wandb \
    --save_checkpoint \
    --method swav \
    --proj_hidden_dim 2048 \
    --queue_size 3840 \
    --output_dim 128 \
    --num_prototypes 3000 \
    --epoch_queue_starts 50 \
    --freeze_prototypes_epochs 2