#!/usr/bin/env bash

export CUDA_VISIBLE_DEVICES=0,1

python -m torch.distributed.launch --nproc_per_node=2 run.py \
    --data_root /mnt/data2/romualdo_data \
    --batch_size 12 \
    --num_workers 4 \
    --dataset potsdam \
    --name mib-01 \
    --task 3-2 \
    --step 0 \
    --lr 0.01 \
    --epochs 60 \
    --backbone resnet101 \
    --method MiB

python -m torch.distributed.launch --nproc_per_node=2 run.py \
    --data_root /mnt/data2/romualdo_data \
    --batch_size 12 \
    --num_workers 4 \
    --dataset potsdam \
    --name mib-01 \
    --task 3-2 \
    --step 1 \
    --lr 0.001 \
    --epochs 60 \
    --backbone resnet101 \
    --method MiB
