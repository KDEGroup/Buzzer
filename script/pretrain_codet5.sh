#!/bin/bash



CUDA_VISIBLE_DEVICES=1,3,4,5,6,7 torchrun --nnodes 1 --nproc_per_node 6 --master_port 29501 src/train/train_codet5/main.py \
                --batch-size 8 --gradient-accumulation-steps 2 \
                --fp16 --logging-steps 5 \
                --save-steps 2000 \
                --eval-steps 2000 \
                --do-eval --learning-rate 5e-5 \
                --model-name codet5_all_pretrain --dataset-mode pre_train


CUDA_VISIBLE_DEVICES=1,3,4,5,6,7 torchrun --nnodes 1 --nproc_per_node 6 --master_port 29501 src/train/train_codet5/main.py \
                --batch-size 8 --gradient-accumulation-steps 2 \
                --fp16 --logging-steps 5 \
                --save-steps 2000 \
                --eval-steps 2000 \
                --do-eval --learning-rate 5e-5 \
                --model-name codet5_all_shadow --dataset-mode shadow

CUDA_VISIBLE_DEVICES=1,3,4,5,6,7 torchrun --nnodes 1 --nproc_per_node 6 --master_port 29501 src/train/train_codet5/main.py \
                --batch-size 8 --gradient-accumulation-steps 2 \
                --fp16 --logging-steps 5 \
                --save-steps 2000 \
                --eval-steps 2000 \
                --do-eval --learning-rate 5e-5 \
                --model-name codet5_all_caliberate --dataset-mode caliberate                