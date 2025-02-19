#!/bin/bash


CUDA_VISIBLE_DEVICES=1,2,3,4,5,6 torchrun --nnodes 1 --nproc_per_node 6 --master_port 29503 src/train/train_codebert/main.py \
                    --batch-size 8 \
                    --fp16 --logging-steps 5 \
                    --save-steps 2000 \
                    --gradient-accumulation-steps 2 \
                    --eval-steps 2000 --n-epoch 20 \
                    --do-eval --learning-rate 5e-5 \
                    --eval-batch-size 16 \
                    --model-name codebert_mlmrtd_all_pre_train_50epoch_test \
                    --do-pretrain --dataset-mode pre_train 

CUDA_VISIBLE_DEVICES=1,2,3,4,5,6 torchrun --nnodes 1 --nproc_per_node 6 --master_port 29503 src/train/train_codebert/main.py \
                    --batch-size 8 \
                    --fp16 --logging-steps 5 \
                    --save-steps 2000 \
                    --gradient-accumulation-steps 2 \
                    --eval-steps 2000 --n-epoch 20 \
                    --do-eval --learning-rate 5e-5 \
                    --eval-batch-size 16 \
                    --model-name codebert_mlmrtd_all_shadow_50epoch_test \
                    --do-pretrain --dataset-mode shadow 

CUDA_VISIBLE_DEVICES=1,2,3,4,5,6 torchrun --nnodes 1 --nproc_per_node 6 --master_port 29503 src/train/train_codebert/main.py \
                    --batch-size 8 \
                    --fp16 --logging-steps 5 \
                    --save-steps 2000 \
                    --gradient-accumulation-steps 2 \
                    --eval-steps 2000 --n-epoch 20 \
                    --do-eval --learning-rate 5e-5 \
                    --eval-batch-size 16 \
                    --model-name codebert_mlmrtd_all_caliberate_50epoch_test \
                    --do-pretrain --dataset-mode caliberate 