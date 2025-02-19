#!/bin/bash
# 用于训练提取好的feature，如mlm中的loss


CUDA_VISIBLE_DEVICES=1 python src/clf/codebert_mia.py \
        --run_name codebert_combine_cal_wonl_wb_test --mia_type wb \
        --use_mlm --use_rtd --use_cal