#!/bin/bash

CUDA_VISIBLE_DEVICES=0 python src/clf/codet5_mia.py --use_cal \
                    --feature_name codet5_mip --model_type selfattn --use_bdg --use_mip --use_msp \
                    --mia_type wb --run_name codet5_wb

CUDA_VISIBLE_DEVICES=0 python src/clf/codet5_mia.py --use_cal \
                    --feature_name codet5_mip --model_type selfattn --use_bdg --use_mip --use_msp \
                    --mia_type bb --run_name codet5_bb