#!/bin/bash


codebert_target_raw=output/codebert_pretrain/codebert_mlmrtd_all_pre_train_50epoch/models/rtd
codebert_caliberte_raw=output/codebert_pretrain/codebert_mlmrtd_all_caliberate_50epoch/models/rtd
codebert_shadow_raw=output/codebert_pretrain/codebert_mlmrtd_all_shadow_50epoch/models/rtd

for seq in ntimes_mlm_loss ntimes_rtd_loss ;do
    for path in $codebert_target_raw_wonl ;do
        for output_name in codebert_seqfea;do
            for data_name in wb_mem_train wb_non_train wb_mem_test wb_non_test  ;do
                CUDA_VISIBLE_DEVICES=2,3,4,5,6,7 python -m torch.distributed.launch --nproc_per_node=6 src/signal_extractor/main.py \
                                                --target-model codebert_pretrain \
                                                --learning-rate 1e-5 --data-name $data_name \
                                                --logging-steps 5 --code-len 128  \
                                                --shadow-path $path --model-name codebert_mlmrtd_all_pre_train_50epoch_wonl \
                                                --caliberate-path $codebert_caliberte_raw_wonl \
                                                --batch-size 32 --do-seq-fea --seq-fea-name $seq --output-name $output_name
            done
        done
    done
done
