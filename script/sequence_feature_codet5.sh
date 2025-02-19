#!/bin/bash

codet5_target_raw=output/codet5_pretrain/codet5_all_pretrain/models/final
codet5_caliberte_raw=output/codet5_pretrain/codet5_all_caliberate/models/final
codet5_shadow_raw=output/codet5_pretrain/codet5_all_shadow/models/final

codet5_target_raw_wonl=output/codet5_pretrain/codet5_all_pretrain_wonl/models/final
codet5_caliberte_raw_wonl=output/codet5_pretrain/codet5_all_caliberate_wonl/models/final
codet5_shadow_raw_wonl=output/codet5_pretrain/codet5_all_shadow_wonl/models/final

for seq in   codet5_bdg ;do
    for path in $codet5_target_raw_wonl ;do
        for data_name in wb_mem_train wb_non_train wb_mem_test wb_non_test;do
        # -m torch.distributed.launch --nproc_per_node=1
            CUDA_VISIBLE_DEVICES=1,2,3,4,5,6,7 python -m torch.distributed.launch --nproc_per_node=7 src/signal_extractor/main.py \
                                            --target-model codet5_pretrain \
                                            --learning-rate 1e-5 --data-name $data_name \
                                            --logging-steps 5 --code-len 128  \
                                            --shadow-path $path --model-name $(echo $path | awk -F'/' '{print $(NF-2)}') \
                                            --caliberate-path $codet5_caliberte_raw_wonl \
                                            --batch-size 32 --do-seq-fea --seq-fea-name $seq --output-name codet5_seqfea
        done
    done
done

for seq in   codet5_bdg ;do
    for path in $codet5_shadow_raw_wonl ;do
        for data_name in non_shadow non_utils;do
            CUDA_VISIBLE_DEVICES=1,2,3,4,5,6,7 python -m torch.distributed.launch --nproc_per_node=7 src/signal_extractor/main.py \
                                            --target-model codet5_pretrain \
                                            --learning-rate 1e-5 --data-name $data_name \
                                            --logging-steps 5 --code-len 128  \
                                            --shadow-path $path --model-name $(echo $path | awk -F'/' '{print $(NF-2)}') \
                                            --caliberate-path $codet5_caliberte_raw_wonl \
                                            --batch-size 32 --do-seq-fea --seq-fea-name $seq --output-name codet5_seqfea
        done
    done
done