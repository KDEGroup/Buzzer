#!/bin/bash

for dt in mem_pretrain non_shadow non_calibrate;do
    deepspeed --include localhost:1,2,3,4,5,6 src/train/train_llm/train_ds_pp.py pipe_parallel_size=6 \
        gradient_accumulation_steps=32 model_max_length=2048 per_gpu_train_batch_size=2 \
        data_name=$dt model_init_path=cache/LLM_Weight/codellama-7b-base-init-ckpt \
        run_name=code_llama_mia_$dt learning_rate=5e-5 save_steps=-1
done

deepseek_path="/data/zs/LLM_Weight/deepseek-coder-1.3b-base"
CUDA_VISIBLE_DEVICES=1,2 torchrun --nnodes 1 --nproc_per_node 2 --master_port 29502 src/train/train_llm/train_sft.py \
            --model_name_or_path $deepseek_path \
            --tokenizer_path $deepseek_path \
            --config_path $deepseek_path \
            --mode train_sft \
            --data_name $dt \
            --output_dir $OUTPUT_PATH \
            --num_train_epochs $epoch_num \
            --model_max_length 2048 \
            --per_device_train_batch_size 1 \
            --per_device_eval_batch_size 1 \
            --gradient_accumulation_steps 64 \
            --evaluation_strategy "no" \
            --save_strategy "steps" \
            --save_steps 50 \
            --save_total_limit 3 \
            --learning_rate 5e-5 \
            --adam_beta2 0.95 \
            --warmup_steps 0 \
            --logging_steps 1 \
            --lr_scheduler_type "cosine" \
            --gradient_checkpointing True \
            --report_to "tensorboard" \
            --ddp_find_unused_parameters False \
            --bf16 True