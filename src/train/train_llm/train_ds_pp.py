
import time
import random
import warnings
from dataclasses import dataclass, field
from typing import Optional, Literal

import torch
import transformers
from transformers import Qwen2Config, LlamaConfig
import numpy as np
import deepspeed
import torch.distributed as dist
from torch.utils.data import DistributedSampler, DataLoader, RandomSampler, SequentialSampler

import os
import hydra
from omegaconf import DictConfig, OmegaConf
from llama_pipeline_model import get_model
from accelerate.utils import set_seed
import datasets
import sys 

from tqdm import tqdm

from typing import Sequence, Dict, Tuple
import copy
from deepspeed.utils import logger
from src.train.llm.data_utils import DataCollatorForSupervisedDataset, train_tokenize_function


warnings.filterwarnings("ignore")


def save_model(engine, output_dir, args, tokenizer):
    '''
    TODO: should save rng state?
    '''
    engine.save_checkpoint(output_dir)
    
    if args.local_rank not in [-1, 0]:
        dist.barrier()
        
    if args.local_rank in [-1, 0]:
        tokenizer.save_pretrained(output_dir)

        OmegaConf.save(args, os.path.join(output_dir, "config.yaml"))

        if args.local_rank == 0:
            dist.barrier()


@hydra.main(config_path=".", config_name="llama_config", version_base="1.2")
def main(args: DictConfig):
    # parser = transformers.HfArgumentParser(TrainerArguments)
    # args, = parser.parse_args_into_dataclasses()

    # setup deepspeed and other stuff
    
    deepspeed.init_distributed(dist_backend="nccl")
    
    args.world_size = torch.distributed.get_world_size()

    args.pp_num = args.pipe_parallel_size
    args.mp_num = args.model_parallel_size
    args.dp_num = args.world_size // args.pipe_parallel_size
    
    ds_config = args.deepspeed_config

    #TODO: need fix bug, default = 0
    gradient_checkpoint_interval = args.gradient_checkpoint_interval 

    set_seed(args.seed)   
    deepspeed.runtime.utils.set_random_seed(args.seed)
    

    tokenizer = transformers.AutoTokenizer.from_pretrained(
        args.tokenizer.tokenizer_path,
        model_max_length=args.model_max_length,
        padding_side="right",
        use_fast=True,
        trust_remote_code=True
    )
    
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id
    
    if args.local_rank not in [-1, 0]:
        dist.barrier()
    
    saved_path = f'LLM_MIA/data/evol_inst_110k/{args.data_name}_mapped'
    if args.local_rank in [-1, 0]:
        train_dataset = datasets.load_from_disk(f'LLM_MIA/data/evol_inst_110k/{args.data_name}')
        train_dataset = train_dataset.map(
                train_tokenize_function,
                batched=True,
                batch_size=1024,
                num_proc=32,
                load_from_cache_file=False, # not args.overwrite_cache
                desc="Running Encoding",
                fn_kwargs={ "tokenizer": tokenizer }
            )
        if args.run_test:
            train_dataset = train_dataset.select(range(0, 64))
        train_dataset.save_to_disk(saved_path)
    else:
        train_dataset = datasets.load_from_disk(saved_path)
        
    if args.local_rank == 0:
        dist.barrier()
        
    # pipeline model
    # partition_method="type:ParallelTransformerLayerPipe"
    model_config = LlamaConfig.from_dict(args.model_arguments)
    model = get_model(model_config, args, partition_method="uniform")
    
    data_collator = DataCollatorForSupervisedDataset(tokenizer=tokenizer, args=args, pp_format=True,
                                                     pad=True)
    ds_config = OmegaConf.to_container(ds_config, resolve=True)
    
    
    engine, _, data_loader, _ = deepspeed.initialize(
        config=ds_config,
        # training_data=train_dataset,
        # collate_fn=data_collator,
        model=model,
        model_parameters=[p for p in model.parameters() if p.requires_grad],
    )
    
    if args.model_init_path != "":
        logger.info(f'loading init model weight from: {args.model_init_path}')
        engine.load_checkpoint(args.model_init_path,
                               load_module_only=True,
                               )
    
    resume_step = -1
    if args.resume_from_checkpoint is not None:
        ckpt_path = args.resume_from_checkpoint
        resume_step = int(ckpt_path.split('-')[-1])
        
        engine.load_checkpoint(ckpt_path)
        tokenizer = transformers.AutoTokenizer.from_pretrained(
            ckpt_path,
            model_max_length=args.model_max_length,
            padding_side="right",
            use_fast=False,
        )
        logger.info(f'resume from checkpoint: {args.resume_from_checkpoint} resume step:{resume_step}')

    logger.info(f'dataset number:{len(train_dataset)}')
    logger.info(f'data parallel number: {args.dp_num}')
    
    # if args.dp_num > 1:
    '''
        若数据并行组大于1，则要通过distributedSampler将数据放大对应的组上
        
        TODO: deepspeed中也实现了数据加载的方法，初始化engine时候可以传入dataset和data_collator
    '''
    
    # args.dp_num = args.world_size // args.pipe_parallel_size
    # dp_id = model._grid.get_data_parallel_id()        
    # print(engine.dp_world_size, dp_id)
    data_sampler = DistributedSampler(train_dataset, num_replicas=engine.dp_world_size,
                    rank=model._grid.get_data_parallel_id())
    data_loader = DataLoader(dataset=train_dataset,
                                    sampler=data_sampler,
                                    batch_size=args.per_gpu_train_batch_size,
                                    collate_fn=data_collator,
                                    num_workers=args.num_workers,
                                    pin_memory=False,
                                    prefetch_factor=args.prefetch_factor,
                                    drop_last=False,
                                    )
    data_loader = iter(deepspeed.utils.RepeatingLoader(data_loader))
    
    
    start = time.time()
    epoch_update_steps = len(train_dataset) // args.per_gpu_train_batch_size // (args.world_size // args.pipe_parallel_size)
    epoch_update_steps = epoch_update_steps // args.gradient_accumulation_steps
    bar = tqdm(range(1, epoch_update_steps + 1), disable=args.local_rank not in [-1, 0], dynamic_ncols=True)
    
    global_step = 0
    for step in bar:
        if global_step < resume_step:
            for _ in range(args.gradient_accumulation_steps):
                next(data_loader)
            global_step += 1
            continue

        loss = engine.train_batch(data_iter=data_loader)
        global_step += 1
        
        if args.local_rank == 0:
            if step % args.log_steps == 0:
                now = time.time()
                avg_time = (now-start) / args.log_steps
                # logger.info(f"Step={step:>6}, loss={loss.item():.4f}, {avg_time:.2f} it/s")
                bar.set_description(f"Step={step:>2}, loss={loss.item():.2f}, {avg_time:.2f} it/s")
                start = now

        if step % args.eval_steps == 0:
            # TODO
            pass

        if args.save_steps > 0 and global_step % args.save_steps == 0:
            output_dir = f'{args.output_dir}/checkpoint-{global_step}'
            # engine.save_checkpoint(args.output_dir)
            logger.info(f"Saving at global step: {global_step} to {output_dir}")
            save_model(engine, output_dir, args, tokenizer)
    
    output_dir = f'{args.output_dir}/final'
    save_model(engine, output_dir, args, tokenizer)


if __name__ == "__main__":
    
    hydra_formatted_args = []
    # convert the cli params added by torch.distributed.launch into Hydra format
    for arg in sys.argv:
        if arg.startswith("--"):
            hydra_formatted_args.append(arg[len("--"):])
        else:
            hydra_formatted_args.append(arg)
    sys.argv = hydra_formatted_args
    main()
