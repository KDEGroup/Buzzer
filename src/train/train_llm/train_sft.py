import os
import pickle
import random
import numpy as np
import torch
from tqdm import tqdm
import transformers 
from transformers import Trainer, AutoConfig, LlamaForCausalLM
from peft import get_peft_model
import datasets 
from dataclasses import dataclass, field 
from typing import Optional, Sequence, Dict
import copy
from accelerate.utils import set_seed
from torch.utils.data import DataLoader
from src.train.train_llm.data_utils import DataCollatorForSupervisedDataset, train_tokenize_function
from transformers.trainer_pt_utils import torch_distributed_zero_first
from peft import LoraConfig, get_peft_model

from loguru import logger

@dataclass
class ModelArguments:
    model_name_or_path: Optional[str] = field(default="")
    mode: str = field(default="train_sft")
    tokenizer_path: str = field(default="")
    config_path: str = field(default="")
    
@dataclass
class DataArguments:
    data_path: str = field(default=None, metadata={"help": "Path to the training data."})
    data_name: str = field(default='mem_pretrain')

@dataclass
class TrainingArguments(transformers.TrainingArguments):
    cache_dir: Optional[str] = field(default=None)
    optim: str = field(default="adamw_torch")
    model_max_length: int = field(
        default=2048,
        metadata={"help": "Maximum sequence length. Sequences will be right padded (and possibly truncated)."},
    )
    use_lora: bool = field(default=False)
    

def safe_save_model_for_hf_trainer(trainer: transformers.Trainer, output_dir: str):
    """Collects the state dict and dump to disk."""
    state_dict = trainer.model.state_dict()
    if trainer.args.should_save:
        cpu_state_dict = {key: value.cpu() for key, value in state_dict.items()}
        del state_dict
        trainer._save(output_dir, state_dict=cpu_state_dict)  # noqa


def train():
    parser = transformers.HfArgumentParser((ModelArguments, DataArguments, TrainingArguments))
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()
    
    set_seed(42)
    logger.info(training_args.local_rank)
    
    if training_args.local_rank == 0:
        logger.info('='*100)
        logger.info(training_args)
    
    tokenizer = transformers.AutoTokenizer.from_pretrained(
        model_args.tokenizer_path,
        model_max_length=training_args.model_max_length,
        padding_side="right",
        use_fast=True,
        trust_remote_code=True
    )

    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id

    
    if training_args.local_rank == 0:
        logger.info("PAD Token:", tokenizer.pad_token, tokenizer.pad_token_id)
        logger.info("BOS Token", tokenizer.bos_token, tokenizer.bos_token_id)
        logger.info("EOS Token", tokenizer.eos_token, tokenizer.eos_token_id)
        logger.info("Load tokenizer from {} over.".format(model_args.model_name_or_path))
        
    config = AutoConfig.from_pretrained(model_args.config_path)
    
    model = transformers.AutoModelForCausalLM.from_pretrained(
        model_args.model_name_or_path,
        torch_dtype=torch.bfloat16,
        config=config,
        device_map='cuda'
    )
    
    if training_args.local_rank == 0:
        logger.info("Load model from {} over.".format(model_args.model_name_or_path))

    logger.info("Load data from {} over.".format(f'LLM_MIA/data/evol_inst_110k/{data_args.data_name}'))
        
    # with torch_distributed_zero_first(training_args.local_rank):
    train_dataset = datasets.load_from_disk(f'LLM_MIA/data/evol_inst_110k/{data_args.data_name}')
    train_dataset = train_dataset.map(
        train_tokenize_function,
        batched=True,
        batch_size=1024,
        num_proc=32,
        load_from_cache_file=False, # not args.overwrite_cache
        keep_in_memory=True,
        desc="Running Encoding",
        fn_kwargs={ "tokenizer": tokenizer }
    )

    # for testing shadow dataset size
    # if data_args.data_name == 'non_shadow' or data_args.data_name == 'non_caliberate':
    #     train_len = len(train_dataset)
    #     train_dataset = train_dataset.select(range(0, train_len // 4))
        
    if model_args.mode == 'train_sft':
        data_collator = DataCollatorForSupervisedDataset(tokenizer=tokenizer, 
                                                         args=training_args, 
                                                         pp_format=False,
                                                         pad=True)
        data_module = dict(train_dataset=train_dataset, eval_dataset=None, data_collator=data_collator)
        
        trainer = Trainer(model=model, tokenizer=tokenizer, args=training_args, **data_module)

        trainer.train()
        trainer.save_state()
        
        # model.save_pretrained(training_args.output_dir)
        # model.save_pretrained(os.path.join(training_args.output_dir, 'final'), safe_serialization=False)
        output_dir=os.path.join(training_args.output_dir, 'final')
        safe_save_model_for_hf_trainer(trainer=trainer, output_dir=output_dir)

            
    elif model_args.mode.startswith('signal_extractor'):
        device = torch.device('cuda')
        
        mia_type = model_args.mode.split('signal_extractor_')[-1]
        assert mia_type in ['shadow', 'target', 'calibrate']
        output_dir = os.path.join(training_args.output_dir, mia_type)                        
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
            
        data_collator = DataCollatorForSupervisedDataset(tokenizer=tokenizer, 
                                                         args=training_args, 
                                                         pp_format=False,
                                                         pad=False)       
         
        data_loader = DataLoader(train_dataset, batch_size=1,
                                collate_fn=data_collator)
        
        model = model.to(device)
        
        holder = []
        
        bar = tqdm(total=len(data_loader))
        for idx, batch in enumerate(data_loader):  
            batch['reduction'] = 'none'
            '''
            Should modify transformer package source code
            '''
            bs = batch['input_ids'].size(0)
            prompt_len = batch.pop('prompt_len')[0]
            
            batch['input_ids'] = batch['input_ids'].to(device)
            batch['labels'] = batch['labels'].to(device)
            batch['attention_mask'] = batch['attention_mask'].to(device)
            
            with torch.no_grad(): 
                output = model(**batch)
                signals = output.loss.tolist()
                                
                signals = signals[prompt_len:-1]
                item = signals[:2047]
                need_pad_len = 2047 - len(item)
                
                if need_pad_len == 2047:
                    continue
                
                item = item + [0] * need_pad_len
                holder.append(item)
                        
            bar.update(1)
        
        holder = np.array(holder).reshape(-1, 2047)

        with open(os.path.join(output_dir, f'{data_args.data_name}_signal.npy'), 'wb') as f:
            pickle.dump(holder, f, protocol=4)

if __name__ == "__main__":
    train()