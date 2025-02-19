from pathlib import Path
import time
import pandas as pd
import torch 
import numpy as np 
import warnings
import os

warnings.filterwarnings('ignore')
import pickle

import argparse

from transformers import AutoTokenizer, RobertaConfig, T5Config, RobertaModel, RobertaTokenizer, RobertaForMaskedLM, LlamaForCausalLM
from transformers import BartConfig, Seq2SeqTrainingArguments, IntervalStrategy, SchedulerType, TrainingArguments
from safetensors.torch import load_model, save_model

from signal_extractor.args import add_args

from src.common.utils import set_seed

# from utils.get_models import shadow_model_factory, target_model_factory
from src.data_utils.csn import CsnMiaData

from src.train.train_codebert.codebert_model import CodeBERTForClassification, MaskGenerator
from src.train.train_codet5.t5_model import CodeT5ForClassificationAndGeneration

from src.signal_extractor.sequence_feature_codebert import (NtimesMLMLoss, NtimesRTDLoss)
from src.signal_extractor.sequence_feature_codet5 import Codet5NtimesMSP, Codet5BDGLoss, Codet5MIPLoss, Codet5ITLoss

import torch.distributed as dist


if __name__ == "__main__":
    
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.register('type', 'bool', lambda v: v.lower() in ['yes', 'true', 't', '1', 'y'])

    add_args(parser)
    
    args = parser.parse_args()
    args.device = torch.device('cuda')
    args.output_path = f'output/{args.output_name}/{args.model_name}'
    
    if not os.path.exists(args.output_path):
        os.makedirs(args.output_path)
    
    set_seed(42)

    if args.local_rank != -1:
        torch.cuda.set_device(args.local_rank)
        torch.distributed.init_process_group(backend='nccl', world_size=torch.cuda.device_count())
            
    
    target_tokenizer = RobertaTokenizer.from_pretrained('microsoft/codebert-base')
    
    if args.do_seq_fea:
        config = RobertaConfig.from_pretrained('FacebookAI/roberta-base')
        
        print('loading data...')
        data = CsnMiaData(args, target_tokenizer).data(f'{args.data_name}')
        print('finish loading data...')
            
        
        if args.seq_fea_name == 'ntimes_mlm_loss':
            config = RobertaConfig.from_pretrained(args.shadow_path)
            try:
                shadow_model = CodeBERTForClassification.from_pretrained(args.shadow_path, config=config)
            except:
                shadow_model = CodeBERTForClassification(config)
                load_model(shadow_model, os.path.join(args.shadow_path, "model.safetensors"))
            try:
                caliberate_model = CodeBERTForClassification.from_pretrained(args.caliberate_path, config=config)
            except:
                caliberate_model = CodeBERTForClassification(config)
                load_model(caliberate_model, os.path.join(args.caliberate_path, "model.safetensors"))
            
            shadow_model.mode = 'mlm'
            caliberate_model.mode = 'mlm'
            sf = NtimesMLMLoss(args, target_tokenizer, shadow_model, data, caliberate_model)
        
        elif args.seq_fea_name == 'ntimes_rtd_loss':
            generator_config = RobertaConfig.from_pretrained('FacebookAI/roberta-base')
            generator_config.num_hidden_layers = 4
            generator = MaskGenerator.from_pretrained('FacebookAI/roberta-base', config=generator_config)
            
            codebert_config = RobertaConfig.from_pretrained('FacebookAI/codebert-base')
            codebert_config.num_labels = 2
            
            shadow_model = CodeBERTForClassification(codebert_config, generator)
            load_model(shadow_model, os.path.join(args.shadow_path, "model.safetensors"))
            shadow_model.generator = generator
            
            caliberate_model = CodeBERTForClassification(codebert_config, generator)
            load_model(caliberate_model, os.path.join(args.caliberate_path, "model.safetensors"))
            caliberate_model.generator = generator
            
            shadow_model.mode = 'rtd'
            caliberate_model.mode = 'rtd'
            sf = NtimesRTDLoss(args, target_tokenizer, shadow_model, data, caliberate_model)
            
        elif args.seq_fea_name.startswith('codet5'):
            config = T5Config.from_pretrained('Salesforce/codet5-base')
            config.num_labels = 2
            tokenizer = AutoTokenizer.from_pretrained('Salesforce/codet5-base', add_prefix_space=True)
            
            try:
                target_model = CodeT5ForClassificationAndGeneration.from_pretrained(args.shadow_path, config=config)
                caliberate_model = CodeT5ForClassificationAndGeneration.from_pretrained(args.caliberate_path, config=config)
            except:
                target_model = CodeT5ForClassificationAndGeneration(config=config)
                load_model(target_model, os.path.join(args.shadow_path, 'model.safetensors'))
                caliberate_model = CodeT5ForClassificationAndGeneration(config=config)
                load_model(caliberate_model, os.path.join(args.caliberate_path, 'model.safetensors'))
                
            if args.seq_fea_name == 'codet5_ntimes_msp':
                sf = Codet5NtimesMSP(args, tokenizer, target_model, data, caliberate_model)
            elif args.seq_fea_name == 'codet5_bdg':
                sf = Codet5BDGLoss(args, tokenizer, target_model, data, caliberate_model)
            elif args.seq_fea_name == 'codet5_mip':
                sf = Codet5MIPLoss(args, tokenizer, target_model, data, caliberate_model)
            elif args.seq_fea_name == 'codet5_it':
                sf = Codet5ITLoss(args, tokenizer, target_model, data, caliberate_model)
        
        
        feature = sf.get_feature()
        if args.local_rank == -1:
            with open(os.path.join(args.output_path, f'{args.data_name}_{args.seq_fea_name}.npy'), 'wb') as f:
                    pickle.dump(feature, f, protocol=4)
        else:
            
            with open(os.path.join(args.output_path, f'{args.data_name}_{args.seq_fea_name}_rank{args.local_rank}.npy'), 'wb') as f:
                pickle.dump(feature, f, protocol=4)
                            
            torch.distributed.barrier()
            
            if args.local_rank == 0:
                dicts = []
                
                files = Path(args.output_path)
                outputs = [str(i) for i in files.iterdir()]
                data_file = list(filter(lambda x:f'{args.data_name}_{args.seq_fea_name}' in x, outputs))
                
                for file in data_file:
                    item = np.load(file, allow_pickle=True)
                    dicts.append(item)
                    
                res = {}
                for d in dicts:
                    for key, value in d.items():
                        if key in res:
                            res[key] = np.concatenate([res[key], value])
                        else:
                            res[key] = value    
                
                with open(os.path.join(args.output_path, f'{args.data_name}_{args.seq_fea_name}.npy'), 'wb') as f:
                    pickle.dump(res, f, protocol=4)
                