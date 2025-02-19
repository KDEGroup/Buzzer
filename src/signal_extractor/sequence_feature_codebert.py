import copy
# from data.data_utils import *
# from data.asts.ast_parser import *

from typing import List, Dict
from loguru import logger

import torch
from tqdm import tqdm

from pathlib import Path
import re
import numpy as np
import pandas as pd
from transformers import RobertaModel, AutoTokenizer, pipeline, RobertaForMaskedLM, RobertaTokenizer, RobertaConfig
from more_itertools import chunked
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP

from torch.utils.data import Dataset, DataLoader, TensorDataset, SequentialSampler

import random
from heapq import nlargest
import torch.nn.functional as F


class SequenceFeature:
    def __init__(self, args, tokenizer, encoder, data):
        self.args = args 
        self.tokenizer = tokenizer
        self.encoder = encoder
        
        self.data = []
        for line in data:
            ct = line['code_tokens']
            if isinstance(ct, str):
                ct = ct.split(' ')
            self.data.append(ct)
        
        idx = torch.arange(len(self.data), dtype=torch.int32).view(-1, 1)
        
        self.inputs = tokenizer(self.data, padding='max_length', 
                        truncation=True, max_length=self.args.code_len, is_split_into_words=True, return_tensors='pt')
        
        self.tensor_dataset = TensorDataset(self.inputs['input_ids'], self.inputs['attention_mask'], idx)
        
        if args.local_rank != -1:
            self.sampler = torch.utils.data.distributed.DistributedSampler(self.tensor_dataset, shuffle=False)
            self.dataloader = DataLoader(self.tensor_dataset, batch_size=self.args.batch_size, sampler=self.sampler)
            self.device = torch.device(f'cuda:{self.args.local_rank}')
            self.encoder = encoder.to(self.device)
            
        else:
            self.sampler = SequentialSampler(self.tensor_dataset)
            self.dataloader = DataLoader(self.tensor_dataset, batch_size=self.args.batch_size, sampler=self.sampler)
            self.device = torch.device(f'cuda')
            self.encoder = encoder.to(self.device)
        
    def get_feature_score(self):
        raise NotImplementedError
        
    def get_feature(self):
        res = self.get_feature_score()
        return res
    
    
class NtimesMLMLoss(SequenceFeature):
    def __init__(self, args, tokenizer, encoder, data, caliberate_model, feature='last_hidden_state'):
        super().__init__(args, tokenizer, encoder, data)
        self.feature = feature
        self.encoder.mode = 'mlm'
        self.caliberate_model = caliberate_model
        self.caliberate_model = self.caliberate_model.to(self.device)
        self.caliberate_model.mode = 'mlm'
        self.n = 100
    
    def mask_tokens(self, inputs, attention_mask, mask_token_index, special_token_indices, mlm_probability=0.15, ignore_index=-100):
        device = inputs.device
        labels = inputs.clone()

        probability_matrix = torch.full(labels.shape, mlm_probability, device=device)
        special_tokens_mask = torch.full(inputs.shape, False, dtype=torch.bool, device=device)
        
        for sp_id in special_token_indices:
            special_tokens_mask = special_tokens_mask | (inputs==sp_id)
        probability_matrix.masked_fill_(special_tokens_mask, value=0.0)
        probability_matrix.masked_fill_(~attention_mask.bool(), value=0.0)
        
        mlm_mask = torch.bernoulli(probability_matrix).bool()
        labels[~mlm_mask] = ignore_index  # We only compute loss on mlm applied tokens  

        inputs[mlm_mask] = mask_token_index  
        return inputs, labels, mlm_mask
    
    def get_feature_score(self):
        special_indices = [self.tokenizer.bos_token_id, self.tokenizer.eos_token_id]
        attention_masks = []
        input_ids_all = []
        
        with torch.no_grad():
            data_num = len(self.data) if self.args.local_rank == -1 else len(self.sampler)
            print(data_num)
            total_len = len(self.dataloader)
            bar = tqdm(total=total_len)
            
            target_feature_score = np.zeros((data_num, self.n))
            caliberate_feature_score = np.zeros((data_num, self.n))
            data_all_idx = []
            
            for batch_idx, batch in enumerate(self.dataloader):
                input_ids, attention_mask_raw, data_idx = batch
                
                bs, seq_len = input_ids.size(0), input_ids.size(1)
                
                for idx in range(self.n):
                    tmp_inputs = input_ids.clone()
                    
                    inputs, labels, mlm_mask = self.mask_tokens(tmp_inputs, attention_mask_raw, self.tokenizer.mask_token_id, special_indices)
                    inputs = inputs.to(self.device)
                    labels = labels.to(self.device)
                    attention_mask = attention_mask_raw.to(self.device)
                    
                    target_output = self.encoder(input_ids=inputs, attention_mask=attention_mask, labels=labels, reduction='none')
                    cal_output = self.caliberate_model(input_ids=inputs, attention_mask=attention_mask, labels=labels, reduction='none')
                    
                    tar_loss = target_output.loss.view(bs, -1)
                    cal_loss = cal_output.loss.view(bs, -1)
                    mlm_num = mlm_mask.sum(dim=1).cpu()
                    
                    tar_loss = tar_loss.sum(dim=1)
                    cal_loss = cal_loss.sum(dim=1)
                    
                    # Repeat the mlm n times, each time we calculate the mean loss
                    target_feature_score[self.args.batch_size * batch_idx : \
                            self.args.batch_size * batch_idx + bs, idx] = tar_loss.cpu() / mlm_num
                    caliberate_feature_score[self.args.batch_size * batch_idx : \
                            self.args.batch_size * batch_idx + bs, idx] = cal_loss.cpu() / mlm_num
                    
                bar.update(1)    
                data_all_idx.append(data_idx.cpu())  
                attention_masks.append(attention_mask.cpu())
                input_ids_all.append(input_ids.cpu())
            
        attention_masks = np.vstack(attention_masks)
        input_ids_all = np.vstack(input_ids_all)
        data_all_idx = np.vstack(data_all_idx)
        res = {
            'target_feature': target_feature_score,
            'caliberate_feature': caliberate_feature_score,
            'attention_mask': attention_masks,
            'input_ids_all': input_ids_all,
            'data_all_idx': data_all_idx
        }
        return res


class NtimesRTDLoss(SequenceFeature):
    def __init__(self, args, tokenizer, encoder, data, caliberate_model, feature='last_hidden_state'):
        super().__init__(args, tokenizer, encoder, data)
        self.encoder.mode = 'rtd'
        
        mg_config = RobertaConfig.from_pretrained('FacebookAI/roberta-base')
        self.mask_generator = RobertaForMaskedLM.from_pretrained('FacebookAI/roberta-base', config=mg_config)
        self.mask_generator = self.mask_generator.to(self.device)
        self.caliberate_model = caliberate_model
        self.caliberate_model = self.caliberate_model.to(self.device)
        self.caliberate_model.mode = 'mlm'
        self.feature = feature
        self.n = 100
    
    def mask_tokens(self, inputs, attention_mask, mask_token_index, special_token_indices, mlm_probability=0.15, ignore_index=-100):
        device = inputs.device
        labels = inputs.clone()

        probability_matrix = torch.full(labels.shape, mlm_probability, device=device)
        special_tokens_mask = torch.full(inputs.shape, False, dtype=torch.bool, device=device)
        
        for sp_id in special_token_indices:
            special_tokens_mask = special_tokens_mask | (inputs==sp_id)
        probability_matrix.masked_fill_(special_tokens_mask, value=0.0)
        probability_matrix.masked_fill_(~attention_mask.bool(), value=0.0)
        
        mlm_mask = torch.bernoulli(probability_matrix).bool()
        labels[~mlm_mask] = ignore_index  # We only compute loss on mlm applied tokens  

        inputs[mlm_mask] = mask_token_index  
        return inputs, labels, mlm_mask
    
    def get_loss(self, generated, attention_mask, is_replaced, model):
        
        outputs = model.encoder(
            generated,
            attention_mask=attention_mask,
        )
        sequence_output = outputs[0]
        
        prediction_scores = model.discrimiator(sequence_output)
        
        loss_fct = torch.nn.CrossEntropyLoss(reduction='none')
        is_replaced = is_replaced.to(torch.int64)
        is_replaced[~attention_mask.bool()] = -100
        is_replaced[generated == 0] = -100
        is_replaced[generated == 2] = -100
                
        loss_item_num = is_replaced.ne(-100).sum(dim=1)
        
        loss = loss_fct(prediction_scores.view(-1, prediction_scores.size(-1)), is_replaced.view(-1), ) 
        bs = generated.size(0)
        loss = loss.view(bs, -1)
        loss_ = loss.sum(dim=1) / loss_item_num
        return loss_
    
    def get_feature_score(self):
        special_indices = [self.tokenizer.bos_token_id, self.tokenizer.eos_token_id]
        attention_masks = []
        input_ids_all = []
        data_all_idx = []
        
        with torch.no_grad():
            data_num = len(self.data) if self.args.local_rank == -1 else len(self.sampler)
            print(data_num)
            total_len = len(self.dataloader)
            bar = tqdm(total=total_len)
            target_feature_score = np.zeros((data_num, self.n))
            caliberate_feature_score = np.zeros((data_num, self.n))
            
            for batch_idx, batch in enumerate(self.dataloader):
                input_ids, attention_mask_raw, data_idx = batch
                
                bs, seq_len = input_ids.size(0), input_ids.size(1)
                
                with torch.no_grad():   
                    for idx in range(self.n):
                        tmp_inputs = input_ids.clone()
                        inputs, labels, mlm_mask = self.mask_tokens(tmp_inputs, attention_mask_raw, self.tokenizer.mask_token_id, special_indices)
                        
                        inputs = inputs.to(self.device)
                        labels = labels.to(self.device)
                        attention_mask = attention_mask_raw.to(self.device)
                        mlm_mask = mlm_mask.to(self.device)
                        bs = inputs.size(0)
                        gen_logits = self.mask_generator(input_ids=inputs, attention_mask=attention_mask).logits
                        mlm_gen_logits = gen_logits[mlm_mask, :]
                        
                        pred_toks = self.encoder.sample(mlm_gen_logits) # ( #mlm_positions, )
                        generated = inputs.clone()
                        generated[mlm_mask] = pred_toks 
                        is_replaced = mlm_mask.clone() 
                        is_replaced[mlm_mask] = (pred_toks != labels[mlm_mask]) # (B,L)
                        
                        tar_loss_ = self.get_loss(generated, attention_mask, is_replaced, self.encoder)
                        cal_loss_ = self.get_loss(generated, attention_mask, is_replaced, self.caliberate_model)
                        
                        target_feature_score[self.args.batch_size * batch_idx : self.args.batch_size * batch_idx + bs, idx] = tar_loss_.cpu()
                        caliberate_feature_score[self.args.batch_size * batch_idx : self.args.batch_size * batch_idx + bs, idx] = cal_loss_.cpu()
                
                data_all_idx.append(data_idx.cpu())
                
                bar.update(1)      
                attention_masks.append(attention_mask_raw.cpu())
                input_ids_all.append(input_ids.cpu())
        
        attention_masks = np.vstack(attention_masks)
        input_ids_all = np.vstack(input_ids_all)
        data_all_idx = np.vstack(data_all_idx)
        res = {
            'target_feature': target_feature_score,
            'caliberate_feature': caliberate_feature_score,
            'attention_mask': attention_masks,
            'input_ids_all': input_ids_all,
            'data_all_idx': data_all_idx,
        }
        
        return res
