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


class Codet5Feature:
    def __init__(self, args, tokenizer, target_model, data):
        self.args = args 
        self.tokenizer = tokenizer
        self.target_model = target_model
        
        self.data = data
        
        idx = torch.arange(len(self.data), dtype=torch.int32).view(-1, 1)        
        
        if args.local_rank != -1:
            self.sampler = torch.utils.data.distributed.DistributedSampler(self.data, shuffle=False)
            self.device = torch.device(f'cuda:{self.args.local_rank}')
            self.target_model = target_model.to(self.device)
            
        else:
            self.sampler = SequentialSampler(self.data)
            self.device = torch.device(f'cuda')
            self.target_model = target_model.to(self.device)
        
    def get_feature_score(self):
        raise NotImplementedError
        
    def get_feature(self):
        res = self.get_feature_score()
            
        return res


class Codet5NtimesMSP(Codet5Feature):
    def __init__(self, args, tokenizer, target_model, data, caliberate_model, feature=None):
        super().__init__(args, tokenizer, target_model, data)
        self.target_model.mode = 'generation'
        
        self.caliberate_model = caliberate_model
        self.caliberate_model = self.caliberate_model.to(self.device)
        self.caliberate_model.mode = 'generation'
        self.feature = feature
        self.n = 100

    def mask_strategy_2(self, source, mask_num, mask_start_id=0):
        mask = set()
        sent_length = len(source)
        spans = []
        cnt = 0
        while cnt < mask_num:
            cur_length = np.random.randint(1, 6)
            
            choice_from = sent_length - 1 - cur_length
            if choice_from <= 0:
                continue
            anchor = np.random.choice(choice_from)
            if anchor in mask:
                continue
            
            left = anchor
            
            right = left + cur_length - 1
            
            spans.append([left, right])
            for i in range(left, right + 1):
                mask.add(i)
            cnt += 1

        spans.sort(key = lambda x:int(x[0]))
        
        maskid2label = []
        label = []
        output = []
            
        cur_span_idx, idx = 0, 0
        span_len = len(spans)
        while idx < sent_length:
            if cur_span_idx >= span_len:
                output.append(source[idx])
                idx += 1
            else:
                l, r = spans[cur_span_idx][0], spans[cur_span_idx][1]
                if idx >= l and idx <= r:
                    mask_token = f'<extra_id_{cur_span_idx + mask_start_id}>'
                    label += [mask_token] + source[l:r+1]
                    idx = spans[cur_span_idx][1] + 1
                    cur_span_idx += 1
                    output.append(mask_token)
                    maskid2label.append({
                        mask_token: source[l:r+1]
                    })
                else:
                    output.append(source[idx])
                    idx += 1
            
        return output, label, maskid2label, spans
    
    def compute_loss(self, model, inputs):
        labels = inputs["labels"]
        # breakpoint()
        outputs = model(**inputs)
        # breakpoint()
        logits = outputs.get("logits")
        
        shift_logits = logits[:, :-1, :].contiguous()
        shift_labels = labels[:, 1:].contiguous()
        loss_fct = torch.nn.CrossEntropyLoss(reduction='none')
                
        loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
        bs = logits.size(0)
        loss = loss.view(bs, -1)
        loss = loss.sum(dim=1) / loss.size(1)
        # breakpoint()
        return loss
    
    def get_feature_score(self):
        data_num = len(self.data) if self.args.local_rank == -1 else len(self.sampler)
        print(data_num)
        
        attention_masks = []
        input_ids_all = []
        cur_gpu_idxs = [i for i in self.sampler]
        data_all_idx = np.array(cur_gpu_idxs).reshape(-1, 1)
        
        target_feature_score = np.zeros((data_num, self.n))
        caliberate_feature_score = np.zeros((data_num, self.n))
        
        batch_size = self.args.batch_size
        with torch.no_grad():
            
            total_len = data_num // self.args.batch_size 
            if data_num % self.args.batch_size != 0:
                total_len += 1
            bar = tqdm(total=total_len)
            
            for batch_start in range(0, data_num, batch_size):
                for i in range(self.n):
                    cur_batch_code = []
                    cur_batch_label = []
                    
                    for cur_idx in cur_gpu_idxs[batch_start : batch_start + batch_size]:
                        # generate mask span
                        cur_item = self.data[cur_idx]
                        cur_code = cur_item['code_tokens']
                        cur_nl = cur_item['docstring']
                        if isinstance(cur_code, str):
                            cur_code = cur_code.strip().split(' ')
                        if isinstance(cur_nl, str):
                            cur_nl = cur_nl.strip().split(' ')    
                            
                        code_length = len(cur_code) 
                        nl_length = len(cur_nl)
                        
                        code_mask_num = int(code_length * 0.15 / 3) 
                        if code_mask_num == 0: code_mask_num = 1
                        nl_mask_num = int(nl_length * 0.15 / 3) 
                        if nl_mask_num == 0: nl_mask_num = 1
                        
                        code_mask_start_id = 0
                        
                        code, code_label, _, _ = self.mask_strategy_2(cur_code, code_mask_num, code_mask_start_id)
                    
                        cur_batch_code.append(code)
                        cur_batch_label.append(code_label)

                    
                    cur_bs = len(cur_batch_code)
                    encoder_input = self.tokenizer(text=cur_batch_code,
                                    is_split_into_words=True, 
                                    add_special_tokens=True,
                                    padding='max_length', 
                                    truncation=True, max_length=512)        
                    decoder_input = self.tokenizer(text=cur_batch_label,
                                    is_split_into_words=True, 
                                    add_special_tokens=True,
                                    padding='max_length', 
                                    truncation=True, max_length=256)
                    
                    model_inputs = {}
                    model_inputs['input_ids'] = torch.tensor(encoder_input['input_ids']).to(self.device)
                    model_inputs['attention_mask'] = torch.tensor(encoder_input['attention_mask']).to(self.device)
                    model_inputs['decoder_input_ids'] = torch.tensor(decoder_input['input_ids']).to(self.device)
                    model_inputs['decoder_attention_mask'] = torch.tensor(decoder_input['attention_mask']).to(self.device)
                    model_inputs['labels'] = torch.tensor(decoder_input['input_ids']).to(self.device)
                    model_inputs['labels'][model_inputs['labels'] == self.tokenizer.pad_token_id] = -100
                    model_inputs['return_dict'] = True
                    
                    # breakpoint()
                    tar_loss = self.compute_loss(self.target_model, model_inputs)
                    cal_loss = self.compute_loss(self.caliberate_model, model_inputs)
                    
                    # print(batch_idx, self.args.batch_size * batch_idx + cur_bs, i)
                    target_feature_score[batch_start : batch_start + cur_bs, i] = tar_loss.cpu()
                    caliberate_feature_score[batch_start : batch_start + cur_bs, i] = cal_loss.cpu()

                input_ids_all.append(model_inputs['decoder_input_ids'][:, 1:].cpu())
                attention_masks.append(model_inputs['decoder_attention_mask'][:, 1:].cpu())
                bar.update(1)    
        # holder.sort(key=lambda x:len(x['code']))
        # breakpoint()
        attention_masks = np.vstack(attention_masks)
        input_ids_all = np.vstack(input_ids_all)
        
        res = {
            'target_feature': target_feature_score,
            'caliberate_feature': caliberate_feature_score,
            'attention_mask': attention_masks, 
            'input_ids_all': input_ids_all,
            'data_all_idx': data_all_idx,
        }
        return res
    
    def get_feature(self):
        res = self.get_feature_score()
            
        return res
    

class Codet5MIPLoss(Codet5Feature):
    def __init__(self, args, tokenizer, target_model, data, caliberate_model, feature=None):
        super().__init__(args, tokenizer, target_model, data)
        self.target_model.mode = 'generation'
        
        self.caliberate_model = caliberate_model
        self.caliberate_model = self.caliberate_model.to(self.device)
        self.caliberate_model.mode = 'generation'
        self.feature = feature
        self.n = 1
        '''
        这个东西是确定的，因此只执行一次
        '''

    def format_input(self, code_tokenized, identifiers):
        identifier_with_idx = []
        i = 0
        st = set()
        for idx, word in enumerate(code_tokenized):
            if word in identifiers and word not in st:
                st.add(word)
                identifier_with_idx.append((word, i))
                i += 1
        identifier_with_idx.sort(key=lambda x: x[1])
        
        labels = ""
        for i, item in enumerate(identifier_with_idx):
            word = item[0]
            idx = item[1]
            identifier_token = f'<extra_id_{idx}>'
            for tmp_idx, tmp_word in enumerate(code_tokenized):
                if tmp_word == word:
                    code_tokenized[tmp_idx] = identifier_token
            labels += identifier_token + ' ' + word + ' '
        labels = labels.strip()      

        inputs = code_tokenized
        labels = labels.split(' ')
        return inputs, labels

    
    def compute_loss(self, model, inputs):
        labels = inputs["labels"]
        # breakpoint()
        outputs = model(**inputs)
        # breakpoint()
        logits = outputs.get("logits")
        
        shift_logits = logits[:, :-1, :].contiguous()
        shift_labels = labels[:, 1:].contiguous()
        loss_fct = torch.nn.CrossEntropyLoss(reduction='none')
                
        loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
        bs = logits.size(0)
        loss = loss.view(bs, -1)
        # loss = loss.sum(dim=1) / loss.size(1)
        # breakpoint()
        return loss
    
    def get_feature_score(self):
        data_num = len(self.data) if self.args.local_rank == -1 else len(self.sampler)
        print(data_num)
        
        attention_masks = []
        input_ids_all = []
        data_all_idx = []
        target_feature_score = np.zeros((data_num, 255))
        caliberate_feature_score = np.zeros((data_num, 255))
        
        batch_size = self.args.batch_size
        with torch.no_grad():
            cur_gpu_idxs = [i for i in self.sampler]
            data_all_idx = np.array(cur_gpu_idxs).reshape(-1, 1)
            total_len = data_num // self.args.batch_size 
            if data_num % self.args.batch_size != 0:
                total_len += 1
            bar = tqdm(total=total_len)
            
            for batch_idx in range(0, data_num, batch_size):
                
                cur_batch_code = []
                cur_batch_label = []
                
                for cur_idx in cur_gpu_idxs[batch_idx : batch_idx + batch_size]:
                    cur_item = self.data[cur_idx]
                    cur_code = cur_item['code_tokens']
                    cur_idf = cur_item['identifiers']
                    cur_nl = cur_item['docstring']
                    
                    if isinstance(cur_code, str):
                        cur_code = cur_code.strip().split(' ')
                    if isinstance(cur_nl, str):
                        cur_nl = cur_nl.strip().split(' ')    
                        
                    code_length = len(cur_code) 
                    nl_length = len(cur_nl)
                    
                    code, code_label = self.format_input(cur_code, cur_idf)
                    
                    cur_batch_code.append(code)
                    cur_batch_label.append(code_label)
                
                cur_bs = len(cur_batch_code)
                encoder_input = self.tokenizer(text=cur_batch_code,
                                is_split_into_words=True, 
                                add_special_tokens=True,
                                padding='max_length', 
                                truncation=True, max_length=512)
    
                decoder_input = self.tokenizer(text=cur_batch_label,
                                is_split_into_words=True, 
                                add_special_tokens=True,
                                padding='max_length', 
                                truncation=True, max_length=256)
                model_inputs = {}
                model_inputs['input_ids'] = torch.tensor(encoder_input['input_ids']).to(self.device)
                model_inputs['attention_mask'] = torch.tensor(encoder_input['attention_mask']).to(self.device)
                model_inputs['decoder_input_ids'] = torch.tensor(decoder_input['input_ids']).to(self.device)
                model_inputs['decoder_attention_mask'] = torch.tensor(decoder_input['attention_mask']).to(self.device)
                model_inputs['labels'] = torch.tensor(decoder_input['input_ids']).to(self.device)
                model_inputs['labels'][model_inputs['labels'] == self.tokenizer.pad_token_id] = -100
                model_inputs['return_dict'] = True
                # breakpoint()
                tar_loss = self.compute_loss(self.target_model, model_inputs)
                cal_loss = self.compute_loss(self.caliberate_model, model_inputs)
                
                # print(batch_idx, self.args.batch_size * batch_idx + cur_bs, i)
                target_feature_score[batch_idx : batch_idx + cur_bs, :] = tar_loss.cpu()
                caliberate_feature_score[batch_idx : batch_idx + cur_bs, :] = cal_loss.cpu()
                
                input_ids_all.append(model_inputs['decoder_input_ids'][:, 1:].cpu())
                attention_masks.append(model_inputs['decoder_attention_mask'][:, 1:].cpu())
                bar.update(1)

        attention_masks = np.vstack(attention_masks)
        input_ids_all = np.vstack(input_ids_all)
        
        res = {
            'target_feature': target_feature_score,
            'caliberate_feature': caliberate_feature_score,
            'attention_mask': attention_masks, 
            'input_ids_all': input_ids_all,
            'data_all_idx': data_all_idx,
        }
        return res
    
    def get_feature(self):
        res = self.get_feature_score()
            
        return res


class Codet5BDGLoss(Codet5Feature):
    def __init__(self, args, tokenizer, target_model, data, caliberate_model, feature=None):
        super().__init__(args, tokenizer, target_model, data)
        self.target_model.mode = 'generation'
        
        self.caliberate_model = caliberate_model
        self.caliberate_model = self.caliberate_model.to(self.device)
        self.caliberate_model.mode = 'generation'
        self.feature = feature
    
    def compute_loss(self, model, inputs):
        labels = inputs["labels"]
        # breakpoint()
        outputs = model(**inputs)
        # breakpoint()
        logits = outputs.get("logits")
        
        shift_logits = logits[:, :-1, :].contiguous()
        shift_labels = labels[:, 1:].contiguous()
        loss_fct = torch.nn.CrossEntropyLoss(reduction='none')
                
        loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
        bs = logits.size(0)
        loss = loss.view(bs, -1)
        # loss = loss.sum(dim=1) / loss.size(1)
        # breakpoint()
        return loss
    
    def get_feature_score(self):
        data_num = len(self.data) if self.args.local_rank == -1 else len(self.sampler)
        print(data_num)
        
        attention_masks = []
        input_ids_all = []
        data_all_idx = []
        target_feature_score = np.zeros((data_num, 255))
        caliberate_feature_score = np.zeros((data_num, 255))
        
        batch_size = self.args.batch_size
        with torch.no_grad():
            cur_gpu_idxs = [i for i in self.sampler]
            data_all_idx = np.array(cur_gpu_idxs).reshape(-1, 1)
            total_len = data_num // self.args.batch_size 
            if data_num % self.args.batch_size != 0:
                total_len += 1
            bar = tqdm(total=total_len)
            
            for batch_idx in range(0, data_num, batch_size):
                
                cur_batch_code = []
                cur_batch_label = []
                
                for cur_idx in cur_gpu_idxs[batch_idx : batch_idx + batch_size]:
                    cur_item = self.data[cur_idx]
                    cur_code = cur_item['code_tokens']
                    cur_nl = cur_item['docstring']
                    if isinstance(cur_code, str):
                        cur_code = cur_code.strip().split(' ')
                    if isinstance(cur_nl, str):
                        cur_nl = cur_nl.strip().split(' ')    
                        
                    code_length = len(cur_code) 
                    nl_length = len(cur_nl)
                    
                    cur_batch_code.append(cur_code)
                    cur_batch_label.append(cur_nl)
                
                cur_bs = len(cur_batch_code)
                encoder_input = self.tokenizer(text=cur_batch_code,
                                is_split_into_words=True, 
                                add_special_tokens=True,
                                padding='max_length', 
                                truncation=True, max_length=512)
    
                decoder_input = self.tokenizer(text=cur_batch_label,
                                is_split_into_words=True, 
                                add_special_tokens=True,
                                padding='max_length', 
                                truncation=True, max_length=256)
                
                model_inputs = {}
                model_inputs['input_ids'] = torch.tensor(encoder_input['input_ids']).to(self.device)
                model_inputs['attention_mask'] = torch.tensor(encoder_input['attention_mask']).to(self.device)
                model_inputs['decoder_input_ids'] = torch.tensor(decoder_input['input_ids']).to(self.device)
                model_inputs['decoder_attention_mask'] = torch.tensor(decoder_input['attention_mask']).to(self.device)
                model_inputs['labels'] = torch.tensor(decoder_input['input_ids']).to(self.device)
                model_inputs['labels'][model_inputs['labels'] == self.tokenizer.pad_token_id] = -100
                model_inputs['return_dict'] = True
                # breakpoint()
                tar_loss = self.compute_loss(self.target_model, model_inputs)
                cal_loss = self.compute_loss(self.caliberate_model, model_inputs)
                
                # print(batch_idx, self.args.batch_size * batch_idx + cur_bs, i)
                target_feature_score[batch_idx : batch_idx + cur_bs, :] = tar_loss.cpu()
                caliberate_feature_score[batch_idx : batch_idx + cur_bs, :] = cal_loss.cpu()
                
                # breakpoint()
                input_ids_all.append(model_inputs['decoder_input_ids'][:, 1:].cpu())
                attention_masks.append(model_inputs['decoder_attention_mask'][:, 1:].cpu())
                bar.update(1)    
        # holder.sort(key=lambda x:len(x['code']))
        
        attention_masks = np.vstack(attention_masks)
        input_ids_all = np.vstack(input_ids_all)
        
        res = {
            'target_feature': target_feature_score,
            'caliberate_feature': caliberate_feature_score,
            'attention_mask': attention_masks, 
            'input_ids_all': input_ids_all,
            'data_all_idx': data_all_idx,
        }
        return res
    
    def get_feature(self):
        res = self.get_feature_score()
        return res


class Codet5ITLoss(Codet5Feature):
    def __init__(self, args, tokenizer, target_model, data, caliberate_model, feature=None):
        super().__init__(args, tokenizer, target_model, data)
        self.target_model.mode = 'identifier_cls'
        
        self.caliberate_model = caliberate_model
        self.caliberate_model = self.caliberate_model.to(self.device)
        self.caliberate_model.mode = 'identifier_cls'
        self.feature = feature
    
    def compute_loss(self, model, inputs):
        labels = inputs["labels"]
        outputs = model(**inputs)
        logits = outputs.get("logits")
        loss_fct = torch.nn.CrossEntropyLoss(reduction='none')
                
        loss = loss_fct(logits.view(-1, logits.size(-1)), labels.view(-1))
        bs = logits.size(0)
        loss = loss.view(bs, -1)
        loss = loss.mean(dim=-1)
        return loss
    
    def get_feature_score(self):
        data_num = len(self.data) if self.args.local_rank == -1 else len(self.sampler)
        print(data_num)
        special_token_indices = [self.tokenizer.bos_token_id, self.tokenizer.eos_token_id]
    
        attention_masks = []
        input_ids_all = []
        data_all_idx = []
        target_feature_score = np.zeros((data_num, 1))
        caliberate_feature_score = np.zeros((data_num, 1))
        
        batch_size = self.args.batch_size
        with torch.no_grad():
            cur_gpu_idxs = [i for i in self.sampler]
            data_all_idx = np.array(cur_gpu_idxs).reshape(-1, 1)
            total_len = data_num // self.args.batch_size 
            if data_num % self.args.batch_size != 0:
                total_len += 1
            bar = tqdm(total=total_len)
            
            for batch_start in range(0, data_num, batch_size):
                
                cur_batch_code = []
                cur_batch_label = []
                
                for cur_idx in cur_gpu_idxs[batch_start : batch_start + batch_size]:
                    code_tokenized = self.data[cur_idx]['code_tokens']
                    
                    if isinstance(code_tokenized, str):
                        code_tokenized = code_tokenized.strip().split(' ')
                    identifiers = self.data[cur_idx]['identifiers']
                    
                    labels = []
                    for word in code_tokenized:
                        if word in identifiers:
                            labels.append(1)
                        else:
                            labels.append(0)
                    inputs = code_tokenized
                    labels = labels
                    cur_batch_code.append(inputs)
                    cur_batch_label.append(labels)
                
                cur_bs = len(cur_batch_code)
                encoder_input = self.tokenizer(text=cur_batch_code, is_split_into_words=True, add_special_tokens=True,
                          padding='max_length', truncation=True, max_length=512)
                output_labels = []
                for i, label in enumerate(cur_batch_label):
                    word_ids = encoder_input.word_ids(batch_index=i)  # Map tokens to their respective word.
                    previous_word_idx = None
                    label_ids = []
                    for word_idx in word_ids:  # Set the special tokens to -100.
                        if word_idx is None:
                            label_ids.append(-100)
                        elif word_idx != previous_word_idx:  # Only label the first token of a given word.
                            label_ids.append(label[word_idx])
                        else:
                            label_ids.append(-100)
                        previous_word_idx = word_idx

                    output_labels.append(label_ids)
                model_labels = output_labels
                
                model_inputs = {}
                model_inputs['input_ids'] = torch.tensor(encoder_input['input_ids']).to(self.device)
                model_inputs['attention_mask'] = torch.tensor(encoder_input['attention_mask']).to(self.device)
                model_inputs['labels'] = torch.tensor(model_labels).to(self.device)
        
                special_tokens_mask = torch.full(model_inputs['input_ids'].shape, False, dtype=torch.bool).to(self.device)
                for sp_id in special_token_indices:
                    special_tokens_mask = special_tokens_mask | (model_inputs['input_ids']==sp_id)
                
                model_inputs['labels'][special_tokens_mask] = -100
                model_inputs['return_dict'] = True
                # breakpoint()
                tar_loss = self.compute_loss(self.target_model, model_inputs).view(-1, 1)
                cal_loss = self.compute_loss(self.caliberate_model, model_inputs).view(-1, 1)
                # breakpoint()
                # print(batch_idx, self.args.batch_size * batch_idx + cur_bs, i)
                target_feature_score[batch_start : batch_start + cur_bs, :] = tar_loss.cpu()
                caliberate_feature_score[batch_start : batch_start + cur_bs, :] = cal_loss.cpu()
                
                # breakpoint()
                input_ids_all.append(model_inputs['input_ids'].cpu())
                attention_masks.append(model_inputs['attention_mask'].cpu())
                bar.update(1)    
        # holder.sort(key=lambda x:len(x['code']))
        
        attention_masks = np.vstack(attention_masks)
        input_ids_all = np.vstack(input_ids_all)
        
        res = {
            'target_feature': target_feature_score,
            'caliberate_feature': caliberate_feature_score,
            'attention_mask': attention_masks, 
            'input_ids_all': input_ids_all,
            'data_all_idx': data_all_idx,
        }
        return res
    
    def get_feature(self):
        res = self.get_feature_score()
        return res

