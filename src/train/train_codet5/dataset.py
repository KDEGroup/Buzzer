import torch.utils.data
from torch.utils.data.dataset import Dataset

import os
import json
import random
import logging
import pickle
import random

import enums
import pandas as pd
import numpy as np

from src.data_utils.csn import CsnMiaData

from threading import Lock

lock = Lock()

logger = logging.getLogger(__name__)


class CodeDataset(Dataset):

    def __init__(self, args, dataset_name, mode='pre_train', task=None, language=None, split=None, clone_mapping=None):

        super(CodeDataset, self).__init__()
        self.args = args
        self.dataset_name = dataset_name
        self.task = task
        self.mode = mode
        self.split = split
        self.paths = {}

        self.dataset_dir = os.path.join(args.dataset_root, self.mode)
        self.csn_dataset = CsnMiaData()
        
        if self.mode == 'pretrain':
            self.code = self.csn_dataset.data('mem_pretrain')
        elif self.mode == 'shadow':
            self.code = self.csn_dataset.data('non_shadow')
        elif self.mode == 'calibrate':
            self.code = self.csn_dataset.data('non_calibrate')     
            
        self.size = len(self.code)

    
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
            
        return output, label, maskid2label
    
    def __getitem__(self, idx):
        # cap
        
        code_tokenized = self.code[idx]['code_tokens']
        if isinstance(code_tokenized, str):
            code_tokenized = code_tokenized.strip().split(' ')
                
        ri = random.randint(0, 1)
        
        if self.task == enums.TASK_BIMODAL_DUAL_GENERATION:
            nl = self.code[idx]['docstring']
            if isinstance(nl, str):
                nl = nl.strip().split(' ')
        elif not self.args.no_nl and ri == 0:
            nl = self.code[idx]['docstring']
            if isinstance(nl, str):
                nl = nl.strip().split(' ')
        else:
            nl = ""
            
        lang = self.code[idx]['language']
        identifiers = self.code[idx]['identifiers']
        code_length = len(code_tokenized) 
        nl_length = len(nl)
        
        if self.task == enums.TASK_MASK_SPAN_PREDICTION:
            code_mask_num = int(code_length * 0.15 / 3) 
            if code_mask_num == 0: code_mask_num = 1
            
            code_mask_start_id = 0
            
            if nl != "":
                nl_mask_num = int(nl_length * 0.15 / 3) 
                if nl_mask_num == 0: nl_mask_num = 1
                
                nl, nl_label, nl_maskid2label = self.mask_strategy_2(nl, nl_mask_num)
                code_mask_start_id = len(nl_maskid2label)
            
            code, code_label, code_maskid2label = self.mask_strategy_2(code_tokenized, code_mask_num, mask_start_id=code_mask_start_id)
            
            if nl!= '':
                inputs = nl + ['</s>'] + code
                label = nl_label + ['</s>'] + code_label
            else:
                inputs = code
                label = code_label
            
            return inputs, label
            
        elif self.task == enums.TASK_IDENTIFIER_TAGGING:
            identifiers = list(set(identifiers))

            labels = [0] * len(nl) if nl != '' else []
            for word in code_tokenized:
                if word in identifiers:
                    labels.append(1)
                else:
                    labels.append(0)
            
            if nl == "":
                inputs = code_tokenized
                labels = labels
            else:
                inputs = nl + ['</s>'] + code_tokenized 
                labels = [0] + labels
            return inputs, labels

        elif self.task == enums.TASK_MASK_IDENTIFER_PREDICTION:
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

            if nl == "":
                inputs = code_tokenized
                labels = labels.split(' ')
                return inputs, labels
            else:
                inputs = nl + ['</s>'] + code_tokenized
                labels = labels.split(' ')
                return inputs, labels
            
        elif self.task == enums.TASK_BIMODAL_DUAL_GENERATION:
            i = random.randint(0, 1)
            if i == 0:
                return code_tokenized, nl
            else:
                return nl, code_tokenized
            
            
    def __len__(self):
        return self.size

    def set_task(self, task):
        self.task = task

    def save(self):
        """Save to binary pickle file"""
        path = os.path.join(self.args.dataset_save_dir, f'{self.dataset_name}.pk')
        with open(path, mode='wb') as f:
            pickle.dump(self, f)
        logger.info(f'Dataset saved to {path}')

    def subset(self, ratio):
        assert 0 < ratio <= 1, f'The subset ratio supposed to be 0 < ratio <= 1, but got ratio={ratio}'
        if ratio == 1:
            return self
        indices = random.sample(range(self.size), int(self.size * ratio))
        return torch.utils.data.Subset(self, indices)


def init_dataset(args, mode, task=None, language=None, split=None, clone_mapping=None,
                 load_if_saved=True) -> CodeDataset:
    name = '.'.join([sub_name for sub_name in [mode, task, language, split] if sub_name is not None])
        
    dataset = CodeDataset(args=args,
                          dataset_name=name,
                          mode=mode,
                          task=task,
                          language=language,
                          split=split,
                          clone_mapping=clone_mapping)
    return dataset


