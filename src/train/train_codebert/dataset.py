import torch.utils.data
from torch.utils.data.dataset import Dataset

import os
import json
import random
import logging
import pickle
import random

from src.data_utils.csn import CsnMiaData

from threading import Lock

lock = Lock()

logger = logging.getLogger(__name__)


class CodeDataset(Dataset):
    def __init__(self, args, dataset_name, 
                            mode='pre_train', 
                            task=None,
                            split=None,):

        super(CodeDataset, self).__init__()
        self.args = args
        self.dataset_name = dataset_name
        self.task = task
        self.mode = mode
        self.split = split
        self.paths = {}

        self.csn_dataset = CsnMiaData()
        
        self.dataset_dir = os.path.join(args.dataset_root, self.mode)

        # load dataset
        if self.mode == 'pretrain':
            self.code = self.csn_dataset.data('mem_pretrain')
        elif self.mode == 'shadow':
            self.code = self.csn_dataset.data('non_shadow')
        elif self.mode == 'calibrate':
            self.code = self.csn_dataset.data('non_calibrate')     
        # breakpoint()
        self.size = len(self.code)
        
    def __getitem__(self, idx):
        # cap
        code_tokenized = self.code[idx]['code_tokens']
        if isinstance(code_tokenized, str):
            code_tokenized = code_tokenized.strip().split(' ')
        nl = self.code[idx]['docstring']
        if isinstance(nl, str):
            nl = nl.strip().split(' ')
        code_length = len(code_tokenized) 
        nl_length = len(nl)
    
        if self.task == 'mlm':
            ri = random.randint(0, 2)
            
            if ri == 0:
                return code_tokenized
            elif ri == 1:
                return nl
            else:
                return nl + ['</s>'] +  code_tokenized
            
        elif self.task == 'rtd':
            ri = random.randint(0, 2)
                
            if ri == 0:
                return code_tokenized
            elif ri == 1:
                return nl
            else:
                return nl + ['</s>'] +  code_tokenized

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
                 ) -> CodeDataset:
    
    name = '.'.join([sub_name for sub_name in [mode, task, language, split] if sub_name is not None])
        
    dataset = CodeDataset(args=args,
                          dataset_name=name,
                          mode=mode,
                          task=task,
                          split=split,
                          )
    return dataset


