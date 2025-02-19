import argparse
# import logging
from loguru import logger
import os
from pathlib import Path
import random

from sklearn.metrics import accuracy_score
import torch
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, RobertaTokenizer, Trainer
from src.common.utils import plot_roc
import numpy as np
from sklearn.svm import SVR
import torch.nn as nn
from sklearn.metrics import accuracy_score, roc_auc_score, \
                precision_score, recall_score, f1_score, precision_recall_curve, roc_curve
from src.common.utils import set_seed
from common import Attn, minmax_norm_2d_feature
from src.clf.core import train, test


def load_feature(data_path):
    feature = np.load(data_path, allow_pickle=True)
    feature = torch.Tensor(feature)
    attn = feature.ne(0) # not the llm attn, is the siganl llm
    
    return feature, attn

def process_llm_feature(args, mem_path, non_path, cal_mem_path=None, cal_non_path=None):
    mem_fea, mem_attn = load_feature(mem_path)
    non_fea, non_attn = load_feature(non_path)
        
    mem_feature = torch.nan_to_num(mem_fea, nan=1)
    non_feature = torch.nan_to_num(non_fea, nan=1)
    if args.use_cal:
        cal_mem_fea, _ = load_feature(cal_mem_path)
        cal_non_fea, _ = load_feature(cal_non_path)
        
        cal_mem_fea = torch.nan_to_num(cal_mem_fea, nan=1)
        cal_non_fea = torch.nan_to_num(cal_non_fea, nan=1)
        
        mem_feature = torch.log(mem_feature + 1) - torch.log(cal_mem_fea + 1)
        non_feature = torch.log(non_feature + 1) - torch.log(cal_non_fea + 1)
    else:
        mem_feature = torch.log(mem_feature + 1)
        non_feature = torch.log(non_feature + 1) 
    
    return mem_feature, non_feature, mem_attn, non_attn


class InferenceModel(nn.Module):
    def __init__(self, args) -> None:
        super(InferenceModel, self).__init__()

        self.args = args
        self.attn = Attn()
        self.mclf = nn.Sequential(
            nn.Linear(128, 32),
            nn.GELU(),
            nn.Linear(32, 1),
        )
    
    def compute_loss(self, mem_logits, non_logits):
        return ((0.5 - mem_logits) + non_logits).clamp(min=1e-6).mean()
    
    def forward(self, mem_input=None, mem_attn_input=None, non_input=None, non_attn_input=None, 
                **kwargs):
        
        mem_input_ = mem_input.unsqueeze(2)
        mem_attn = self.attn(mem_input_.cuda(), mem_attn_input.cuda())
        mem_logits = self.mclf(mem_attn)
        
        loss = None
        if non_input is not None:
            non_input_ = non_input.unsqueeze(2)
            non_attn = self.attn(non_input_.cuda(), non_attn_input.cuda())
            non_logits = self.mclf(non_attn)
                    
            loss = self.compute_loss(mem_logits, non_logits)
            
            return loss
        
        return mem_logits


class SignalDataset(torch.utils.data.Dataset):
    def __init__(self, mem, non, mem_attn, non_attn, mode='train'):
        self.mem = mem 
        self.non = non 
        self.mem_attn = mem_attn 
        self.non_attn = non_attn
        self.mode = mode        
        self.mem_len = len(mem)
        self.non_len = len(non)
        
    def __len__(self):
        if self.mode == 'train':
            return self.non_len
        else:
            return self.mem_len + self.non_len

    def __getitem__(self, i):
        if self.mode == 'train':
            ri = random.randint(0, self.mem_len - 1)
            return self.mem[ri], self.mem_attn[ri], self.non[i], self.non_attn[i]
        else:
            if i < self.mem_len:
                return self.mem[i], self.mem_attn[i], 1 
            else:
                return self.non[i - self.mem_len], self.non_attn[i - self.mem_len], 0
    
    def collate(self, batch):
        if self.mode == 'train':
            model_inputs = {}
            mem, mem_attn, non, non_attn = map(list, zip(*batch))

            bs = len(mem)
            mem = torch.cat(mem)
            mem = mem.view(bs, -1)
            mem_attn = torch.cat(mem_attn)
            mem_attn = mem_attn.view(bs, -1)
            
            non = torch.cat(non)
            non = non.view(bs, -1)
            non_attn = torch.cat(non_attn)
            non_attn = non_attn.view(bs, -1)

            model_inputs['mem_input'] = mem
            model_inputs['mem_attn_input'] = mem_attn
            model_inputs['non_input'] = non
            model_inputs['non_attn_input'] = non_attn
            
            return model_inputs 
        
        else:
            model_inputs = {}
            mem, mem_attn, labels = map(list, zip(*batch))
            bs = len(mem)
            mem = torch.cat(mem)
            mem = mem.view(bs, -1)
            mem_attn = torch.cat(mem_attn)
            mem_attn = mem_attn.view(bs, -1)

            labels = torch.tensor(labels)
            
            model_inputs['mem_input'] = mem
            model_inputs['mem_attn_input'] = mem_attn
            model_inputs['labels'] = labels
            return model_inputs 

if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument('--target_model', type=str, default='codet5')
    parser.add_argument('--mia_type', type=str, default='wb')
    parser.add_argument('--use_cal', default=False, action='store_true')
    parser.add_argument('--output_root', type=str, default='LLM_MIA/output/feature_clf')
    parser.add_argument('--model_type', type=str, default='selfattn')
    parser.add_argument('--run_name', type=str, default='default')
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--wonl', action='store_true')
    parser.add_argument('--epoch_num', default=1, type=int)
    parser.add_argument('--eval_steps', default=300, type=int)
    
    parser.add_argument('--signal_dir', type=str, default='')
    
    
    args = parser.parse_args()
    
    args.output_path = os.path.join(args.output_root, args.run_name)
    if os.path.exists(args.output_path) is False:
        os.makedirs(args.output_path)
    
    logger.add(os.path.join(args.output_path, 'run.log'), level='INFO')
    logger.add(os.path.join(args.output_path, 'error.log'), level='ERROR')
    
    set_seed(42)

    en = args.epoch_num
    if args.mia_type == 'wb':
        train_mem_name = 'wb_mem_train'
        train_non_name = 'wb_non_train'
        train_from = 'target'
        test_from = 'target'
    elif args.mia_type == 'bb':
        train_mem_name = 'non_shadow'
        train_non_name = 'non_utils'
        train_from = 'shadow'
        test_from = 'target'
    
    # train on shadow, test on target
    # both black and white should test on target model
    test_mem_name = 'mem_test'
    test_non_name = 'non_test'
    
    target_model = args.target_model.lower()
    train_from_dir = os.path.join(args.signal_dir, train_from)
    test_from_dir = os.path.join(args.signal_dir, test_from)
    calibrate_dir = os.path.join(args.signal_dir, 'calibrate')
    
    train_mem_path = os.path.join(train_from_dir, f'{train_mem_name}_signal.npy')
    train_non_path = os.path.join(train_from_dir, f'{train_non_name}_signal.npy')
    cal_mem_path = os.path.join(calibrate_dir, f'{train_mem_name}_signal.npy')
    cal_non_path = os.path.join(calibrate_dir, f'{train_non_name}_signal.npy')
    
    test_mem_path = os.path.join(test_from_dir, f'{test_mem_name}_signal.npy')
    test_non_path = os.path.join(test_from_dir, f'{test_non_name}_signal.npy')
    test_cal_mem_path = os.path.join(calibrate_dir, f'{test_mem_name}_signal.npy')
    test_cal_non_path = os.path.join(calibrate_dir, f'{test_non_name}_signal.npy')
    
    train_mem_feature, train_non_feature, mem_attn, non_attn = \
        process_llm_feature(args, train_mem_path, train_non_path, cal_mem_path, cal_non_path)
        
    logger.info('Constructing training feature...')
    eval_dataset = SignalDataset(train_mem_feature[:2000], train_non_feature[:2000], \
                        mem_attn[:2000], non_attn[:2000],\
                        mode='eval')
    train_dataset = SignalDataset(train_mem_feature[2000:], train_non_feature[2000:], \
                        mem_attn[2000:], non_attn[2000:],\
                        mode='train')
    
    saved_root = args.output_path
    
    logger.info('Training...')
    model = InferenceModel(args)
    model = train(args, train_dataset, model, saved_root, eval_dataset, eval_steps=args.eval_steps)

    model.load_state_dict(torch.load(os.path.join(saved_root, 'best_model.bin')))
    
    test_mem_feature, test_non_feature, mem_attn, non_attn = \
        process_llm_feature(args, test_mem_path, test_non_path, test_cal_mem_path, test_cal_non_path)
    
    dataset = SignalDataset(test_mem_feature, test_non_feature, mem_attn, non_attn, mode='test')
    logger.info('Testing...')
    auc = test(args, dataset, model, more_detail=True)
