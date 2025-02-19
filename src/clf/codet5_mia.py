import argparse
from loguru import logger
import os
from pathlib import Path
import pickle
import random
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingClassifier, GradientBoostingRegressor
from sklearn.neural_network import MLPClassifier, MLPRegressor

from sklearn.metrics import accuracy_score
import torch
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, RobertaTokenizer, Trainer
import numpy as np
from sklearn.svm import SVR
import torch.nn as nn
from sklearn.metrics import accuracy_score, roc_auc_score, \
                precision_score, recall_score, f1_score, precision_recall_curve, roc_curve
from src.common.utils import set_seed
from src.common.block import Attn, minmax_norm_2d_feature
from src.clf.core import train, test


class InferenceModel(nn.Module):
    def __init__(self, args, input_size=100) -> None:
        super(InferenceModel, self).__init__()

        self.args = args
        self.bdg_attn = Attn()
        self.bdg_clf = nn.Sequential(
            nn.Linear(128, 32),
            nn.GELU(),
            nn.Linear(32, 1),
        )
        
        self.mip_attn = Attn()
        self.mip_clf = nn.Sequential(
            nn.Linear(128, 32),
            nn.GELU(),
            nn.Linear(32, 1),
        )
        
        self.msp_clf = nn.Sequential(
            nn.Linear(100, 64),
            nn.GELU(),
            
            nn.Linear(64, 16),
            nn.GELU(),
            
            nn.Linear(16, 1)
        )
        self.it_clf = nn.Sequential(
            nn.Linear(1, 16),
            nn.GELU(),
            nn.Linear(16, 1),
        )
        
        self.combine = nn.Sequential(
            nn.Linear(4, 1)
        )
    
    def compute_loss(self, mem_logits, non_logits):
        return ((0.5 - mem_logits) + non_logits).clamp(min=1e-6).mean()
    
    
    def forward(self, mem_bdg_input=None, mem_mip_input=None, non_mip_input=None, 
                non_bdg_input=None, mem_msp_input=None, non_msp_input=None,
                labels=None, mem_bdg_attn_mask=None, mem_mip_attn_mask=None, 
                mem_it_input=None, mem_it_attn_mask=None, non_it_input=None, non_it_attn_mask=None,
                non_bdg_attn_mask=None, non_mip_attn_mask=None,
                mem_msp_attn_mask=None, non_msp_attn_mask=None, **kwargs):
        
        # breakpoint()
        if self.args.use_bdg:
            mem_bdg_input = mem_bdg_input.unsqueeze(2)
            mem_bdg_attn = self.bdg_attn(mem_bdg_input.cuda(), mem_bdg_attn_mask.cuda())
            mem_bdg_logits = self.bdg_clf(mem_bdg_attn)
        
        if self.args.use_mip:
            mem_mip_input = mem_mip_input.unsqueeze(2)
            mem_mip_attn = self.mip_attn(mem_mip_input.cuda(), mem_mip_attn_mask.cuda())
            mem_mip_logits = self.mip_clf(mem_mip_attn)
        
        if self.args.use_msp:
            mem_msp_logits = self.msp_clf(mem_msp_input.cuda())
            
        if self.args.use_it:
            mem_it_logits = self.it_clf(mem_it_input.cuda())
        
        if self.args.use_bdg and self.args.use_mip and self.args.use_msp and self.args.use_it:
            mem_logits = torch.cat([mem_bdg_logits, mem_mip_logits, mem_msp_logits, mem_it_logits], dim=-1)
            mem_logits = self.combine(mem_logits)
        else:
            if self.args.use_bdg:
                mem_logits = mem_bdg_logits
            elif self.args.use_mip:
                mem_logits = mem_mip_logits
            elif self.args.use_msp:
                mem_logits = mem_msp_logits
            elif self.args.use_it:
                mem_logits = mem_it_logits
        
        loss = None
        if non_mip_input is not None:
            if self.args.use_bdg:
                non_bdg_input = non_bdg_input.unsqueeze(2)
                non_bdg_attn = self.bdg_attn(non_bdg_input.cuda(), non_bdg_attn_mask.cuda())
                non_bdg_logits = self.bdg_clf(non_bdg_attn)
            
            if self.args.use_mip:
                non_mip_input = non_mip_input.unsqueeze(2)
                non_mip_attn = self.mip_attn(non_mip_input.cuda(), non_mip_attn_mask.cuda())
                non_mip_logits = self.mip_clf(non_mip_attn)
            
            if self.args.use_msp:
                non_msp_logits = self.msp_clf(non_msp_input.cuda())
            
            if self.args.use_it:
                non_it_logits = self.it_clf(non_it_input.cuda())
            
            if self.args.use_bdg and self.args.use_mip and self.args.use_msp and self.args.use_it:
                non_logits = torch.cat([non_bdg_logits, non_mip_logits, non_msp_logits, non_it_logits], dim=-1)
                non_logits = self.combine(non_logits)
            else:
                if self.args.use_bdg:
                    non_logits = non_bdg_logits
                elif self.args.use_mip:
                    non_logits = non_mip_logits
                elif self.args.use_msp:
                    non_logits = non_msp_logits
                elif self.args.use_it:
                    non_logits = non_it_logits
                    
            loss = self.compute_loss(mem_logits, non_logits)
            return loss
        
        return mem_logits


def read_data(data_root, data_name, fea_name):
    path = Path(data_root) / f'{data_name}_{fea_name}.npy'
    
    data = np.load(path, allow_pickle=True)
    
    return data


class Feature:
    def __init__(self, feature, attn_mask, input_ids, type='seq',tokenizer=None):
        self.fea = feature 
        self.attn = attn_mask 
        self.input_ids = input_ids
        self.tokenizer = tokenizer
        self.type = type
        
        self.fea_tensor, self.attn_tensor = self.process_2d_feature(self.fea, self.attn, input_ids=self.input_ids)
    
    def get_feature(self):
        return self.fea_tensor, self.attn_tensor
    
    def process_2d_feature(self, feature, attn, input_ids=None):
        feature = torch.Tensor(feature)
        attn = torch.Tensor(attn)
        return feature, attn.bool()


def load_feature_by_name(args, target_model_root, train_mem_name, train_non_name, feature_name):
    target_train_mem = read_data(target_model_root, train_mem_name, feature_name)
    target_train_non = read_data(target_model_root, train_non_name, feature_name)
    target_train_mem_feature = Feature(target_train_mem['target_feature'], 
                                       target_train_mem['attention_mask'], target_train_mem['input_ids_all'], type=feature_type)
    target_train_non_feature = Feature(target_train_non['target_feature'], 
                                       target_train_non['attention_mask'], target_train_non['input_ids_all'], type=feature_type)
    train_mem_feature, train_mem_attn = target_train_mem_feature.get_feature()
    train_non_feature, train_non_attn = target_train_non_feature.get_feature()
    
    train_mem_feature = torch.nan_to_num(train_mem_feature, nan=1)
    train_non_feature = torch.nan_to_num(train_non_feature, nan=1)
    
    if args.use_cal:
        cal_train_mem = target_train_mem['caliberate_feature']
        cal_train_non = target_train_non['caliberate_feature']
        cal_train_mem_attn = target_train_mem['attention_mask']
        cal_train_non_attn = target_train_non['attention_mask']
        cal_train_mem_ids = target_train_mem['input_ids_all']
        cal_train_non_ids = target_train_non['input_ids_all']
        
        caliberte_train_mem_feature = Feature(cal_train_mem, cal_train_mem_attn, cal_train_mem_ids, type=feature_type)
        caliberte_train_non_feature = Feature(cal_train_non, cal_train_non_attn, cal_train_non_ids, type=feature_type)
        
        cal_train_mem_feature, cal_train_mem_attn = caliberte_train_mem_feature.get_feature()
        cal_train_non_feature, cal_train_non_attn = caliberte_train_non_feature.get_feature()
        
        cal_train_mem_feature = torch.nan_to_num(cal_train_mem_feature, nan=1)
        cal_train_non_feature = torch.nan_to_num(cal_train_non_feature, nan=1)

        mem_feature = torch.log(train_mem_feature + 1) - torch.log(cal_train_mem_feature + 1)
        non_feature = torch.log(train_non_feature + 1) - torch.log(cal_train_non_feature + 1)
    else:
        mem_feature = torch.log(train_mem_feature + 1)
        non_feature = torch.log(train_non_feature + 1) 

    return mem_feature, non_feature, train_mem_attn, train_non_attn, target_train_mem['data_all_idx'], target_train_non['data_all_idx']


class SignalDataset(torch.utils.data.Dataset):
    def __init__(self, mem, non, mode='train', model_type='lstm'):
        self.mem = mem 
        self.non = non 
        self.mode = mode
        self.model_type = model_type
        
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
                
            return self.mem[ri]['bdg_fea'], self.mem[ri]['bdg_att'], self.mem[ri]['mip_fea'], self.mem[ri]['mip_att'], \
                self.mem[ri]['msp_fea'], self.mem[ri]['msp_att'], self.mem[ri]['it_fea'], self.mem[ri]['it_fea'], \
                self.non[i]['bdg_fea'], self.non[i]['bdg_att'], self.non[i]['mip_fea'], self.non[i]['mip_att'], \
                self.non[i]['msp_fea'], self.non[i]['msp_att'],self.non[i]['it_fea'], self.non[i]['it_fea'],
        
        else:
            if i < self.mem_len:
                return self.mem[i]['bdg_fea'], self.mem[i]['bdg_att'], self.mem[i]['mip_fea'], self.mem[i]['mip_att'], \
                    self.mem[i]['msp_fea'], self.mem[i]['msp_att'],self.mem[i]['it_fea'], self.mem[i]['it_att'], self.mem[i]['idx'], 1 
            else:
                return self.non[i - self.mem_len]['bdg_fea'], self.non[i - self.mem_len]['bdg_att'], \
                    self.non[i - self.mem_len]['mip_fea'], self.non[i - self.mem_len]['mip_att'], \
                    self.non[i - self.mem_len]['msp_fea'], self.non[i - self.mem_len]['msp_att'], \
                    self.non[i - self.mem_len]['it_fea'], self.non[i - self.mem_len]['it_att'], self.non[i - self.mem_len]['idx'], 0
    
    def collate(self, batch):
        if self.mode == 'train':
            model_inputs = {}
            mem_bdg_raw, mem_bdg_attn_raw, mem_mip_raw, mem_mip_attn_raw, \
                mem_msp_raw, mem_msp_attn_raw, mem_it_raw, mem_it_attn_raw, \
                non_bdg_raw, non_bdg_attn_raw, non_mip_raw, non_mip_attn_raw, \
                non_msp_raw, non_msp_attn_raw, non_it_raw, non_it_attn_raw = map(list, zip(*batch))
            
            bs = len(mem_bdg_raw)
            mem_bdg = torch.concat(mem_bdg_raw, dim=0)
            mem_bdg_attn = torch.concat(mem_bdg_attn_raw, dim=0)
            mem_mip = torch.concat(mem_mip_raw, dim=0)
            mem_mip_attn = torch.concat(mem_mip_attn_raw, dim=0)
            mem_msp = torch.concat(mem_msp_raw, dim=0)
            mem_msp_attn = torch.concat(mem_msp_attn_raw, dim=0)
            mem_it = torch.concat(mem_it_raw, dim=0)
            mem_it_attn = torch.concat(mem_it_attn_raw, dim=0)
            
            non_bdg = torch.concat(non_bdg_raw, dim=0)
            non_bdg_attn = torch.concat(non_bdg_attn_raw, dim=0)
            non_mip = torch.concat(non_mip_raw, dim=0)
            non_mip_attn = torch.concat(non_mip_attn_raw, dim=0)
            non_msp = torch.concat(non_msp_raw, dim=0)
            non_msp_attn = torch.concat(non_msp_attn_raw, dim=0)
            non_it = torch.concat(non_it_raw, dim=0)
            non_it_attn = torch.concat(non_it_attn_raw, dim=0)
        
            mem_bdg = mem_bdg.view(bs, -1)
            mem_mip = mem_mip.view(bs, -1)
            mem_msp = mem_msp.view(bs, -1)
            mem_it = mem_it.view(bs, -1)
            non_bdg = non_bdg.view(bs, -1)
            non_mip = non_mip.view(bs, -1)
            non_msp = non_msp.view(bs, -1)   
            non_it = non_it.view(bs, -1) 

            mem_bdg_attn = mem_bdg_attn.view(bs, -1)
            mem_mip_attn = mem_mip_attn.view(bs, -1)
            mem_msp_attn = mem_msp_attn.view(bs, -1)
            mem_it_attn = mem_it_attn.view(bs, -1)
            non_bdg_attn = non_bdg_attn.view(bs, -1)
            non_mip_attn = non_mip_attn.view(bs, -1)
            non_msp_attn = non_msp_attn.view(bs, -1)
            non_it_attn = non_it_attn.view(bs, -1)
            
            # # breakpoint()
            # mem_length = mem_attn.sum(dim=1).view(-1)    
            # non_length = non_attn.sum(dim=1).view(-1)

            # if self.model_type == 'lstm':
            #     mem = pack_padded_sequence(mem, mem_length, batch_first=True, enforce_sorted=False)
            #     non = pack_padded_sequence(non, non_length, batch_first=True, enforce_sorted=False)

            model_inputs['mem_bdg_input'] = mem_bdg
            model_inputs['mem_mip_input'] = mem_mip
            model_inputs['mem_msp_input'] = mem_msp
            model_inputs['mem_it_input'] = mem_it
            model_inputs['non_bdg_input'] = non_bdg
            model_inputs['non_mip_input'] = non_mip
            model_inputs['non_msp_input'] = non_msp
            model_inputs['non_it_input'] = non_it
            
            model_inputs['mem_bdg_attn_mask'] = mem_bdg_attn
            model_inputs['mem_mip_attn_mask'] = mem_mip_attn
            model_inputs['mem_msp_attn_mask'] = mem_msp_attn
            model_inputs['mem_it_attn_mask'] = mem_it_attn
            model_inputs['non_bdg_attn_mask'] = non_bdg_attn
            model_inputs['non_mip_attn_mask'] = non_mip_attn
            model_inputs['non_msp_attn_mask'] = mem_msp_attn
            model_inputs['non_it_attn_mask'] = mem_it_attn
            return model_inputs
        
        else:
            model_inputs = {}
            mem_bdg_raw, mem_bdg_attn_raw, mem_mip_raw, mem_mip_attn_raw,\
                mem_msp_raw, mem_msp_attn_raw, mem_it_raw, mem_it_attn_raw, idx, labels = map(list, zip(*batch))
            
            bs = len(mem_bdg_raw)
            mem_bdg = torch.concat(mem_bdg_raw, dim=0)
            mem_bdg_attn = torch.concat(mem_bdg_attn_raw, dim=0)
            mem_mip = torch.concat(mem_mip_raw, dim=0)
            mem_mip_attn = torch.concat(mem_mip_attn_raw, dim=0)
            mem_msp = torch.concat(mem_msp_raw, dim=0)
            mem_msp_attn = torch.concat(mem_msp_attn_raw, dim=0)
            mem_it = torch.concat(mem_it_raw, dim=0)
            mem_it_attn = torch.concat(mem_it_attn_raw, dim=0)

            mem_bdg = mem_bdg.view(bs, -1)
            mem_mip = mem_mip.view(bs, -1)
            mem_msp = mem_msp.view(bs, -1)
            mem_it = mem_it.view(bs, -1)
                
            mem_bdg_attn = mem_bdg_attn.view(bs, -1)
            mem_mip_attn = mem_mip_attn.view(bs, -1)
            mem_msp_attn = mem_msp_attn.view(bs, -1)
            mem_it_attn = mem_it_attn.view(bs, -1)
            
            # mem_length = mem_attn.sum(dim=1).view(-1)  
              
            # if self.model_type == 'lstm':
            #     mem = pack_padded_sequence(mem, mem_length, batch_first=True, enforce_sorted=False)

            labels = torch.tensor(labels)
            model_inputs['mem_bdg_input'] = mem_bdg
            model_inputs['mem_mip_input'] = mem_mip
            model_inputs['mem_msp_input'] = mem_msp
            model_inputs['mem_it_input'] = mem_it
            model_inputs['mem_bdg_attn_mask'] = mem_bdg_attn
            model_inputs['mem_mip_attn_mask'] = mem_mip_attn
            model_inputs['mem_msp_attn_mask'] = mem_msp_attn
            model_inputs['mem_it_attn_mask'] = mem_it_attn
            model_inputs['labels'] = labels
            model_inputs['idx'] = torch.tensor(idx)
            return model_inputs

if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument('--target_model', type=str, default='codet5')
    parser.add_argument('--mia_type', type=str, default='wb')
    parser.add_argument('--feature_name', type=str, default='loss_each')
    parser.add_argument('--use_cal', default=False, action='store_true')
    
    parser.add_argument('--use_bdg', default=False, action='store_true')
    parser.add_argument('--use_mip', default=False, action='store_true')
    parser.add_argument('--use_msp', default=False, action='store_true')
    parser.add_argument('--use_it', default=False, action='store_true')
    
    parser.add_argument('--output_root', type=str, default='outputs/feature_clf_default')
    parser.add_argument('--model_type', type=str, default='selfattn')
    parser.add_argument('--run_name', type=str, default='default')
    parser.add_argument('--loss_type', type=str, default='mrl')
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--wonl', action='store_true')
    
    args = parser.parse_args()
    
    args.output_path = os.path.join(args.output_root, args.run_name)
    if os.path.exists(args.output_path) is False:
        os.makedirs(args.output_path)
    
    logger.add(os.path.join(args.output_path, 'run.log'), level='INFO')
    logger.add(os.path.join(args.output_path, 'error.log'), level='ERROR')

    set_seed()

    if args.mia_type == 'wb':
        train_mem_name = 'wb_mem_train'
        train_non_name = 'wb_non_train'
        train_from = 'target'
        test_from = 'target'
    elif args.mia_type == 'gb':
        train_mem_name = 'non_shadow'
        train_non_name = 'non_utils'
        train_from = 'target'
        test_from = 'shadow'
    
    test_mem_name = 'mem_test'
    test_non_name = 'non_test'
    
    test_mem_name = 'mem_test'
    test_non_name = 'non_test'
    
    target_model = args.target_model.lower()
    
    target_model = args.target_model.lower()
    
    target_model_root = f'outputs/{target_model}_signals/{target_model}_{train_from}'
    
    model, model_type, feature_type = InferenceModel(args), 'mlp', 'non_seq'
    
    logger.info('Constructing bdg feature...')
    bdg_train_mem_feature, bdg_train_non_feature, train_bdg_mem_attn, \
        train_bdg_non_attn, bdg_mem_idx, bdg_non_idx = load_feature_by_name(args, target_model_root, train_mem_name, 
                                                                                train_non_name, 'codet5_bdg')
    
    bdg_train_mem_feature[~train_bdg_mem_attn] = 0
    bdg_train_non_feature[~train_bdg_non_attn] = 0
    
    bdg_mem_idx = bdg_mem_idx.reshape(-1).tolist()
    bdg_non_idx = bdg_non_idx.reshape(-1).tolist()
    bdg_mem_idx2_fea = {}
    for fea, att, idx in zip(bdg_train_mem_feature, train_bdg_mem_attn, bdg_mem_idx):
        bdg_mem_idx2_fea[idx] = {
            'fea': fea,
            'att': att
        }
    bdg_non_idx2_fea = {}
    for fea, att, idx in zip(bdg_train_non_feature, train_bdg_non_attn, bdg_non_idx):
        bdg_non_idx2_fea[idx] = {
            'fea': fea,
            'att': att
        }
    
    logger.info('Constructing msp feature...')
    msp_train_mem_feature, msp_train_non_feature, train_msp_mem_attn, train_msp_non_attn, msp_mem_idx, msp_non_idx = load_feature_by_name(args, target_model_root, train_mem_name, 
                                                                                train_non_name, 'codet5_ntimes_msp')
    
    msp_mem_idx = msp_mem_idx.reshape(-1).tolist()
    msp_non_idx = msp_non_idx.reshape(-1).tolist()
    
    msp_mem_idx2_fea = {}
    for fea, att, idx in zip(msp_train_mem_feature, train_msp_mem_attn, msp_mem_idx):
        msp_mem_idx2_fea[idx] = {
            'fea': fea,
            'att': att
        }
    msp_non_idx2_fea = {}
    for fea, att, idx in zip(msp_train_non_feature, train_msp_non_attn, msp_non_idx):
        msp_non_idx2_fea[idx] = {
            'fea': fea,
            'att': att
        }
    
    logger.info('Constructing it feature...')
    it_train_mem_feature, it_train_non_feature, train_it_mem_attn, train_it_non_attn, it_mem_idx, it_non_idx = load_feature_by_name(args, target_model_root, train_mem_name, 
                                                                                train_non_name, 'codet5_it')
    
    it_mem_idx = it_mem_idx.reshape(-1).tolist()
    it_non_idx = it_non_idx.reshape(-1).tolist()
    
    it_mem_idx2_fea = {}
    for fea, att, idx in zip(it_train_mem_feature, train_it_mem_attn, it_mem_idx):
        it_mem_idx2_fea[idx] = {
            'fea': fea,
            'att': att
        }
    it_non_idx2_fea = {}
    for fea, att, idx in zip(it_train_non_feature, train_it_non_attn, it_non_idx):
        it_non_idx2_fea[idx] = {
            'fea': fea,
            'att': att
        }
    
        
    logger.info('Constructing mip feature...')
    mip_train_mem_feature, mip_train_non_feature, train_mip_mem_attn, train_mip_non_attn, mip_mem_idx, mip_non_idx = load_feature_by_name(args, target_model_root, train_mem_name, 
                                                                                train_non_name, 'codet5_mip')
    mip_train_mem_feature[~train_mip_mem_attn] = 0
    mip_train_non_feature[~train_mip_non_attn] = 0
    
    mip_train_mem_feature = mip_train_mem_feature.masked_fill(~train_mip_mem_attn, 0)
    mip_train_non_feature = mip_train_non_feature.masked_fill(~train_mip_non_attn, 0)

    mip_mem_idx = mip_mem_idx.reshape(-1).tolist()
    mip_non_idx = mip_non_idx.reshape(-1).tolist()
    
    all_mem = []
    for fea, att, idx in zip(mip_train_mem_feature, train_mip_mem_attn, mip_mem_idx):
        cur_bdg_fea = bdg_mem_idx2_fea[idx]['fea']
        cur_bdg_att = bdg_mem_idx2_fea[idx]['att']
        cur_msp_fea = msp_mem_idx2_fea[idx]['fea']
        cur_msp_att = msp_mem_idx2_fea[idx]['att']
        cur_it_fea = it_mem_idx2_fea[idx]['fea']
        cur_it_att = it_mem_idx2_fea[idx]['att']
        
        all_mem.append({
            'bdg_fea': cur_bdg_fea,
            'bdg_att': cur_bdg_att,
            'msp_fea': cur_msp_fea,
            'msp_att': cur_msp_att,
            'it_fea': cur_it_fea,
            'it_att': cur_it_att,
            'mip_fea': fea,
            'mip_att': att,
            'idx': idx 
        })
    
    all_non = []
    for fea, att, idx in zip(mip_train_non_feature, train_mip_non_attn, mip_non_idx):
        cur_bdg_fea = bdg_non_idx2_fea[idx]['fea']
        cur_bdg_att = bdg_non_idx2_fea[idx]['att']
        cur_msp_fea = msp_non_idx2_fea[idx]['fea']
        cur_msp_att = msp_non_idx2_fea[idx]['att']
        cur_it_fea = it_non_idx2_fea[idx]['fea']
        cur_it_att = it_non_idx2_fea[idx]['att']
        all_non.append({
            'bdg_fea': cur_bdg_fea,
            'bdg_att': cur_bdg_att,
            'msp_fea': cur_msp_fea,
            'msp_att': cur_msp_att,
            'it_fea': cur_it_fea,
            'it_att': cur_it_att,
            'mip_fea': fea,
            'mip_att': att,
            'idx': idx 
        })
    
    eval_dataset = SignalDataset(all_mem[:2000], all_non[:2000], \
                        mode='eval', model_type=model_type)
    train_dataset = SignalDataset(all_mem[2000:], all_non[2000:], \
                        mode='train', model_type=model_type)
    
    saved_root = args.output_path

    model = train(args, train_dataset, model, saved_root, eval_dataset, eval_steps=100)

    model.load_state_dict(torch.load(os.path.join(saved_root, 'best_model.bin')))

    # Testing, train on shadow, test on
    target_model_root = f'outputs/{target_model}_signals/{target_model}_{test_from}'
    bdg_test_mem_feature, bdg_test_non_feature, \
        test_bdg_mem_attn, test_bdg_non_attn, bdg_mem_idx, bdg_non_idx = \
            load_feature_by_name(args, target_model_root, test_mem_name, 
                                test_non_name, 'codet5_bdg')
            
    bdg_test_mem_feature[~test_bdg_mem_attn] = 0
    bdg_test_non_feature[~test_bdg_non_attn] = 0
    
    bdg_mem_idx = bdg_mem_idx.reshape(-1).tolist()
    bdg_non_idx = bdg_non_idx.reshape(-1).tolist()
    bdg_mem_idx2_fea = {}
    for fea, att, idx in zip(bdg_test_mem_feature, test_bdg_mem_attn, bdg_mem_idx):
        bdg_mem_idx2_fea[idx] = {
            'fea': fea,
            'att': att
        }
    bdg_non_idx2_fea = {}
    for fea, att, idx in zip(bdg_test_non_feature, test_bdg_non_attn, bdg_non_idx):
        bdg_non_idx2_fea[idx] = {
            'fea': fea,
            'att': att
        }
    
    logger.info('Constructing msp feature....')
    msp_test_mem_feature, msp_test_non_feature, test_msp_mem_attn, test_msp_non_attn, msp_mem_idx, msp_non_idx = load_feature_by_name(args, target_model_root, test_mem_name, 
                                                                                test_non_name, 'codet5_ntimes_msp')
    
    msp_mem_idx = msp_mem_idx.reshape(-1).tolist()
    msp_non_idx = msp_non_idx.reshape(-1).tolist()
    
    msp_mem_idx2_fea = {}
    for fea, att, idx in zip(msp_test_mem_feature, test_msp_mem_attn, msp_mem_idx):
        msp_mem_idx2_fea[idx] = {
            'fea': fea,
            'att': att
        }
    msp_non_idx2_fea = {}
    for fea, att, idx in zip(msp_test_non_feature, test_msp_non_attn, msp_non_idx):
        msp_non_idx2_fea[idx] = {
            'fea': fea,
            'att': att
        }
    
    logger.info('Constructing it feature....')
    it_test_mem_feature, it_test_non_feature, test_it_mem_attn, test_it_non_attn, it_mem_idx, it_non_idx = load_feature_by_name(args, target_model_root, test_mem_name, 
                                                                                test_non_name, 'codet5_it')
    
    it_mem_idx = it_mem_idx.reshape(-1).tolist()
    it_non_idx = it_non_idx.reshape(-1).tolist()
    
    it_mem_idx2_fea = {}
    for fea, att, idx in zip(it_test_mem_feature, test_it_mem_attn, it_mem_idx):
        it_mem_idx2_fea[idx] = {
            'fea': fea,
            'att': att
        }
    it_non_idx2_fea = {}
    for fea, att, idx in zip(it_test_non_feature, test_it_non_attn, it_non_idx):
        it_non_idx2_fea[idx] = {
            'fea': fea,
            'att': att
        }
    
    logger.info('Constructing mip feature....')
    mip_test_mem_feature, mip_test_non_feature, test_mip_mem_attn, test_mip_non_attn, mip_mem_idx, mip_non_idx = load_feature_by_name(args, target_model_root, test_mem_name, 
                                                                                test_non_name, 'codet5_mip')
    
    mip_test_mem_feature = mip_test_mem_feature.masked_fill(~test_mip_mem_attn, 0)
    mip_test_non_feature = mip_test_non_feature.masked_fill(~test_mip_non_attn, 0)
    
    
    mip_mem_idx = mip_mem_idx.reshape(-1).tolist()
    mip_non_idx = mip_non_idx.reshape(-1).tolist()
    
    
    all_mem = []
    for fea, att, idx in zip(mip_test_mem_feature, test_mip_mem_attn, mip_mem_idx):
        cur_bdg_fea = bdg_mem_idx2_fea[idx]['fea']
        cur_bdg_att = bdg_mem_idx2_fea[idx]['att']
        cur_msp_fea = msp_mem_idx2_fea[idx]['fea']
        cur_msp_att = msp_mem_idx2_fea[idx]['att']
        cur_it_fea = it_mem_idx2_fea[idx]['fea']
        cur_it_att = it_mem_idx2_fea[idx]['att']
        all_mem.append({
            'bdg_fea': cur_bdg_fea,
            'bdg_att': cur_bdg_att,
            'msp_fea': cur_msp_fea,
            'msp_att': cur_msp_att,
            'it_fea': cur_it_fea,
            'it_att': cur_it_att,
            'mip_fea': fea,
            'mip_att': att,
            'idx': idx 
        })
    
    all_non = []
    for fea, att, idx in zip(mip_test_non_feature, test_mip_non_attn, mip_non_idx):
        cur_bdg_fea = bdg_non_idx2_fea[idx]['fea']
        cur_bdg_att = bdg_non_idx2_fea[idx]['att']
        cur_msp_fea = msp_non_idx2_fea[idx]['fea']
        cur_msp_att = msp_non_idx2_fea[idx]['att']
        cur_it_fea = it_non_idx2_fea[idx]['fea']
        cur_it_att = it_non_idx2_fea[idx]['att']
        all_non.append({
            'bdg_fea': cur_bdg_fea,
            'bdg_att': cur_bdg_att,
            'msp_fea': cur_msp_fea,
            'msp_att': cur_msp_att,
            'it_fea': cur_it_fea,
            'it_att': cur_it_att,
            'mip_fea': fea,
            'mip_att': att,
            'idx': idx 
        })

    dataset = SignalDataset(all_mem, all_non, mode='test', model_type=model_type)
    logger.info('Testing ......')
    auc = test(args, dataset, model, more_detail=False)
    logger.info(auc)
