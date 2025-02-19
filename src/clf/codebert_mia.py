import argparse
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
from src.common.evaluate_metrics import plot_roc
import numpy as np
from sklearn.svm import SVR
import torch.nn as nn
from sklearn.metrics import accuracy_score, roc_auc_score, \
                precision_score, recall_score, f1_score, precision_recall_curve, roc_curve
from src.common.utils import set_seed
from src.common.block import Attn 
from src.common.block import minmax_norm_2d_feature
from src.clf.core import train, test
from loguru import logger



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
                
            return self.mem[ri]['mlm_fea'], self.mem[ri]['mlm_att'], self.mem[ri]['rtd_fea'], self.mem[ri]['rtd_att'], \
                self.non[i]['mlm_fea'], self.non[i]['mlm_att'], self.non[i]['rtd_fea'], self.non[i]['rtd_att']
        
        else:
            if i < self.mem_len:
                return self.mem[i]['mlm_fea'], self.mem[i]['mlm_att'], self.mem[i]['rtd_fea'], self.mem[i]['rtd_att'], self.mem[i]['idx'], 1 
            else:
                return self.non[i - self.mem_len]['mlm_fea'], self.non[i - self.mem_len]['mlm_att'], self.non[i - self.mem_len]['rtd_fea'], self.non[i - self.mem_len]['rtd_att'], self.non[i - self.mem_len]['idx'], 0
    
    
    def collate(self, batch):
        if self.mode == 'train':
            model_inputs = {}
            mem_mlm_raw, mem_mlm_attn_raw, mem_rtd_raw, mem_rtd_attn_raw, \
               non_mlm_raw, non_mlm_attn_raw, non_rtd_raw, non_rtd_attn_raw = map(list, zip(*batch))
            
            bs = len(mem_mlm_raw)
            mem_mlm = torch.concat(mem_mlm_raw, dim=0)
            mem_mlm_attn = torch.concat(mem_mlm_attn_raw, dim=0)
            mem_rtd = torch.concat(mem_rtd_raw, dim=0)
            mem_rtd_attn = torch.concat(mem_rtd_attn_raw, dim=0)
            
            non_mlm = torch.concat(non_mlm_raw, dim=0)
            non_mlm_attn = torch.concat(non_mlm_attn_raw, dim=0)
            non_rtd = torch.concat(non_rtd_raw, dim=0)
            non_rtd_attn = torch.concat(non_rtd_attn_raw, dim=0)
            
            if self.model_type in ['lstm', 'selfattn']:
                mem = mem.view(bs, -1, 1)
                non = non.view(bs, -1, 1)
            else:
                mem_mlm = mem_mlm.view(bs, -1)
                mem_rtd = mem_rtd.view(bs, -1)
                non_mlm = non_mlm.view(bs, -1)
                non_rtd = non_rtd.view(bs, -1)

            mem_mlm_attn = mem_mlm_attn.view(bs, -1)
            mem_rtd_attn = mem_rtd_attn.view(bs, -1)
            non_mlm_attn = non_mlm_attn.view(bs, -1)
            non_rtd_attn = non_rtd_attn.view(bs, -1)

            model_inputs['mem_mlm_input'] = mem_mlm
            model_inputs['mem_rtd_input'] = mem_rtd
            model_inputs['non_mlm_input'] = non_mlm
            model_inputs['non_rtd_input'] = non_rtd
            
            model_inputs['mem_mlm_attn_mask'] = mem_mlm_attn
            model_inputs['mem_rtd_attn_mask'] = mem_rtd_attn
            model_inputs['non_mlm_attn_mask'] = non_mlm_attn
            model_inputs['non_rtd_attn_mask'] = non_rtd_attn
            return model_inputs
        
        else:
            model_inputs = {}
            mem_mlm_raw, mem_mlm_attn_raw, mem_rtd_raw, mem_rtd_attn_raw, idx, labels = map(list, zip(*batch))
            
            bs = len(mem_mlm_raw)
            mem_mlm = torch.concat(mem_mlm_raw, dim=0)
            mem_mlm_attn = torch.concat(mem_mlm_attn_raw, dim=0)
            mem_rtd = torch.concat(mem_rtd_raw, dim=0)
            mem_rtd_attn = torch.concat(mem_rtd_attn_raw, dim=0)
            
            if self.model_type in ['lstm', 'selfattn']:
                mem = mem.view(bs, -1, 1)
            else:
                mem_mlm = mem_mlm.view(bs, -1)
                mem_rtd = mem_rtd.view(bs, -1)
                
            mem_mlm_attn = mem_mlm_attn.view(bs, -1)
            mem_rtd_attn = mem_rtd_attn.view(bs, -1)

            labels = torch.tensor(labels)
            idx = torch.tensor(idx)
            model_inputs['mem_mlm_input'] = mem_mlm
            model_inputs['mem_rtd_input'] = mem_rtd
            model_inputs['mem_mlm_attn_mask'] = mem_mlm_attn
            model_inputs['mem_rtd_attn_mask'] = mem_rtd_attn
            model_inputs['labels'] = labels
            model_inputs['idx'] = idx
            return model_inputs
            

class InferenceModel(nn.Module):
    def __init__(self, args, input_size=100) -> None:
        super(InferenceModel, self).__init__()

        self.args = args
        self.mlm_clf = nn.Sequential(
            nn.Linear(input_size, 64),
            nn.GELU(),
            
            nn.Linear(64, 16),
            nn.GELU(),
            
            nn.Linear(16, 1)
        )
        self.rtd_clf = nn.Sequential(
            nn.Linear(input_size, 64),
            nn.GELU(),
            
            nn.Linear(64, 16),
            nn.GELU(),
            
            nn.Linear(16, 1)
        )
        self.combine = nn.Sequential(
            nn.Linear(2, 1)
        )
    
    def compute_loss(self, mem_logits, non_logits):
        return ((0.5 - mem_logits) + non_logits).clamp(min=1e-6).mean()
    
    
    def forward(self, mem_mlm_input=None, mem_rtd_input=None, non_rtd_input=None, 
                non_mlm_input=None, labels=None, mem_mlm_attn_mask=None, mem_rtd_attn_mask=None, 
                non_mlm_attn_mask=None, non_rtd_attn_mask=None, **kwargs):
            
        if self.args.use_mlm and self.args.use_rtd:
            mem_clf_logits = self.mlm_clf(mem_mlm_input.cuda())
            mem_rtd_logits = self.rtd_clf(mem_rtd_input.cuda())   
            mem_logits = self.combine(torch.cat([mem_clf_logits, mem_rtd_logits], dim=-1))
        elif self.args.use_mlm:
            mem_logits = self.mlm_clf(mem_mlm_input.cuda())
        elif self.args.use_rtd:
            mem_logits = self.rtd_clf(mem_rtd_input.cuda())    
                
        loss = None
        if non_rtd_input is not None:
            if self.args.use_mlm and self.args.use_rtd:
                non_clf_logits = self.mlm_clf(non_mlm_input.cuda())
                non_rtd_logits = self.rtd_clf(non_rtd_input.cuda())
                non_logits = self.combine(torch.cat([non_clf_logits, non_rtd_logits], dim=-1))
            elif self.args.use_mlm:
                non_logits = self.mlm_clf(non_mlm_input.cuda())
            elif self.args.use_rtd:
                non_logits = self.rtd_clf(non_rtd_input.cuda())

            loss = self.compute_loss(mem_logits, non_logits)
            return loss
        
        return mem_logits


def read_data(data_root, data_name, fea_name):
    path = Path(data_root) / f'{data_name}_{fea_name}.npy'
    data = np.load(path, allow_pickle=True)
    
    return data


class Feature:
    def __init__(self, feature, attn_mask, input_ids, type='seq'):
        self.fea = feature 
        self.attn = attn_mask 
        self.input_ids = input_ids
        self.type = type
        
        self.fea_tensor, self.attn_tensor = self.process_2d_feature(self.fea, self.attn, input_ids=self.input_ids)
    
    def get_feature(self):
        return self.fea_tensor, self.attn_tensor
    
    def process_2d_feature(self, feature, attn, input_ids):
        feature_tensor = torch.Tensor(feature)
        attn_tensor = torch.Tensor(attn)
        
        return feature_tensor, attn_tensor.bool()


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
    
    if feature_type == 'seq':
        mem_feature[~train_mem_attn] = 0
        non_feature[~train_non_attn] = 0

    return mem_feature, non_feature, train_mem_attn, train_non_attn, target_train_mem['data_all_idx'], target_train_non['data_all_idx']


if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument('--target_model', type=str, default='codebert')
    parser.add_argument('--mia_type', type=str, default='wb')
    parser.add_argument('--feature_name', type=str, default='loss_each')
    parser.add_argument('--use_cal', default=False, action='store_true')
    parser.add_argument('--output_path', type=str, default='outputs/feature_clf_default')
    parser.add_argument('--model_type', type=str, default='selfattn')
    parser.add_argument('--run_name', type=str, default='default')
    parser.add_argument('--loss_type', type=str, default='mrl')
    parser.add_argument('--wonl', action='store_true')
    parser.add_argument('--use_mlm', action='store_true')
    parser.add_argument('--use_rtd', action='store_true')
    parser.add_argument('--batch_size', type=int, default=64)
    
    args = parser.parse_args()
    
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
        
    elif args.mia_type == 'bb':
        train_mem_name = 'non_shadow'
        train_non_name = 'non_utils'
        train_from = 'shadow'
        test_from = 'target'
    
    test_mem_name = 'mem_test'
    test_non_name = 'non_test'
    
    # feature_name = args.feature_name
    target_model = args.target_model.lower()
    
    target_model_root = f'outputs/{target_model}_signals/{target_model}_{train_from}'
    
    model, model_type, feature_type = InferenceModel(args), 'mlp', 'non_seq'
    mlm_train_mem_feature, mlm_train_non_feature, train_mlm_mem_attn, \
        train_mlm_non_attn, mlm_mem_idx, mlm_non_idx = load_feature_by_name(args, target_model_root, train_mem_name, 
                                                                                train_non_name, 'ntimes_mlm_loss')
    mlm_mem_idx = mlm_mem_idx.reshape(-1).tolist()
    mlm_non_idx = mlm_non_idx.reshape(-1).tolist()

    mlm_mem_idx2_fea = {}
    for fea, att, idx in zip(mlm_train_mem_feature, train_mlm_mem_attn, mlm_mem_idx):
        mlm_mem_idx2_fea[idx] = {
            'fea': fea,
            'att': att
        }
    mlm_non_idx2_fea = {}
    for fea, att, idx in zip(mlm_train_non_feature, train_mlm_non_attn, mlm_non_idx):
        mlm_non_idx2_fea[idx] = {
            'fea': fea,
            'att': att
        }
    
    rtd_train_mem_feature, rtd_train_non_feature, train_rtd_mem_attn, \
        train_rtd_non_attn, rtd_mem_idx, rtd_non_idx = load_feature_by_name(args, target_model_root, train_mem_name, 
                                                                                train_non_name, 'ntimes_rtd_loss')

    rtd_mem_idx = rtd_mem_idx.reshape(-1).tolist()
    rtd_non_idx = rtd_non_idx.reshape(-1).tolist()
    
    all_mem = []
    for fea, att, idx in zip(rtd_train_mem_feature, train_rtd_mem_attn, rtd_mem_idx):
        cur_mlm_fea = mlm_mem_idx2_fea[idx]['fea']
        cur_mlm_att = mlm_mem_idx2_fea[idx]['att']
        all_mem.append({
            'mlm_fea': cur_mlm_fea,
            'mlm_att': cur_mlm_att,
            'rtd_fea': fea,
            'rtd_att': att,
            'idx': idx 
        })
    all_non = []
    for fea, att, idx in zip(rtd_train_non_feature, train_rtd_non_attn, rtd_non_idx):
        cur_mlm_fea = mlm_non_idx2_fea[idx]['fea']
        cur_mlm_att = mlm_non_idx2_fea[idx]['att']
        all_non.append({
            'mlm_fea': cur_mlm_fea,
            'mlm_att': cur_mlm_att,
            'rtd_fea': fea,
            'rtd_att': att,
            'idx': idx 
        })  
    
    eval_dataset = SignalDataset(all_mem[:2000], all_non[:2000], \
                        mode='eval', model_type=model_type)
    train_dataset = SignalDataset(all_mem[2000:], all_non[2000:], \
                        mode='train', model_type=model_type)
    
    saved_root = os.path.join(args.output_path, args.run_name)
    
    if os.path.exists(saved_root) is False:
        os.makedirs(saved_root)
        
    model = train(args, train_dataset, model, saved_root, eval_dataset, eval_steps=100)
    model.load_state_dict(torch.load(os.path.join(saved_root, 'best_model.bin')))
    
    
    target_model_root = f'outputs/{target_model}_signals/{target_model}_{test_from}'
    
    mlm_test_mem_feature, mlm_test_non_feature, test_mlm_mem_attn, test_mlm_non_attn, mlm_mem_idx, mlm_non_idx = load_feature_by_name(args, target_model_root, test_mem_name, 
                                                                                test_non_name, 'ntimes_mlm_loss')
    mlm_mem_idx = mlm_mem_idx.reshape(-1).tolist()
    mlm_non_idx = mlm_non_idx.reshape(-1).tolist()
    mlm_mem_idx2_fea = {}
    for fea, att, idx in zip(mlm_test_mem_feature, test_mlm_mem_attn, mlm_mem_idx):
        mlm_mem_idx2_fea[idx] = {
            'fea': fea,
            'att': att
        }
    mlm_non_idx2_fea = {}
    for fea, att, idx in zip(mlm_test_non_feature, test_mlm_non_attn, mlm_non_idx):
        mlm_non_idx2_fea[idx] = {
            'fea': fea,
            'att': att
        }
        
    rtd_test_mem_feature, rtd_test_non_feature, test_rtd_mem_attn, test_rtd_non_attn, rtd_mem_idx, rtd_non_idx = load_feature_by_name(args, target_model_root, test_mem_name, 
                                                                                test_non_name, 'ntimes_rtd_loss')

    rtd_mem_idx = rtd_mem_idx.reshape(-1).tolist()
    rtd_non_idx = rtd_non_idx.reshape(-1).tolist()
    
    all_mem = []
    for fea, att, idx in zip(rtd_test_mem_feature, test_rtd_mem_attn, rtd_mem_idx):
        cur_mlm_fea = mlm_mem_idx2_fea[idx]['fea']
        cur_mlm_att = mlm_mem_idx2_fea[idx]['att']
        all_mem.append({
            'mlm_fea': cur_mlm_fea,
            'mlm_att': cur_mlm_att,
            'rtd_fea': fea,
            'rtd_att': att,
            'idx': idx 
        })
    all_non = []
    for fea, att, idx in zip(rtd_test_non_feature, test_rtd_non_attn, rtd_non_idx):
        cur_mlm_fea = mlm_non_idx2_fea[idx]['fea']
        cur_mlm_att = mlm_non_idx2_fea[idx]['att']
        all_non.append({
            'mlm_fea': cur_mlm_fea,
            'mlm_att': cur_mlm_att,
            'rtd_fea': fea,
            'rtd_att': att,
            'idx': idx 
        })  

    dataset = SignalDataset(all_mem, all_non, mode='test', model_type=model_type)
    logger.info('testing ......')
    auc = test(args, dataset, model, more_detail=False)
    logger.info(auc)
