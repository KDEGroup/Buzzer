
import os
import numpy as np
import torch
from loguru import logger
from src.common.evaluate_metrics import plot_roc
from sklearn.metrics import roc_auc_score


def test(args, dataset, model, more_detail=False, **kwargs):
    dataloader = torch.utils.data.DataLoader(dataset, shuffle=False, batch_size=args.batch_size, collate_fn=dataset.collate)
    
    model = model.cuda()
    model.eval()
    # model.return_only_loss = False
    scores, labels, = [], []
    mem, non = [], []
    idxs = []

    for batch_idx, batch in enumerate(dataloader):
        dic = batch
        label = batch.pop('labels')
        for k, v in dic.items():
            v = v.cuda()

        with torch.no_grad():
            logits = model(**dic)
            scores.append(logits.view(-1, 1).cpu())
            labels.append(label.view(-1, 1).cpu())
            
    scores = np.vstack(scores)
    labels = np.vstack(labels)
    
    for s, l in zip(scores.reshape(-1).tolist(), labels.reshape(-1).tolist()):
        if l == 1:
            mem.append(s)
        if l == 0:
            non.append(s)
            
    auc = roc_auc_score(labels, scores)
    metrics = plot_roc(1, mem, non)
    logger.info(metrics)
    
    if more_detail:
        mem_idx_score, non_idx_score = [], []
        scores = scores.reshape(-1).tolist()
        labels = labels.reshape(-1).tolist()
        
        for i in range(len(idxs)):
            if labels[i] == 1:
                mem_idx_score.append((idxs[i], scores[i]))
            else:
                non_idx_score.append((idxs[i], scores[i]))
        
        holder = {}
        holder['mem'] = mem_idx_score
        holder['non'] = non_idx_score
        
        if args.mia_type == 'gb':
            mt = 'gb'
        else:
            mt = 'wb'
        if args.use_cal:
            flag = 'cal'
        else:
            flag = 'non_cal'
        np.save(open(f'outputs/analysis/{args.target_model}_score_idx_{mt}_{flag}.npy', 'wb'), holder, allow_pickle=True)
    
    return auc 


def save_model(model, path):
    model_to_save = model.module if hasattr(model, 'module') else model
    torch.save(model_to_save.state_dict(), path)
    

def train(args, train_set, model, saved_root, eval_set=None, log_steps=1, eval_steps=100):
    args = args
    dataloader = torch.utils.data.DataLoader(train_set, shuffle=True, batch_size=args.batch_size, collate_fn=train_set.collate)
    optim = torch.optim.Adam(model.parameters(), lr=0.0001)
    
    model = model.cuda()
    total_loss, step = 0, 0
    logging_steps = 100
    best_auc = 0
    model.train()
    
    for epoch_id in range(1):
        for batch_idx, batch in enumerate(dataloader):
            dic = batch
                        
            for k, v in dic.items():
                v = v.cuda()
                
            optim.zero_grad()
            loss = model(**dic)
            loss.backward()
            optim.step()
            
            step += 1
            total_loss += loss.item()
            
            if (step + 1) % log_steps == 0:
                logger.info(f'epoch_id:{epoch_id} batch_id:{batch_idx} loss:{total_loss / step}')
            
            if eval_set is not None and (step + 1) % eval_steps == 0:
                logger.info('evaluating ...................')
                auc = test(args, eval_set, model)
                logger.info(f'epoch_id:{epoch_id} batch_id:{batch_idx} evaluate auc:{auc}')
                if auc > best_auc:
                    best_auc = auc
                    save_model(model, os.path.join(saved_root, 'best_model.bin'))
                model.train()
                # model.return_only_loss = True
            
    return model