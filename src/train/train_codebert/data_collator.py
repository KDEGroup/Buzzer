
import torch
from typing import List
import itertools
import logging 

logger = logging.getLogger(__name__)

def mask_tokens(inputs, attention_mask, mask_token_index, vocab_size, special_token_indices, mlm_probability=0.15, replace_prob=0.1, orginal_prob=0.1, ignore_index=-100):
    """ 
    Prepare masked tokens inputs/labels for masked language modeling: (1-replace_prob-orginal_prob)% MASK, replace_prob% random, orginal_prob% original within mlm_probability% of tokens in the sentence. 
    * ignore_index in nn.CrossEntropy is default to -100, so you don't need to specify ignore_index in loss
    """

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


def collate_fn(batch, args, task, tokenizer):
    model_inputs = {}
    encoder_input = None 
    mask_token_index = tokenizer.mask_token_id
    vocab_size = tokenizer.vocab_size 
    special_token_indices = [tokenizer.bos_token_id, tokenizer.eos_token_id]
    mlm_mask = None 
    
    if task == 'mlm':
        inputs = batch
    
        encoder_input = tokenizer(text=inputs,
                                    is_split_into_words=True, 
                                    add_special_tokens=True,
                                    padding='max_length', return_tensors='pt',
                                    truncation=True, max_length=512)
        
        input_ids, labels, mlm_mask = mask_tokens(encoder_input['input_ids'], encoder_input['attention_mask'],
                                               mask_token_index, vocab_size, special_token_indices,
                                               replace_prob=0, orginal_prob=0)        

    elif task == 'rtd':
        inputs = batch
        
        encoder_input = tokenizer(text=inputs,
                                    is_split_into_words=True, 
                                    add_special_tokens=True,
                                    padding='max_length', return_tensors='pt',
                                    truncation=True, max_length=512)
        input_ids, labels, mlm_mask = mask_tokens(encoder_input['input_ids'], encoder_input['attention_mask'],
                                            mask_token_index, vocab_size, special_token_indices,
                                            replace_prob=0, orginal_prob=0)
        input_ids = encoder_input['input_ids']
    
    model_inputs['input_ids'] = input_ids
    model_inputs['attention_mask'] = input_ids.ne(tokenizer.pad_token_id)
    model_inputs['labels'] = labels
    model_inputs['mlm_mask'] = mlm_mask
    model_inputs['mlm_labels'] = labels
    return model_inputs
