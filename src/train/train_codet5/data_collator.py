
import torch
from src.train.train_codet5 import enums
import logging 

logger = logging.getLogger(__name__)

def collate_fn(batch, args, task, tokenizer):
    model_inputs = {}
    # cap
    encoder_input = None 
    decoder_input = None 
    model_labels = None
    special_token_indices = [tokenizer.bos_token_id, tokenizer.eos_token_id]
    
    if task == enums.TASK_MASK_SPAN_PREDICTION:
        inputs, labels = map(list, zip(*batch))
                
        encoder_input = tokenizer(text=inputs,
                                    is_split_into_words=True, 
                                    add_special_tokens=True,
                                    padding='max_length', 
                                    truncation=True, max_length=512)
        
        decoder_input = tokenizer(text=labels,
                                    is_split_into_words=True, 
                                    add_special_tokens=True,
                                    padding='max_length', 
                                    truncation=True, max_length=256)
    
    
    elif task == enums.TASK_IDENTIFIER_TAGGING:
        inputs, labels = map(list, zip(*batch))
        encoder_input = tokenizer(text=inputs, is_split_into_words=True, add_special_tokens=True,
                          padding='max_length', truncation=True, max_length=512)
        '''
        https://huggingface.co/docs/transformers/tasks/token_classification
        '''
        
        output_labels = []
        for i, label in enumerate(labels):
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


    elif task == enums.TASK_MASK_IDENTIFER_PREDICTION:
        inputs, labels = map(list, zip(*batch))
        
        encoder_input = tokenizer(text=inputs, 
                                    is_split_into_words=True, 
                                    add_special_tokens=True,
                                    padding='max_length', 
                                    truncation=True, max_length=512)
        decoder_input = tokenizer(text=labels,
                                is_split_into_words=True, 
                                add_special_tokens=True,
                                padding='max_length', 
                                truncation=True, max_length=256)


    elif task == enums.TASK_BIMODAL_DUAL_GENERATION:
        text_a, text_b = map(list, zip(*batch))
        
        encoder_input = tokenizer(text=text_a,
                                    is_split_into_words=True, 
                                    add_special_tokens=True,
                                    padding='max_length', 
                                    truncation=True, max_length=512)
        
        decoder_input = tokenizer(text=text_b,
                                is_split_into_words=True, 
                                add_special_tokens=True,
                                padding='max_length',
                                truncation=True, max_length=256)
        
    model_inputs['input_ids'] = torch.tensor(encoder_input['input_ids'])
    model_inputs['attention_mask'] = torch.tensor(encoder_input['attention_mask'])
    
    if decoder_input is not None:
        model_inputs['decoder_input_ids'] = torch.tensor(decoder_input['input_ids'])
        model_inputs['decoder_attention_mask'] = torch.tensor(decoder_input['attention_mask'])
        model_inputs['labels'] = torch.tensor(decoder_input['input_ids'])
        model_inputs['labels'][model_inputs['labels'] == tokenizer.pad_token_id] = -100
        
    if model_labels is not None:
        model_inputs['labels'] = torch.tensor(model_labels)
        
        special_tokens_mask = torch.full(model_inputs['input_ids'].shape, False, dtype=torch.bool)
        for sp_id in special_token_indices:
            special_tokens_mask = special_tokens_mask | (model_inputs['input_ids']==sp_id)
        model_inputs['labels'][special_tokens_mask] = -100
        
    return model_inputs
