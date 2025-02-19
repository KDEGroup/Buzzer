import copy
import datasets
from dataclasses import dataclass, field
import torch 
import transformers
from typing import Sequence, Dict, Optional, List, Union
from transformers import TrainingArguments
from omegaconf import DictConfig


IGNORE_INDEX = -100

def build_instruction_prompt(instruction: str):
    return '''
### Instruction:
{}
### Response:
'''.format(instruction.strip()).lstrip()


def get_position_ids(input_ids):
    seq_length = input_ids.shape[1]
    # Position ids.
    position_ids = torch.arange(seq_length, dtype=torch.long)
    return position_ids.unsqueeze(0).expand_as(input_ids)

def create_attention_mask(bs, seq_length, attention_mask_2d):
    mask = torch.tril(torch.ones((bs, seq_length, seq_length))).view(
        bs, 1, seq_length, seq_length
    )
    mask = mask < 0.5
    attention_mask = attention_mask_2d[:, None, None, :].expand(bs, 1, seq_length, seq_length)
    
    # 因为padding，padding部分设为true，此时表示无需注意力
    attn_mask = mask | (~attention_mask)
    attn_mask = torch.where(attn_mask == True, float("-inf"), 0).long()
    
    return attn_mask
    
def generate_data():
    data = datasets.load_dataset('ise-uiuc/Magicoder-Evol-Instruct-110K', 
                                cache_dir='LLM_MIA/data/evol_inst_110k',
                                split='train')
    data = data.shuffle(42)

    mem_pretrain = data.select(range(0, 30000))
    non_shadow = data.select(range(30000, 50000))
    non_caliberate = data.select(range(50000, 70000))
    non_utils = data.select(range(70000, 90000))

    non_test = data.select(range(90000, 95000))
    mem_test = mem_pretrain.select(range(0, 5000))

    wb_mem_train = data.select(range(5000, 30000))
    wb_non_train = non_utils

    for each_data, name in zip(
        [mem_pretrain, non_shadow, non_caliberate, non_utils, non_test, mem_test, wb_mem_train, wb_non_train],
        ['mem_pretrain', 'non_shadow', 'non_caliberate', 'non_utils', 'non_test', 'mem_test', 'wb_mem_train', 'wb_non_train']
    ):
        each_data.save_to_disk(f'LLM_MIA/data/evol_inst_110k/{name}')


def _tokenize_fn(strings: Sequence[str], tokenizer: transformers.PreTrainedTokenizer) -> Dict:
    """Tokenize a list of strings."""
    tokenized_list = [
        tokenizer(
            text,
            return_tensors="pt",
            padding=False,
            max_length=tokenizer.model_max_length,
            add_special_tokens=False,
            truncation=True,
        )
        for text in strings
    ]
    input_ids = labels = [tokenized.input_ids[0] for tokenized in tokenized_list]
    input_ids_lens = labels_lens = [
        tokenized.input_ids.ne(tokenizer.pad_token_id).sum().item() for tokenized in tokenized_list
    ]
    
    return dict(
        input_ids=input_ids,
        labels=labels,
        input_ids_lens=input_ids_lens,
        labels_lens=labels_lens,
    )


def preprocess(
    sources: Sequence[str],
    targets: Sequence[str],
    tokenizer: transformers.PreTrainedTokenizer,
) -> Dict:
    """Preprocess the data by tokenizing."""
    examples = [s + t for s, t in zip(sources, targets)]
    examples_tokenized, sources_tokenized = [_tokenize_fn(strings, tokenizer) for strings in (examples, sources)]
    input_ids = examples_tokenized["input_ids"]

    labels = copy.deepcopy(input_ids)
    
    for label, source_len in zip(labels, sources_tokenized["input_ids_lens"]):
        label[:source_len] = IGNORE_INDEX
    return dict(input_ids=input_ids, labels=labels, prompt_len=sources_tokenized["input_ids_lens"])

def train_tokenize_function(examples, tokenizer):
    sources = [
        build_instruction_prompt(instruction)
        for instruction in examples['instruction']
    ]
    targets = [f"{output}\n{tokenizer.eos_token}" for output in examples['response']]
    data_dict = preprocess(sources, targets, tokenizer)
    return data_dict


# @dataclass
# class DataCollatorForSupervisedDataset(object):
#     """Collate examples for supervised fine-tuning."""
#     tokenizer: transformers.PreTrainedTokenizer

#     def __call__(self, instances: Sequence[Dict]) -> Dict[str, torch.Tensor]:
#         input_ids, labels = tuple([instance[key] for instance in instances] for key in ("input_ids", "labels"))
        
#         input_ids = [torch.tensor(x) for x in input_ids]
#         input_ids = torch.nn.utils.rnn.pad_sequence(
#             input_ids, batch_first=True, padding_value=self.tokenizer.pad_token_id
#         )
#         labels = [torch.tensor(x) for x in labels]
#         labels = torch.nn.utils.rnn.pad_sequence(labels, batch_first=True, padding_value=IGNORE_INDEX)
                
#         return dict(
#             input_ids=input_ids,
#             labels=labels,
#             attention_mask=input_ids.ne(self.tokenizer.pad_token_id),
#         )
        
@dataclass
class DataCollatorForSupervisedDataset(object):
    """Collate examples for supervised fine-tuning."""
    tokenizer: transformers.PreTrainedTokenizer
    args: Union[DictConfig, TrainingArguments]
    pp_format: bool = field(default= False)
    pad: bool = field(default=True)
    
    def __call__(self, instances: Sequence[Dict]) -> Dict[str, torch.Tensor]:
        
        input_ids, labels = tuple([instance[key] for instance in instances] for key in ("input_ids", "labels"))
        
        input_ids_tensor = []
        labels_tensor = []
        max_seq_len = self.args.model_max_length
        for idx, (input_idx, label) in enumerate(zip(input_ids, labels)):
            if self.pad:
                need_pad_len = max_seq_len - len(input_idx)
                input_idx = input_idx + [self.tokenizer.pad_token_id] * need_pad_len
                label = label + [-100] * need_pad_len
            
            input_ids_tensor.append(torch.LongTensor(input_idx).view(1, -1))
            labels_tensor.append(torch.LongTensor(label).view(1, -1))
        
        input_ids = torch.cat(input_ids_tensor)
        labels = torch.cat(labels_tensor)
        attention_mask_2d = input_ids.ne(self.tokenizer.pad_token_id)
        
        if 'prompt_len' in instances[0].keys():
            return_dict = dict(
                input_ids = input_ids,
                labels = labels,
                attention_mask=attention_mask_2d,
                prompt_len=[i['prompt_len'] for i in instances] 
            )
        else:
            return_dict = dict(
                input_ids = input_ids,
                labels = labels,
                attention_mask=attention_mask_2d,
            )
        
        if not self.pp_format:
            return return_dict
            
        position_ids = get_position_ids(input_ids)
        bsz, tgt_len = input_ids.size(0), input_ids.size(1)
        attn_mask = create_attention_mask(bsz, tgt_len, attention_mask_2d)
        
        return (
            (
                input_ids,
                position_ids,
                attn_mask,
            ),
            labels
        )


if __name__ == "__main__":
    generate_data()