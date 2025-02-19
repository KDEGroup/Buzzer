
from torch.utils.data.dataloader import DataLoader
from torch.utils.data.dataset import Dataset
from transformers import Seq2SeqTrainer, Trainer
import torch
import argparse
from typing import Optional
from torch.utils.data.distributed import DistributedSampler

from data_collator import collate_fn


class CodeTrainer(Seq2SeqTrainer):

    def __init__(self, main_args: argparse.Namespace, tokenizer, task, **kwargs):
        super(CodeTrainer, self).__init__(**kwargs)
        self.main_args = main_args
        self.tokenizer = tokenizer
        self.task = task
        self.isDist = 0
        if torch.cuda.device_count() > 1:
            self.isDist = 1

    def get_train_dataloader(self) -> DataLoader:
        sampler = None
        if self.isDist:
            sampler = DistributedSampler(self.train_dataset)
        
        return DataLoader(dataset=self.train_dataset,
                          sampler=sampler,
                          batch_size=self.main_args.batch_size,
                        #   shuffle=True,
                          collate_fn=lambda batch: collate_fn(batch,
                                                              args=self.main_args,
                                                              task=self.task,
                                                              tokenizer=self.tokenizer))

    def get_eval_dataloader(self, eval_dataset: Optional[Dataset] = None) -> DataLoader:
        if eval_dataset:
            self.eval_dataset = eval_dataset
        
        sampler = None
        if self.isDist:
            sampler = DistributedSampler(self.eval_dataset, shuffle=False)
            
        return DataLoader(dataset=self.eval_dataset,
                          sampler=sampler,
                          batch_size=self.main_args.eval_batch_size,
                          collate_fn=lambda batch: collate_fn(batch,
                                                              args=self.main_args,
                                                              task=self.task,
                                                              tokenizer=self.tokenizer))

    def get_test_dataloader(self, test_dataset: Dataset) -> DataLoader:
        sampler = None
        if self.isDist:
            sampler = DistributedSampler(test_dataset)
            
        return DataLoader(dataset=test_dataset,
                          sampler=sampler,
                          batch_size=self.main_args.eval_batch_size,
                          collate_fn=lambda batch: collate_fn(batch,
                                                              args=self.main_args,
                                                              task=self.task,
                                                              tokenizer=self.tokenizer))

    def set_task(self, task):
        self.task = task
    
    def compute_loss(self, model, inputs, return_outputs=False):
        labels = inputs.pop("labels")
        
        # forward pass
        outputs = model(**inputs)
        # logger.info(outputs)
        
        logits = outputs.get("logits")
        # # compute custom loss (suppose one has 3 labels with different weights)
        shift_logits = logits[:, :-1, :].contiguous()
        shift_labels = labels[:, 1:].contiguous()
        loss_fct = torch.nn.CrossEntropyLoss()
        
        loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
        return (loss, logits) if return_outputs else loss


class CodeCLSTrainer(Trainer):

    def __init__(self, main_args: argparse.Namespace, tokenizer, task, **kwargs):
        super(CodeCLSTrainer, self).__init__(**kwargs)
        self.main_args = main_args
        self.task = task
        self.tokenizer = tokenizer
        self.isDist = 0
        if torch.cuda.device_count() > 1:
            self.isDist = 1

    def get_train_dataloader(self) -> DataLoader:
        sampler = None
        if self.isDist:
            sampler = DistributedSampler(self.train_dataset)
        
        return DataLoader(dataset=self.train_dataset,
                          batch_size=self.main_args.batch_size,
                          sampler=sampler,
                        #   shuffle=True,
                          collate_fn=lambda batch: collate_fn(batch,
                                                              args=self.main_args,
                                                              task=self.task,
                                                              tokenizer=self.tokenizer))

    def get_eval_dataloader(self, eval_dataset: Optional[Dataset] = None) -> DataLoader:
        if eval_dataset:
            self.eval_dataset = eval_dataset
        sampler = None
        if self.isDist:
            sampler = DistributedSampler(self.eval_dataset, shuffle=False)
            
        return DataLoader(dataset=self.eval_dataset,
                          sampler=sampler,
                          batch_size=self.main_args.eval_batch_size,
                          collate_fn=lambda batch: collate_fn(batch,
                                                              args=self.main_args,
                                                              task=self.task,
                                                              tokenizer=self.tokenizer))
    
    def get_test_dataloader(self, test_dataset: Dataset) -> DataLoader:
        sampler = None
        if self.isDist:
            sampler = DistributedSampler(test_dataset, shuffle=False)
            
        return DataLoader(dataset=test_dataset,
                          sampler=sampler,
                          batch_size=self.main_args.eval_batch_size,
                          collate_fn=lambda batch: collate_fn(batch,
                                                              args=self.main_args,
                                                              task=self.task,
                                                              tokenizer=self.tokenizer))

    def set_task(self, task):
        self.task = task

    # def compute_loss(self, model, inputs, return_outputs=False):
    #     labels = inputs.pop("labels")
        
    #     # forward pass
    #     outputs = model(**inputs)        
    #     logits = outputs.get("logits")
    #     loss_fct = torch.nn.CrossEntropyLoss()
        
    #     loss = loss_fct(logits.view(-1, logits.size(-1)), labels.view(-1))
    #     return (loss, logits) if return_outputs else loss