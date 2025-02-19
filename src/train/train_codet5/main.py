import argparse
import time
from torch.utils.data.dataloader import DataLoader
from torch.utils.data.dataset import Dataset
from transformers import Seq2SeqTrainer, Trainer


import torch.utils.data
from transformers import (Seq2SeqTrainingArguments, IntervalStrategy, SchedulerType, TrainingArguments,
                          T5Config, AutoTokenizer)


import os
from typing import Union, Tuple

from src.train.train_codet5 import enums
from dataset import init_dataset

from t5_model import CodeT5ForClassificationAndGeneration
from trainer import CodeTrainer, CodeCLSTrainer

import logging
from pretrain_args import add_args
from src.common.utils import set_seed

logger = logging.getLogger(__name__)


def pre_train(args,
              trained_model: Union[CodeT5ForClassificationAndGeneration, str] = None,):
    tasks = args.pre_train_tasks
    if tasks is None:
        logger.warning('Was specified for pre-training, but got pre-training tasks to None, '
                       'will default to {}'.format(','.join(enums.PRE_TRAIN_TASKS)))
        tasks = enums.PRE_TRAIN_TASKS
    else:
        supported_tasks = []
        for task in tasks.split(','):
            task = task.strip().lower()
            if task in enums.PRE_TRAIN_TASKS:
                supported_tasks.append(task)
            else:
                logger.warning(f'Pre-training task {task} is not supported and will be ignored.')
        tasks = supported_tasks

    
    assert not trained_model or \
        isinstance(trained_model, str) or \
        isinstance(trained_model, CodeT5ForClassificationAndGeneration), \
        f'The model type is not supported, expect Bart model or string of model dir, got {type(trained_model)}'

    logger.info('*' * 100)
    logger.info('Initializing pre-training environments')

    logger.info('-' * 100)
    logger.info('Loading and parsing datasets')
    
    dataset = init_dataset(args=args, mode=args.dataset_mode)

    if args.do_eval:
        train_size = int(0.99 * len(dataset))
        test_size = len(dataset) - train_size
        train_dataset, valid_dataset = torch.utils.data.random_split(dataset, [train_size, test_size])
        logger.info(f'train set length:{len(train_dataset)}')
        logger.info(f'valid set lenth:{len(valid_dataset)}')
    else:
        train_dataset = dataset    
    
    logger.info(f'The size of pre_training set: {len(dataset)}')
    
    if args.pre_train_subset_ratio:
        dataset = dataset.subset(args.pre_train_subset_ratio)
        logger.info(f'The pre-train dataset is trimmed to subset due to the argument: '
                    f'train_subset_ratio={args.pre_train_subset_ratio}')
        logger.info('The size of trimmed pre-train set: {}'.format(len(dataset)))
        
    logger.info('Datasets loaded and parsed successfully')

    logger.info('-' * 100)
    

    logger.info('-' * 100)
    logger.info('Building model')
    
    config = T5Config.from_pretrained('Salesforce/codet5-base')
    config.num_labels = 2
    
    model = CodeT5ForClassificationAndGeneration(config)
    tokenizer = AutoTokenizer.from_pretrained('Salesforce/codet5-base', add_prefix_space=True)
    logger.info(tasks)
    for task in tasks:
        logger.info('-' * 100)
        logger.info(f'Pre-training task: {task.upper()}')
        
        dataset.set_task(task)
        if task == enums.TASK_MASK_SPAN_PREDICTION:
            logger.info('-' * 100)
            model.set_model_mode(enums.MODEL_MODE_GEN)

            logger.info('-' * 100)
            logger.info('Initializing the running configurations')
            training_args = Seq2SeqTrainingArguments(output_dir=os.path.join(args.pre_train_output_root, task),
                                                     overwrite_output_dir=True,
                                                     do_train=True,
                                                     do_eval=args.do_eval,
                                                     per_device_train_batch_size=args.batch_size,
                                                     gradient_accumulation_steps=args.gradient_accumulation_steps,
                                                     learning_rate=args.learning_rate,
                                                     weight_decay=args.lr_decay_rate,
                                                     max_grad_norm=args.grad_clipping_norm,
                                                     num_train_epochs=30,
                                                    #  num_train_epochs=1,
                                                     lr_scheduler_type=SchedulerType.LINEAR,
                                                     warmup_steps=args.warmup_steps,
                                                     logging_dir=os.path.join(args.tensor_board_root, task),
                                                     logging_strategy=IntervalStrategy.STEPS,
                                                     logging_steps=args.logging_steps,
                                                     save_strategy=IntervalStrategy.STEPS,
                                                     save_total_limit=3,
                                                     save_steps=args.save_steps,
                                                     eval_steps=args.eval_steps,
                                                     evaluation_strategy='steps' if args.do_eval else 'no',
                                                     seed=42,
                                                     fp16=args.fp16,
                                                     dataloader_drop_last=False,
                                                     run_name=args.model_name,
                                                     load_best_model_at_end=True,
                                                     ignore_data_skip=False,
                                                     label_smoothing_factor=args.label_smoothing,
                                                     report_to=['tensorboard'],
                                                     dataloader_pin_memory=True)
            trainer = CodeTrainer(model=model,
                                     task=task,
                                     main_args=args,
                                     args=training_args,
                                     data_collator=None,
                                     train_dataset=train_dataset,
                                     eval_dataset=valid_dataset if args.do_eval else None,
                                     tokenizer=tokenizer,
                                     model_init=None,
                                     compute_metrics=None,
                                     callbacks=None)
            logger.info('Running configurations initialized successfully')

            logger.info('-' * 100)
            logger.info(f'Start pre-training task: {task}')
            cap_result = trainer.train()
            logger.info(f'Pre-training task {task} finished')
            trainer.save_model(os.path.join(args.model_root, task))

        elif task == enums.TASK_IDENTIFIER_TAGGING:
            # set model mode
            logger.info('-' * 100)
            model.set_model_mode(enums.MODEL_MODE_CLS)

            logger.info('-' * 100)
            logger.info('Initializing the running configurations')
            training_args = TrainingArguments(output_dir=os.path.join(args.pre_train_output_root, task),
                                                     overwrite_output_dir=True,
                                                     do_train=True,
                                                     do_eval=args.do_eval,
                                                     per_device_train_batch_size=args.batch_size,
                                                     gradient_accumulation_steps=args.gradient_accumulation_steps,
                                                     learning_rate=args.learning_rate,
                                                     weight_decay=args.lr_decay_rate,
                                                     max_grad_norm=args.grad_clipping_norm,
                                                     num_train_epochs=30,
                                                    #  num_train_epochs=1,
                                                     lr_scheduler_type=SchedulerType.LINEAR,
                                                     warmup_steps=args.warmup_steps,
                                                     logging_dir=os.path.join(args.tensor_board_root, task),
                                                     logging_strategy=IntervalStrategy.STEPS,
                                                     logging_steps=args.logging_steps,
                                                     save_strategy=IntervalStrategy.STEPS,
                                                     save_total_limit=3,
                                                     save_steps=args.save_steps,
                                                     eval_steps=args.eval_steps,
                                                     evaluation_strategy='steps' if args.do_eval else 'no',
                                                     seed=args.random_seed,
                                                     fp16=args.fp16,
                                                     dataloader_drop_last=False,
                                                     run_name=args.model_name,
                                                     load_best_model_at_end=True,
                                                     ignore_data_skip=False,
                                                     label_smoothing_factor=args.label_smoothing,
                                                     report_to=['tensorboard'],
                                                     dataloader_pin_memory=True)
            
            trainer = CodeCLSTrainer(model=model,
                                     task=task,
                                     main_args=args,
                                     args=training_args,
                                     data_collator=None,
                                     train_dataset=train_dataset,
                                     eval_dataset=valid_dataset if args.do_eval else None,
                                     tokenizer=tokenizer,
                                     model_init=None,
                                    #  compute_loss=None,
                                     compute_metrics=None,
                                     callbacks=None)
            logger.info('Running configurations initialized successfully')

            logger.info('-' * 100)
            logger.info(f'Start pre-training task: {task}')
            logger.info('Device: {}'.format(next(model.parameters()).device))
            mass_result = trainer.train()
            logger.info(f'Pre-training task {task} finished')
            trainer.save_model(os.path.join(args.model_root, task))

        elif task == enums.TASK_MASK_IDENTIFER_PREDICTION:
            logger.info('-' * 100)
            model.set_model_mode(enums.MODEL_MODE_GEN)

            logger.info('-' * 100)
            logger.info('Initializing the running configurations')
            training_args = Seq2SeqTrainingArguments(output_dir=os.path.join(args.pre_train_output_root, task),
                                                     overwrite_output_dir=True,
                                                     do_train=True,
                                                     do_eval=args.do_eval,
                                                     per_device_train_batch_size=args.batch_size,
                                                     gradient_accumulation_steps=args.gradient_accumulation_steps,
                                                     learning_rate=args.learning_rate,
                                                     weight_decay=args.lr_decay_rate,
                                                     max_grad_norm=args.grad_clipping_norm,
                                                     num_train_epochs=30,
                                                    #  num_train_epochs=1,
                                                     lr_scheduler_type=SchedulerType.LINEAR,
                                                     warmup_steps=args.warmup_steps,
                                                     logging_dir=os.path.join(args.tensor_board_root, task),
                                                     logging_strategy=IntervalStrategy.STEPS,
                                                     logging_steps=args.logging_steps,
                                                     save_strategy=IntervalStrategy.STEPS,
                                                     save_total_limit=3,
                                                     save_steps=args.save_steps,
                                                     eval_steps=args.eval_steps,
                                                     evaluation_strategy='steps' if args.do_eval else 'no',
                                                     seed=args.random_seed,
                                                     fp16=args.fp16,
                                                     dataloader_drop_last=False,
                                                     run_name=args.model_name,
                                                     load_best_model_at_end=True,
                                                     ignore_data_skip=False,
                                                     label_smoothing_factor=args.label_smoothing,
                                                     report_to=['tensorboard'],
                                                     dataloader_pin_memory=True)
            trainer = CodeTrainer(model=model,
                                     task=task,
                                     main_args=args,
                                     args=training_args,
                                     data_collator=None,
                                     train_dataset=train_dataset,
                                     eval_dataset=valid_dataset if args.do_eval else None,
                                     tokenizer=tokenizer,
                                     model_init=None,
                                     compute_metrics=None,
                                     callbacks=None)
            
            logger.info('Running configurations initialized successfully')

            logger.info('-' * 100)
            logger.info(f'Start pre-training task: {task}')
            mnp_result = trainer.train()
            logger.info(f'Pre-training task {task} finished')
            trainer.save_model(os.path.join(args.model_root, task))
            
        elif task == enums.TASK_BIMODAL_DUAL_GENERATION:
            logger.info('-' * 100)
            
            model.set_model_mode(enums.MODEL_MODE_GEN)
            
            logger.info('-' * 100)
            logger.info('Initializing the running configurations')
            training_args = Seq2SeqTrainingArguments(output_dir=os.path.join(args.pre_train_output_root, task),
                                                     overwrite_output_dir=True,
                                                     do_train=True,
                                                     do_eval=args.do_eval,
                                                     per_device_train_batch_size=args.batch_size,
                                                     gradient_accumulation_steps=args.gradient_accumulation_steps,
                                                     learning_rate=args.learning_rate,
                                                     weight_decay=args.lr_decay_rate,
                                                     max_grad_norm=args.grad_clipping_norm,
                                                     num_train_epochs=50,
                                                    #  num_train_epochs=1,
                                                     lr_scheduler_type=SchedulerType.LINEAR,
                                                     warmup_steps=args.warmup_steps,
                                                     logging_dir=os.path.join(args.tensor_board_root, task),
                                                     logging_strategy=IntervalStrategy.STEPS,
                                                     logging_steps=args.logging_steps,
                                                     save_strategy=IntervalStrategy.STEPS,
                                                     save_total_limit=3,
                                                     save_steps=args.save_steps,
                                                     eval_steps=args.eval_steps,
                                                     evaluation_strategy='steps' if args.do_eval else 'no',
                                                     seed=args.random_seed,
                                                     fp16=args.fp16,
                                                     dataloader_drop_last=False,
                                                     run_name=args.model_name,
                                                     load_best_model_at_end=True,
                                                     ignore_data_skip=False,
                                                     label_smoothing_factor=args.label_smoothing,
                                                     report_to=['tensorboard'],
                                                     dataloader_pin_memory=True)
            trainer = CodeTrainer(model=model,
                         task=task,
                         main_args=args,
                         args=training_args,
                         data_collator=None,
                         train_dataset=train_dataset,
                         eval_dataset=valid_dataset if args.do_eval else None,
                         tokenizer=tokenizer,
                         model_init=None,
                         compute_metrics=None,
                         callbacks=None)
            logger.info('Running configurations initialized successfully')

            logger.info('-' * 100)
            logger.info(f'Start pre-training task: {task}')
            mnp_result = trainer.train()
            logger.info(f'Pre-training task {task} finished')
            trainer.save_model(os.path.join(args.model_root, task))

    logger.info('Pre-training finished')
    trainer.save_model(os.path.join(args.model_root, 'final'))

    return model


if __name__ == "__main__":
    
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.register('type', 'bool', lambda v: v.lower() in ['yes', 'true', 't', '1', 'y'])

    add_args(parser)

    main_args = parser.parse_args()
    
    set_seed()

    main_args.output_root = os.path.join(
        '.',
        'output/codet5_pretrain',
        '{}'.format(main_args.model_name))
    main_args.pre_train_output_root = os.path.join(main_args.output_root, 'pre_train')
    main_args.checkpoint_root = os.path.join(main_args.output_root, 'checkpoints')
    main_args.model_root = os.path.join(main_args.output_root, 'models')
    main_args.vocab_root = os.path.join(main_args.output_root, 'vocabs')
    main_args.tensor_board_root = os.path.join(main_args.output_root, 'runs')
    for d in [main_args.checkpoint_root, main_args.model_root, main_args.vocab_root, main_args.tensor_board_root,
              main_args.dataset_save_dir, main_args.vocab_save_dir]:
        if not os.path.exists(d):
            os.makedirs(d)

    logger = logging.getLogger()
    logger.setLevel(level=logging.DEBUG)

    console = logging.StreamHandler()
    console.setLevel(level=logging.INFO)
    logger.addHandler(console)

    file = logging.FileHandler(os.path.join(main_args.output_root, 'info.log'))
    file.setLevel(level=logging.DEBUG)
    formatter = logging.Formatter('[%(asctime)s | %(filename)s | line %(lineno)d] - %(levelname)s: %(message)s')
    file.setFormatter(formatter)
    logger.addHandler(file)
    
    pre_train(main_args)