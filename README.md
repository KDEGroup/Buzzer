# Buzzer

Source code for EMNLP 2024 Findings paper: Code Membership Inference for Detecting Unauthorized Data Use in Code Pre-trained Language Models.

### Step 1: pretrain model
- pretrain target model
- pretrain shadow model
- pretrain calibrate model

```shell
bash ./script/pretrain_{target_model}.sh
```

### Step 2: extract model loss
- target model loss
- shadow model loss

```shell
bash ./script/sequence_feature_{target_model}.sh
```

### Step 3: classification
#### white-box inference
Train on target model training data, and test on target model testing data.

#### black-box inference
Train on shadow model training data, and test on target model testing data.


```shell
bash ./script/{target_model}_mia.sh
```
