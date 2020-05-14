# Fine-tuning *BlueBERT-Base, Uncased, PubMed*
Here, we show a combination of arguments that have performed the best score of each Dev dataset in our experimental environment.  
The common arguments for all tasks are as follows:  
```bash
#!/bin/bash
export CUDA_VISIBLE_DEVICES=0 # means n_gpu=1

  --model_type=bert \
  --max_seq_length=128 \
  --per_gpu_train_batch_size=32 \
  --per_gpu_eval_batch_size=32 \
  --gradient_accumulation_steps=1 \
  --seed=12 \
  --do_lower_case \
  --fp16
```


Please change the following variables to suit your environment:  
`$CODE_DIR`: where you downloaded this repository  
`$DATASET_DIR`: a dataset directory for each task  
`$MODEL_DIR`: where the pre-trained model is saved  
## Overview
| Task                                | learning rate | epochs | dev_score | test_score |
|:------------------------------------|--------------:|-------:|----------:|-----------:|
| [MedSTS](#medsts)                   | 5e-5          |  9     |  87.2     | 85.3       |
| [BIOSSES](#biosses)                 | 4e-5          | 50     |  91.5     | 88.5       |
| [BC5CDR-disease](#bc5cdr-disease)   | 5e-5          | 30     | 100.0     | 86.2       |
| [BC5CDR-chemical](#bc5cdr-chemical) | 5e-5          | 30     |  99.9     | 93.5       |
| [ShARe/CLEFE](#shareclefe)          | 4e-5          | 30     |  98.9     | 77.7       |
| [DDI](#ddi)                         | 3e-5          |  5     |  85.9     | 81.2       |
| [ChemProt](#chemprot)               | 5e-5          |  5     |  76.4     | 73.5       |
| [i2b2 2010](#i2b2-2010)             | 5e-5          |  5     |  63.2     | 74.2       |
| [HoC](#hoc)                         | 5e-5          |  5     |  87.7     | 86.2       |
| [MedNLI](#mednli)                   | 4e-5          | 15     |  84.8     | 82.7       |

-----  
## Sentence similarity
### MedSTS
```bash
python $CODE_DIR/utils/run_sts.py \
  --do_train \
  --do_eval \
  --do_predict \
  --task_name=medsts \
  --data_dir=$DATASET_DIR \
  --model_name_or_path=$MODEL_DIR \
  --output_dir=./output/medsts \
  --learning_rate=5e-5 \
  --num_train_epochs=9 \
  --logging_steps=0 \
  --save_steps=0 \
  --model_type=bert \
  --max_seq_length=128 \
  --per_gpu_train_batch_size=32 \
  --per_gpu_eval_batch_size=32 \
  --gradient_accumulation_steps=1 \
  --seed=12 \
  --do_lower_case \
  --eval_every_epoch \
  --overwrite_output_dir \
  --fp16
```
### BIOSSES
```bash
python $CODE_DIR/utils/run_sts.py \
  --do_train \
  --do_eval \
  --do_predict \
  --task_name=biosses \
  --data_dir=$DATASET_DIR \
  --model_name_or_path=$MODEL_DIR \
  --output_dir=./output/biosses \
  --learning_rate=4e-5 \
  --num_train_epochs=50 \
  --logging_steps=0 \
  --save_steps=0 \
  --model_type=bert \
  --max_seq_length=128 \
  --per_gpu_train_batch_size=32 \
  --per_gpu_eval_batch_size=32 \
  --gradient_accumulation_steps=1 \
  --seed=12 \
  --do_lower_case \
  --eval_every_epoch \
  --overwrite_output_dir \
  --fp16
```
## Named Entity Recognition
### BC5CDR-disease
```bash
python $CODE_DIR/utils/run_ner.py \
  --do_train \
  --do_eval \
  --do_predict \
  --task_name=bc5cdr \
  --data_dir=$DATASET_DIR \
  --model_name_or_path=$MODEL_DIR \
  --output_dir=./output/bc5cdr_disease \
  --learning_rate=5e-5 \
  --num_train_epochs=30 \
  --logging_steps=0 \
  --save_steps=0 \
  --model_type=bert \
  --max_seq_length=128 \
  --per_gpu_train_batch_size=32 \
  --per_gpu_eval_batch_size=32 \
  --gradient_accumulation_steps=1 \
  --seed=12 \
  --do_lower_case \
  --eval_every_epoch \
  --overwrite_output_dir \
  --fp16
```
### BC5CDR-chemical
```bash
python $CODE_DIR/utils/run_ner.py \
  --do_train \
  --do_eval \
  --do_predict \
  --task_name=bc5cdr \
  --data_dir=$DATASET_DIR \
  --model_name_or_path=$MODEL_DIR \
  --output_dir=./output/bc5cdr_chem \
  --learning_rate=5e-5 \
  --num_train_epochs=30 \
  --logging_steps=0 \
  --save_steps=0 \
  --model_type=bert \
  --max_seq_length=128 \
  --per_gpu_train_batch_size=32 \
  --per_gpu_eval_batch_size=32 \
  --gradient_accumulation_steps=1 \
  --seed=12 \
  --do_lower_case \
  --eval_every_epoch \
  --overwrite_output_dir \
  --fp16
```
### ShARe/CLEFE
```bash
python $CODE_DIR/utils/run_ner.py \
  --do_train \
  --do_eval \
  --do_predict \
  --task_name=clefe \
  --data_dir=$DATASET_DIR \
  --model_name_or_path=$MODEL_DIR \
  --output_dir=./output/clefe \
  --learning_rate=4e-5 \
  --num_train_epochs=30 \
  --logging_steps=0 \
  --save_steps=0 \
  --model_type=bert \
  --max_seq_length=128 \
  --per_gpu_train_batch_size=32 \
  --per_gpu_eval_batch_size=32 \
  --gradient_accumulation_steps=1 \
  --seed=12 \
  --do_lower_case \
  --eval_every_epoch \
  --overwrite_output_dir \
  --fp16
```
## Relation Extraction
### DDI
```bash
python $CODE_DIR/utils/run_multi_class_classifier.py \
  --do_train \
  --do_eval \
  --do_predict \
  --task_name=ddi2013 \
  --data_dir=$DATASET_DIR \
  --model_name_or_path=$MODEL_DIR \
  --output_dir=./output/ddi2013 \
  --learning_rate=3e-5 \
  --num_train_epochs=5 \
  --logging_steps=0 \
  --save_steps=0 \
  --model_type=bert \
  --max_seq_length=128 \
  --per_gpu_train_batch_size=32 \
  --per_gpu_eval_batch_size=32 \
  --gradient_accumulation_steps=1 \
  --seed=12 \
  --do_lower_case \
  --eval_every_epoch \
  --overwrite_output_dir \
  --fp16
```
### ChemProt
```bash
python $CODE_DIR/utils/run_multi_class_classifier.py \
  --do_train \
  --do_eval \
  --do_predict \
  --task_name=chemprot \
  --data_dir=$DATASET_DIR \
  --model_name_or_path=$MODEL_DIR \
  --output_dir=./output/chemprot \
  --learning_rate=5e-5 \
  --num_train_epochs=5 \
  --logging_steps=0 \
  --save_steps=0 \
  --model_type=bert \
  --max_seq_length=128 \
  --per_gpu_train_batch_size=32 \
  --per_gpu_eval_batch_size=32 \
  --gradient_accumulation_steps=1 \
  --seed=12 \
  --do_lower_case \
  --eval_every_epoch \
  --overwrite_output_dir \
  --fp16
```
### i2b2 2010
```bash
python $CODE_DIR/utils/run_multi_class_classifier.py \
  --do_train \
  --do_eval \
  --do_predict \
  --task_name=i2b2_2010 \
  --data_dir=$DATASET_DIR \
  --model_name_or_path=$MODEL_DIR \
  --output_dir=./output/i2b2_2010 \
  --learning_rate=5e-5 \
  --num_train_epochs=5 \
  --logging_steps=0 \
  --save_steps=0 \
  --model_type=bert \
  --max_seq_length=128 \
  --per_gpu_train_batch_size=32 \
  --per_gpu_eval_batch_size=32 \
  --gradient_accumulation_steps=1 \
  --seed=12 \
  --do_lower_case \
  --eval_every_epoch \
  --overwrite_output_dir \
  --fp16
```
## Document multilabel classification
### HoC
```bash
python $CODE_DIR/utils/run_multi_label_classifier.py \
  --do_train \
  --do_eval \
  --do_predict \
  --output_all_logits \
  --task_name=hoc \
  --data_dir=$DATASET_DIR \
  --model_name_or_path=$MODEL_DIR \
  --output_dir=./output/hoc \
  --learning_rate=5e-5 \
  --num_train_epochs=5 \
  --logging_steps=0 \
  --save_steps=0 \
  --model_type=bert \
  --max_seq_length=128 \
  --per_gpu_train_batch_size=32 \
  --per_gpu_eval_batch_size=32 \
  --gradient_accumulation_steps=1 \
  --seed=12 \
  --do_lower_case \
  --eval_every_epoch \
  --overwrite_output_dir \
  --fp16
```
## Inference task
### MedNLI
```bash
python $CODE_DIR/utils/run_multi_class_classifier.py \
  --do_train \
  --do_eval \
  --do_predict \
  --task_name=mednli \
  --data_dir=$DATASET_DIR \
  --model_name_or_path=$MODEL_DIR \
  --output_dir=./output/mednli \
  --learning_rate=4e-5 \
  --num_train_epochs=15 \
  --logging_steps=0 \
  --save_steps=0 \
  --model_type=bert \
  --max_seq_length=128 \
  --per_gpu_train_batch_size=32 \
  --per_gpu_eval_batch_size=32 \
  --gradient_accumulation_steps=1 \
  --seed=12 \
  --do_lower_case \
  --eval_every_epoch \
  --overwrite_output_dir \
  --fp16
```
