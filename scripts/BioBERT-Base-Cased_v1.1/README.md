# Fine-tuning *BioBERT-Base, Cased v1.1 (+ PubMed 1M)*
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
  --fp16
```
Only in [BIOSSES](#biosses), we run it with multiple seeds.  


Please change the following variables to suit your environment:  
`$CODE_DIR`: where you downloaded this repository  
`$DATASET_DIR`: a dataset directory for each task  
`$MODEL_DIR`: where the pre-trained model is saved  
## Overview
| Task                                | learning rate | epochs | seed | dev_score | test_score |
|:------------------------------------|--------------:|-------:|-----:|----------:|-----------:|
| [MedSTS](#medsts)                   | 4e-5          |  8     | 12   |  87.2     | 85.0       |
| [BIOSSES](#biosses)                 | 4e-5          | 50     | 14   |  95.4     | 85.1       |
| [BC5CDR-disease](#bc5cdr-disease)   | 5e-5          | 10     | 12   | 100.0     | 85.6       |
| [BC5CDR-chemical](#bc5cdr-chemical) | 1e-5          | 30     | 12   |  99.9     | 93.2       |
| [ShARe/CLEFE](#shareclefe)          | 5e-5          | 30     | 12   |  98.9     | 76.9       |
| [DDI](#ddi)                         | 5e-5          |  4     | 12   |  85.0     | 79.9       |
| [ChemProt](#chemprot)               | 3e-5          |  8     | 12   |  77.0     | 73.2       |
| [i2b2 2010](#i2b2-2010)             | 5e-5          |  3     | 12   |  70.6     | 72.2       |
| [HoC](#hoc)                         | 5e-5          | 15     | 12   |  89.7     | 85.9       |
| [MedNLI](#mednli)                   | 3e-5          |  6     | 12   |  85.3     | 83.1       |

-----  
## Sentence similarity
### MedSTS
| parameter | candidates |
|:----|:----|
| **epochs** | 3, 4, 5, 6, 7, 8, 9, 10, 15 |
| **learning rate** | 1e-5, 2e-5, 3e-5, 4e-5, 5e-5 |

Table: Parameter Candidates Used for Grid Search of MedSTS.
```bash
python $CODE_DIR/utils/run_sts.py \
  --do_train \
  --do_eval \
  --do_predict \
  --task_name=medsts \
  --data_dir=$DATASET_DIR \
  --model_name_or_path=$MODEL_DIR \
  --output_dir=./output/medsts \
  --learning_rate=4e-5 \
  --num_train_epochs=8 \
  --logging_steps=0 \
  --save_steps=0 \
  --model_type=bert \
  --max_seq_length=128 \
  --per_gpu_train_batch_size=32 \
  --per_gpu_eval_batch_size=32 \
  --gradient_accumulation_steps=1 \
  --seed=12 \
  --eval_every_epoch \
  --overwrite_output_dir \
  --fp16
```
### BIOSSES
| parameter | candidates |
|:----|:----|
| **epochs** | 10, 20, 30, 40, 50 |
| **learning rate** | 1e-5, 2e-5, 3e-5, 4e-5, 5e-5 |
| **seed** | 1, 2, 3, 4, 5, 6, 7, 8, 9, 10,<br> 11, 12, 13, 14, 15, 16, 17, 18, 19, 20 |

Table: Parameter Candidates Used for Grid Search of BIOSSES.
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
  --seed=14 \
  --eval_every_epoch \
  --overwrite_output_dir \
  --fp16
```
## Named Entity Recognition
| parameter | candidates |
|:----|:----|
| **epochs** | 10, 20, 30 |
| **learning rate** | 1e-5, 2e-5, 3e-5, 4e-5, 5e-5 |

Table: Parameter Candidates Used for Grid Search of BC5CDR-disease, BC5CDR-chemical and ShARe/CLEFE.
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
  --num_train_epochs=10 \
  --logging_steps=0 \
  --save_steps=0 \
  --model_type=bert \
  --max_seq_length=128 \
  --per_gpu_train_batch_size=32 \
  --per_gpu_eval_batch_size=32 \
  --gradient_accumulation_steps=1 \
  --seed=12 \
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
  --learning_rate=1e-5 \
  --num_train_epochs=30 \
  --logging_steps=0 \
  --save_steps=0 \
  --model_type=bert \
  --max_seq_length=128 \
  --per_gpu_train_batch_size=32 \
  --per_gpu_eval_batch_size=32 \
  --gradient_accumulation_steps=1 \
  --seed=12 \
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
  --eval_every_epoch \
  --overwrite_output_dir \
  --fp16
```
## Relation Extraction
| parameter | candidates |
|:----|:----|
| **epochs** | 3, 4, 5, 6, 7, 8, 9, 10 |
| **learning rate** | 1e-5, 2e-5, 3e-5, 4e-5, 5e-5 |

Table: Parameter Candidates Used for Grid Search of DDI, ChemProt and i2b2 2010.
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
  --learning_rate=5e-5 \
  --num_train_epochs=4 \
  --logging_steps=0 \
  --save_steps=0 \
  --model_type=bert \
  --max_seq_length=128 \
  --per_gpu_train_batch_size=32 \
  --per_gpu_eval_batch_size=32 \
  --gradient_accumulation_steps=1 \
  --seed=12 \
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
  --learning_rate=3e-5 \
  --num_train_epochs=8 \
  --logging_steps=0 \
  --save_steps=0 \
  --model_type=bert \
  --max_seq_length=128 \
  --per_gpu_train_batch_size=32 \
  --per_gpu_eval_batch_size=32 \
  --gradient_accumulation_steps=1 \
  --seed=12 \
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
  --num_train_epochs=3 \
  --logging_steps=0 \
  --save_steps=0 \
  --model_type=bert \
  --max_seq_length=128 \
  --per_gpu_train_batch_size=32 \
  --per_gpu_eval_batch_size=32 \
  --gradient_accumulation_steps=1 \
  --seed=12 \
  --eval_every_epoch \
  --overwrite_output_dir \
  --fp16
```
## Document multilabel classification
### HoC
| parameter | candidates |
|:----|:----|
| **epochs** | 5, 10, 15, 20 |
| **learning rate** | 1e-5, 2e-5, 3e-5, 4e-5, 5e-5 |

Table: Parameter Candidates Used for Grid Search of HoC
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
  --num_train_epochs=15 \
  --logging_steps=0 \
  --save_steps=0 \
  --model_type=bert \
  --max_seq_length=128 \
  --per_gpu_train_batch_size=32 \
  --per_gpu_eval_batch_size=32 \
  --gradient_accumulation_steps=1 \
  --seed=12 \
  --eval_every_epoch \
  --overwrite_output_dir \
  --fp16
```
## Inference task
### MedNLI
| parameter | candidates |
|:----|:----|
| **epochs** | 3, 4, 5, 6, 7, 8, 9, 10, 15 |
| **learning rate** | 1e-5, 2e-5, 3e-5, 4e-5, 5e-5 |

Table: Parameter Candidates Used for Grid Search of MedNLI
```bash
python $CODE_DIR/utils/run_multi_class_classifier.py \
  --do_train \
  --do_eval \
  --do_predict \
  --task_name=mednli \
  --data_dir=$DATASET_DIR \
  --model_name_or_path=$MODEL_DIR \
  --output_dir=./output/mednli \
  --learning_rate=3e-5 \
  --num_train_epochs=6 \
  --logging_steps=0 \
  --save_steps=0 \
  --model_type=bert \
  --max_seq_length=128 \
  --per_gpu_train_batch_size=32 \
  --per_gpu_eval_batch_size=32 \
  --gradient_accumulation_steps=1 \
  --seed=12 \
  --eval_every_epoch \
  --overwrite_output_dir \
  --fp16
```
