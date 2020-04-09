# Fine-tuning _BlueBERT-Base, Uncased, PubMed_  
Here are the parameters that have performed the best score in each Dev dataset.  
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
`$MODEL_DIR`: where the BERT model is saved  

-----  
