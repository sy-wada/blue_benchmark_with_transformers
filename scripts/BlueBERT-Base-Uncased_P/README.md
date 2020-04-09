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
Only in [BIOSSES](#BIOSSES), we give multiple seeds.  


Please change the following variables to suit your environment:  
`$CODE_DIR`: where you downloaded this repository  
`$DATASET_DIR`: a dataset directory for each task  
`$MODEL_DIR`: where the pre-trained model is saved  

-----  
## Sentence similarity
### MedSTS
| parameter | candidates |
|:----|:----|
| **epochs** | 3, 4, 5, 6, 7, 8, 9, 10, 15 |
| **learning rate** | 1e-5, 2e-5, 3e-5, 4e-5, 5e-5 |

Table: Parameter Candidates Used for Grid Search of MedSTS
### BIOSSES
| parameter | candidates |
|:----|:----|
| **epochs** | 10, 20, 30, 40, 50 |
| **learning rate** | 1e-5, 2e-5, 3e-5, 4e-5, 5e-5 |
| **seed** | 1, 2, 3, 4, 5, 6, 7, 8, 9, 10,<br> 11, 12, 13, 14, 15, 16, 17, 18, 19, 20,<br> 21, 22, 23, 24, 25, 26, 27, 28, 29, 30 |

Table: Parameter Candidates Used for Grid Search of BIOSSES
## Named Entity Recognition
### BC5CDR-disease
| parameter | candidates |
|:----|:----|
| **epochs** | 10, 20, 30 |
| **learning rate** | 1e-5, 2e-5, 3e-5, 4e-5, 5e-5 |

Table: Parameter Candidates Used for Grid Search of BC5CDR-disease
### BC5CDR-chemical
| parameter | candidates |
|:----|:----|
| **epochs** | 10, 20, 30 |
| **learning rate** | 1e-5, 2e-5, 3e-5, 4e-5, 5e-5 |

Table: Parameter Candidates Used for Grid Search of BC5CDR-chemical
### ShARe/CLEFE
| parameter | candidates |
|:----|:----|
| **epochs** | 10, 20, 30 |
| **learning rate** | 1e-5, 2e-5, 3e-5, 4e-5, 5e-5 |

Table: Parameter Candidates Used for Grid Search of ShARe/CLEFE
## Relation Extraction
### DDI
| parameter | candidates |
|:----|:----|
| **epochs** | 3, 4, 5, 6, 7, 8, 9, 10 |
| **learning rate** | 1e-5, 2e-5, 3e-5, 4e-5, 5e-5 |

Table: Parameter Candidates Used for Grid Search of DDI
### ChemProt
| parameter | candidates |
|:----|:----|
| **epochs** | 3, 4, 5, 6, 7, 8, 9, 10 |
| **learning rate** | 1e-5, 2e-5, 3e-5, 4e-5, 5e-5 |

Table: Parameter Candidates Used for Grid Search of ChemProt
### i2b2 2010
| parameter | candidates |
|:----|:----|
| **epochs** | 3, 4, 5, 6, 7, 8, 9, 10 |
| **learning rate** | 1e-5, 2e-5, 3e-5, 4e-5, 5e-5 |

Table: Parameter Candidates Used for Grid Search of i2b2 2010
## Document multilabel classification
### HoC
| parameter | candidates |
|:----|:----|
| **epochs** | 5, 10, 15, 20 |
| **learning rate** | 1e-5, 2e-5, 3e-5, 4e-5, 5e-5 |

Table: Parameter Candidates Used for Grid Search of HoC
## Inference task
### MedNLI
| parameter | candidates |
|:----|:----|
| **epochs** | 3, 4, 5, 6, 7, 8, 9, 10, 15 |
| **learning rate** | 1e-5, 2e-5, 3e-5, 4e-5, 5e-5 |

Table: Parameter Candidates Used for Grid Search of MedNLI
