# BLUE Benchmark with Transformers

**\*\*\*\*\* New April th, 2020: released \*\*\*\*\***

-----  
[Biomedical Language Understanding Evaluation (BLUE) benchmark](https://github.com/ncbi-nlp/BLUE_Benchmark) is a collection of resources for evaluating and analyzing biomedical
natural language representation models ([Peng et al., 2019](#ypeng)).  
This repository provides our implementation of fine-tuning for BLUE benchmark with [🤗Transformers](https://github.com/huggingface/transformers).  
Our models will be available soon.
## Preparations
1. Download the benchmark dataset from https://github.com/ncbi-nlp/BLUE_Benchmark
1. Save  pre-trained models to your directory. For example, [BioBERT](https://github.com/dmis-lab/biobert), [clinicalBERT](https://github.com/EmilyAlsentzer/clinicalBERT), [SciBERT](https://github.com/allenai/scibert), [BlueBERT](https://github.com/ncbi-nlp/bluebert) and so on.
1. Try to use our code in [utils](./utils). Examples of the command can be found in [scripts](./scripts).
### Tips
If you download Tensorflow models, converting them into PyTorch ones comforts your fine-tuning.  
[Converting Tensorflow Checkpoints](https://huggingface.co/transformers/converting_tensorflow_models.html)
```bash
export BERT_BASE_DIR=/path/to/bert/uncased_L-12_H-768_A-12

transformers-cli convert --model_type bert \
  --tf_checkpoint $BERT_BASE_DIR/bert_model.ckpt \
  --config $BERT_BASE_DIR/bert_config.json \
  --pytorch_dump_output $BERT_BASE_DIR/pytorch_model.bin
```
## Our models  
will be released as soon as they are ready...

## Results
now constructing...

-----  
## Table of Contents
* [Preparations](#preparations)
* [Our models](#our-models)
* [Results](#results)
* [BLUE Tasks](#blue-tasks)
* [Sentence similarity](#sentence-similarity)
  + [MedSTS](#medsts)
  + [BIOSSES](#biosses)
* [Named entity recognition](#named-entity-recognition)
  + [BC5CDR-disease](#bc5cdr-disease)
  + [BC5CDR-chemical](#bc5cdr-chemical)
  + [ShARe/CLEFE](#shareclefe)
* [Relation extraction](#relation-extraction)
  + [DDI](#ddi)
  + [ChemProt](#chemprot)
  + [i2b2 2010](#i2b2-2010)
* [Document multilabel classification](#document-multilabel-classification)
  + [HoC](#hoc)
* [Inference task](#inference-task)
  + [MedNLI](#mednli)
* [Citing](#citing)
* [Acknowledgments](#acknowledgments)
* [References](#references)

## BLUE Tasks
| Corpus          | Train |  Dev | Test | Task                    | Metrics             | Domain     |
|-----------------|------:|-----:|-----:|-------------------------|---------------------|------------|
| MedSTS          |   675 |   75 |  318 | Sentence similarity     | Pearson             | Clinical   |
| BIOSSES         |    64 |   16 |   20 | Sentence similarity     | Pearson             | Biomedical |
| BC5CDR-disease  |  4182 | 4244 | 4424 | Named entity recognition| F1                  | Biomedical |
| BC5CDR-chemical |  5205 | 5347 | 5386 | Named entity recognition| F1                  | Biomedical |
| ShARe/CLEFE     |  5151 | 1176 | 5623 | Named entity recognition| F1                  | Clinical   |
| DDI             |  2937 | 1004 |  979 | Relation extraction     | micro F1            | Biomedical |
| ChemProt        |  4154 | 2416 | 3458 | Relation extraction     | micro F1            | Biomedical |
| i2b2-2010       |  3110 |   11 | 6293 | Relation extraction     | micro F1            | Clinical   |
| HoC             |  1108 |  157 |  315 | Document classification | F1                  | Biomedical |
| MedNLI          | 11232 | 1395 | 1422 | Inference               | accuracy            | Clinical   |

Table: BLUE tasks.  
Note: Some of the figures in the table may differ from those in [the reference](#ypeng). We will explain them later.  

## Sentence similarity
### MedSTS
![MedSTS_hist](./img/medsts.png)
### BIOSSES
![BIOSSES_hist](./img/biosses.png)
## Named entity recognition

### BC5CDR-disease
|tag of tokens   | Train |  Dev |  Test |
|----------------|------:|-----:|------:|
|starting with B |  4182 | 4244 |  4424 |
|starting with I |     0 |    0 |     0 |
|I next to O     |     0 |    0 |     0 |
|**Total**       |  4182 | 4244 |  4424 |

### BC5CDR-chemical
|tag of tokens   | Train |  Dev |  Test |
|----------------|------:|-----:|------:|
|starting with B |  5203 | 5347 |  5385 |
|starting with I |     2 |    0 |     1 |
|I next to O     |     0 |    0 |     0 |
|**Total**       |  5205 | 5347 |  5386 |

An example of *starting with I*: `test.tsv#L78550-L78598`
```
  
Compound	10510854	553	O  
7e	-	562	O  
,	-	564	O  
5	-	566	B  
-	-	567	I  
{	-	568	I  
2	-	569	I  
-	-	570	I  
// -------------
1H	-	637	I  
-	-	639	I  
  
indol	10510854	641	I  
-	-	646	I  
2	-	647	I  
-	-	648	I  
one	-	649	I  
,	-	652	O  
// -------------
```

### ShARe/CLEFE
|tag of tokens   | Train |  Dev |  Test |
|----------------|------:|-----:|------:|
|starting with B |  4628 | 1065 |  5195 |
|starting with I |     6 |    1 |    17 |
|I next to O     |   517 |  110 |   411 |
|**Total**       |  5151 | 1176 |  5623 |

An example of *I next to O*: `Test.tsv#L112-L118`  
You'd better check out these original files, too: 
- `Task1Gold_SN2012/Gold_SN2012/00176-102920-ECHO_REPORT.txt#L2`
- `Task1TestSetCorpus100/ALLREPORTS/00176-102920-ECHO_REPORT.txt#L21`
```
The	00176-102920-ECHO_REPORT	426	O
left	-	430	B
atrium	-	435	I
is	-	442	O
moderately	-	445	O
dilated	-	456	I
.	-	463	O
```

## Relation extraction

### DDI
| class        | Train |  Dev |  Test |
|--------------|------:|-----:|------:|
|DDI-advise    |   633 |  193 |   221 |
|DDI-effect    |  1212 |  396 |   360 |
|DDI-int       |   146 |   42 |    96 |
|DDI-mechanism |   946 |  373 |   302 |
|DDI-false     | 15842 | 6240 |  4782 |
|**Total**     |  2937<br>+15842| 1004<br>+6240 |  979<br>+4782 |

### ChemProt
| class | Train |  Dev |  Test |
|-------|------:|-----:|------:|
|CPR:3  |   768 |  550 |   665 |
|CPR:4  |  2251 | 1094 |  1661 |
|CPR:5  |   173 |  116 |   195 |
|CPR:6  |   235 |  199 |   293 |
|CPR:9  |   727 |  457 |   644 |
|false  | 15306 | 9404 | 13485 |
|**Total**|  4154<br>+15306| 2416<br>+9404 |  3458<br>+13485 |

### i2b2 2010
| class | Train |  Dev |  Test |
|-------|------:|-----:|------:|
|PIP    |   755 |    0 |  1448 |
|TeCP   |   158 |    8 |   338 |
|TeRP   |   993 |    0 |  2060 |
|TrAP   |   883 |    2 |  1732 |
|TrCP   |   184 |    0 |   342 |
|TrIP   |    51 |    0 |   152 |
|TrNAP  |    62 |    0 |   112 |
|TrWP   |    24 |    0 |   109 |
|false  | 19050 |   86 | 36707 |
|**Total**|  3110<br>+19050| 10<br>+86 |  6293<br>+36707 |

## Document multilabel classification
### HoC
| label                                 | Train |  Dev |  Test |
|---------------------------------------|------:|-----:|------:|
|0. activating invasion and metastasis  |   458 |   71 |   138 |
|1. avoiding immune destruction         |   148 |   33 |    45 |
|2. cellular energetics                 |   164 |   14 |    35 |
|3. enabling replicative immortality    |   213 |   30 |    52 |
|4. evading growth suppressors          |   264 |   34 |    70 |
|5. genomic instability and mutation    |   563 |   58 |   150 |
|6. inducing angiogenesis               |   238 |   39 |    80 |
|7. resisting cell death                |   596 |   92 |   145 |
|8. sustaining proliferative signaling  |   723 |   86 |   184 |
|9. tumor promoting inflammation        |   346 |   55 |   119 |

Note: This table shows the number of each label on the sentence level, rather than on the abstract level.
- **Train**: sentences: 10527/ articles: 1108
- **Dev**:   sentences:  1496/ articles:  157
- **Test**:  sentences:  2896/ articles:  315

## Inference task
### MedNLI
| class        | Train |  Dev |  Test |
|--------------|------:|-----:|------:|
|contradiction |  3744 |  465 |   474 |
|entailment    |  3744 |  465 |   474 |
|neutral       |  3744 |  465 |   474 |
|**Total**     | 11232 | 1395 |  1422 |

## Citing
currently being prepared...  
## Acknowledgments
We are grateful to the authors of BERT to make the data and codes publicly available. We thank the NVIDIA team because their implimentation of [BERT for PyTorch](https://github.com/NVIDIA/DeepLearningExamples/tree/master/PyTorch/LanguageModeling/BERT) enables us to train our BERT models locally. I would also like to take this opportunity to thank Yifan Peng and shared task organizers for publishing BLUE benchmark.  
This work was supported by Council for Science, Technology and Innovation (CSTI), cross-ministerial Strategic Innovation Promotion Program (SIP), "Innovative AI Hospital System" (Funding Agency: National Instisute of Biomedical Innovation, Health and Nutrition (NIBIOHN)).

## References
- <a id="ypeng"></a>Peng Y, Yan S, Lu Z. [Transfer Learning in Biomedical Natural Language Processing: An
Evaluation of BERT and ELMo on Ten Benchmarking Datasets](https://arxiv.org/abs/1906.05474). In *Proceedings of the Workshop on Biomedical Natural Language Processing (BioNLP)*. 2019.
