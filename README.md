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
-----  
## Tasks
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
Note: Some of the figures in the table above may differ from those in [the reference](#ypeng). We will explain them below.  





| Corpus          | Train |  Dev | Test | Task                    | Metrics             | Domain     |
|-----------------|------:|-----:|-----:|-------------------------|---------------------|------------|
| MedSTS          |   675 |   75 |  318 | Sentence similarity     | Pearson             | Clinical   |
| BIOSSES         |    64 |   16 |   20 | Sentence similarity     | Pearson             | Biomedical |
| BC5CDR-disease  |  4182 | 4244 | 4424 | Named Entity Recognition| F1                  | Biomedical |
| BC5CDR-chemical |  5203 | 5347 | 5385 | NER                     | F1                  | Biomedical |
| ShARe/CLEFE     |  4628 | 1075 | 5195 | NER                     | F1                  | Clinical   |
| DDI             |  2937 | 1004 |  979 | Relation extraction     | micro F1            | Biomedical |
| ChemProt        |  4154 | 2416 | 3458 | Relation extraction     | micro F1            | Biomedical |
| i2b2-2010       |  3110 |   10 | 6293 | Relation extraction     | micro F1            | Clinical   |
| HoC             |  1108 |  157 |  315 | Document classification | F1                  | Biomedical |
| MedNLI          | 11232 | 1395 | 1422 | Inference               | accuracy            | Clinical   |
## Sentence similarity
### MedSTS

### BIOSSES

## Named entity recognition

### BC5CDR-disease

### BC5CDR-chemical

### ShARe/CLEFE

## Relation extraction

### DDI

### ChemProt

### i2b2 2010
| class | Train |  Dev |  Test | note   |
|-------|------:|-----:|------:|:-------|
|PIP    |   755 |    0 |  1448 |        |
|TeCP   |   158 |    8 |   338 |        |
|TeRP   |   993 |    0 |  2060 |        |
|TrAP   |   883 |    2 |  1732 |        |
|TrCP   |   184 |    0 |   342 |        |
|TrIP   |    51 |    0 |   152 |        |
|TrNAP  |    62 |    0 |   112 |        |
|TrWP   |    24 |    0 |   109 |        |
|false  | 19050 |   86 | 36707 |        |

## Document multilabel classification
### HoC

## Inference task
### MedNLI

## Citing
currently being prepared...  
## Acknowledgments
We are grateful to the authors of BERT to make the data and codes publicly available. We thank the NVIDIA team because their implimentation of [BERT for PyTorch](https://github.com/NVIDIA/DeepLearningExamples/tree/master/PyTorch/LanguageModeling/BERT) enables us to train our BERT models locally. I would also like to take this opportunity to thank Yifan Peng for providing BLUE benchmark.  
This work was supported by Council for Science, Technology and Innovation (CSTI), cross-ministerial Strategic Innovation Promotion Program (SIP), "Innovative AI Hospital System" (Funding Agency: National Instisute of Biomedical Innovation, Health and Nutrition (NIBIOHN)).

## References
- <a id="ypeng"></a>Peng Y, Yan S, Lu Z. [Transfer Learning in Biomedical Natural Language Processing: An
Evaluation of BERT and ELMo on Ten Benchmarking Datasets](https://arxiv.org/abs/1906.05474). In *Proceedings of the Workshop on Biomedical Natural Language Processing (BioNLP)*. 2019.
