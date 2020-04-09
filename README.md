# BLUE Benchmark with Transformers

**\*\*\*\*\* New April th, 2020: released \*\*\*\*\***

-----  
Biomedical Language Understanding Evaluation (BLUE) benchmark is a collection of resources for evaluating and analyzing biomedical
natural language representation models ([Peng et al., 2019](#ypeng)).  
This repository provides our implementation of fine-tuning for BLUE benchmark with [🤗Transformers](https://github.com/huggingface/transformers).
## Preparations
1. Download the benchmark dataset from https://github.com/ncbi-nlp/BLUE_Benchmark
1. Set pre-trained models. For example, [BioBERT](https://github.com/dmis-lab/biobert), [clinicalBERT](https://github.com/EmilyAlsentzer/clinicalBERT), [SciBERT](https://github.com/allenai/scibert), [BlueBERT](https://github.com/ncbi-nlp/bluebert) and so on.
1. Try to use our code. Examples of the command can be found in [script](./script).

## Sentence similarity
### MedSTS

### BIOSSES

## Named Entity Recognition

### BC5CDR-disease

### BC5CDR-chemical

### ShARe/CLEFE

## Relation Extraction

### DDI

### ChemProt

### i2b2 2010

## Document multilabel classification
### HoC

## Inference task
### MedNLI

## References
- <a id="ypeng"></a>Peng Y, Yan S, Lu Z. [Transfer Learning in Biomedical Natural Language Processing: An
Evaluation of BERT and ELMo on Ten Benchmarking Datasets](https://arxiv.org/abs/1906.05474). In *Proceedings of the Workshop on Biomedical Natural Language Processing (BioNLP)*. 2019.
