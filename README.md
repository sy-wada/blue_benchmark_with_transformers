# BLUE Benchmark with Transformers

**\*\*\*\*\* New April th, 2020: released \*\*\*\*\***

-----  
Biomedical Language Understanding Evaluation (BLUE) benchmark is a collection of resources for evaluating and analyzing biomedical
natural language representation models ([Peng et al., 2019](#ypeng)).  
This repository provides our implementation of fine-tuning for BLUE benchmark with [🤗Transformers](https://github.com/huggingface/transformers).
## Preparations
1. Download the benchmark dataset from https://github.com/ncbi-nlp/BLUE_Benchmark
1. Set pre-trained models. For example, [BioBERT](https://github.com/dmis-lab/biobert), [clinicalBERT](https://github.com/EmilyAlsentzer/clinicalBERT), [SciBERT](https://github.com/allenai/scibert), [BlueBERT](https://github.com/ncbi-nlp/bluebert) and so on.
1. Try to use our code in [utils](./utils). Examples of the command can be found in [scripts](./scripts).
### Tips
If you download Tensorflow Models, converting to PyTorch ones comforts your fine-tuning.  
[Converting Tensorflow Checkpoints](https://huggingface.co/transformers/converting_tensorflow_models.html)
```bash
export BERT_BASE_DIR=/path/to/bert/uncased_L-12_H-768_A-12

transformers-cli convert --model_type bert \
  --tf_checkpoint $BERT_BASE_DIR/bert_model.ckpt \
  --config $BERT_BASE_DIR/bert_config.json \
  --pytorch_dump_output $BERT_BASE_DIR/pytorch_model.bin
```
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

## Acknowledgments
We are grateful to the authors of BERT to make the data and codes publicly available. We thank the NVIDIA team because their implimentation of BERT for PyTorch enables us to train our BERT models. I would also like to take this opportunity to thank Yifan Peng for providing BLUE benchmark.
This work was supported by Council for Science, Technology and Innovation (CSTI), cross-ministerial Strategic Innovation Promotion Program (SIP), "Innovative AI Hospital System" (Funding Agency: National Instisute of Biomedical Innovation, Health and Nutrition (NIBIOHN)).

## References
- <a id="ypeng"></a>Peng Y, Yan S, Lu Z. [Transfer Learning in Biomedical Natural Language Processing: An
Evaluation of BERT and ELMo on Ten Benchmarking Datasets](https://arxiv.org/abs/1906.05474). In *Proceedings of the Workshop on Biomedical Natural Language Processing (BioNLP)*. 2019.
