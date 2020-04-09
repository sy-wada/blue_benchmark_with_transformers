
# Fine-tuning pre-trained models
The optimal hyperparameters we have searched are described here.  
Please keep in mind that different environments may produce different results from ours.

- [BlueBERT-Base, Uncased, PubMed](./BlueBERT-Base-Uncased_P) [(ref)](#bluebert)
  - **Vocabulary**: BERT-Base
  - **Pre-training**: initialized from BERT-Base-Uncased and pretrained on PubMed abstracts. 

## References
- <a id="bluebert"><a>Peng Y, Yan S, Lu Z. [Transfer Learning in Biomedical Natural Language Processing: An Evaluation of BERT and ELMo on Ten Benchmarking Datasets](https://arxiv.org/abs/1906.05474). In *Proceedings of the Workshop on Biomedical Natural Language Processing (BioNLP)*. 2019.
