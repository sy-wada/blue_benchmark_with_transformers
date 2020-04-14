
# Fine-tuning pre-trained models
The optimal hyperparameters we have searched are described here.  
Please keep in mind that different environments may produce different results from ours.

- [BlueBERT-Base, Uncased, PubMed](./BlueBERT-Base-Uncased_P)  ([Peng et al., 2019](#bluebert))
  - **Vocabulary**: BERT-Base
  - **Pre-training**:
    - initialized from BERT-Base-Uncased
    - pre-trained on PubMed abstracts.
    - **setting**: *using the same vocabulary, sequence length, and other configurations provided by ([Devlin et al., (2019)](#bert).*
      - **batch size** : 256 sequences?
      - **step**: 5M steps
      - **processed tokens** : 256 sequences * 128 tokens * 5M steps = 164B tokens

## References
- <a id="bert"><a>Devlin J, Chang M-W, Lee K, Toutanova K. [BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding](https://www.aclweb.org/anthology/N19-1423/). Proceedings of the 2019 Conference of the North American Chapter of the Association for Computational Linguistics: Human Language Technologies, Volume 1 (Long and Short Papers); 2019: 4171-4186.
- <a id="bluebert"><a>Peng Y, Yan S, Lu Z. [Transfer Learning in Biomedical Natural Language Processing: An Evaluation of BERT and ELMo on Ten Benchmarking Datasets](https://arxiv.org/abs/1906.05474). In *Proceedings of the Workshop on Biomedical Natural Language Processing (BioNLP)*. 2019.
