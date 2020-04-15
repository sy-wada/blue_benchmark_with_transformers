
# Fine-tuning pre-trained models
The optimal hyperparameters we have searched are described here.  
Please keep in mind that different environments may produce different results from ours.

- Our models
  - will be available soon.
- BERT-Base, Uncased ([Devlin et al., (2019)](#bert))
  - **Vocabulary**: BERT-Base (WordPiece tokenization ([Wu et al., 2016](#wordpiece)))
  - **Pre-training**:
    - trained from scratch.
    - pre-trained on BooksCorpus (0.8B words) and English Wikipedia (2.5B words).
    - **setting**:
      - **max_seq_length** = (phase1) 128 tokens/ (phase2) 512 tokens
      - **global_batch_size** = 256 sequences
      - **steps** = (phase1) 0.9M steps/ (phase2) 0.1M steps
      - **number of tokens processed in pre-training** :  
        128 tokens x 256 sequences x 0.9M steps  
        +512 tokens x 256 sequences x 0.1M steps = **43B tokens**
- [BlueBERT-Base, Uncased, PubMed](./BlueBERT-Base-Uncased_P)  ([Peng et al., 2019](#bluebert))
  - **Vocabulary**: BERT-Base
  - **Pre-training**:
    - initialized from BERT-Base-Uncased.
    - pre-trained on PubMed abstracts.
    - **setting**: *using the same vocabulary, sequence length, and other configurations provided by ([Devlin et al., (2019)](#bert)).*
      - **max_seq_length** = 128 tokens
      - **global_batch_size** = 256 sequences (32 x 8)?
      - **steps** = 5M steps
      - **number of tokens processed in pre-training** :  
        128 tokens x 256 sequences x 5M steps = **164B tokens**
- BlueBERT-Base, Uncased, PubMed+MIMIC-III ([Peng et al., 2019](#bluebert))
  - **Vocabulary**: BERT-Base
  - **Pre-training**:
    - initialized from BERT-Base-Uncased.
    - pre-trained on PubMed abstracts and MIMIC-III.
    - **setting**: *using the same vocabulary, sequence length, and other configurations provided by ([Devlin et al., (2019)](#bert)).*
      - **max_seq_length** = 128 tokens
      - **global_batch_size** = 256 sequences (32 x 8)?
      - **steps** = (PubMed) 5M steps + (MIMIC-III) 0.2M steps
      - **number of tokens processed in pre-training** :  
        128 tokens x 256 sequences x 5.2M steps = **170B tokens**
- BioBERT-Base, Cased v1.1 (+ PubMed 1M) ([Lee et al., 2019](#biobert))
  - **Vocabulary**: custom 30k vocabulary
  - **Pre-training**:
    - trained from scratch.
    - pre-trained on PubMed abstracts.
    - **setting**: *other hyper-parameters such as batch size and learning rate scheduling for pre-training BioBERT are the same as those for pre-training BERT unless stated otherwise.*
      - **max_seq_length** = 512 tokens
      - **global_batch_size** = 192 sequences
      - **steps** = 1M steps
      - **number of tokens processed in pre-training** :  
        512 tokens x 192 sequences x 1M steps = **98B tokens**

## References
- <a id="bert"></a>Devlin J, Chang M-W, Lee K, Toutanova K. [BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding](https://www.aclweb.org/anthology/N19-1423/). In *Proceedings of the 2019 Conference of the North American Chapter of the Association for Computational Linguistics: Human Language Technologies*, Volume 1 (Long and Short Papers); 2019: 4171-4186.
- <a id="wordpiece"></a>Wu Y, Schuster M, Chen Z, Le QV, Norouzi M, Macherey W, et al. [Google's Neural Machine Translation System: Bridging the Gap between Human and Machine Translation](https://research.google/pubs/pub45610/). *arXiv preprint*. arXiv:1609.08144. 2016.
- <a id="bluebert"></a>Peng Y, Yan S, Lu Z. [Transfer Learning in Biomedical Natural Language Processing: An Evaluation of BERT and ELMo on Ten Benchmarking Datasets](https://arxiv.org/abs/1906.05474). In *Proceedings of the Workshop on Biomedical Natural Language Processing (BioNLP)*. 2019;58-65.
- <a id="biobert"></a>Lee J, Yoon W, Kim S, Kim D, Kim S, So CH, et al. [BioBERT: a pre-trained biomedical language representation model for biomedical text mining](https://www.ncbi.nlm.nih.gov/pubmed/31501885). 2019;36(4):1234-40.
