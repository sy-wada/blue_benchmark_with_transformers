
# Fine-tuning pre-trained models
The optimal hyperparameters we have searched are described here.  
Please keep in mind that different environments may produce different results from ours.

- Our models
  - [ouBioBERT-Base, Uncased (demo)](./BioMed-Base-Uncased_P_demo)
    - **Vocabulary**: custom 32k vocabulary
    - **Pre-training**:
      - trained from scratch.
      - pre-trained on PubMed abstracts.
      - **setting**: almost same as [BERT For PyTorch](https://github.com/NVIDIA/DeepLearningExamples/blob/master/PyTorch/LanguageModeling/BERT/scripts/run_pretraining.sh) by NVIDIA, but modified to allow us to run on our local machine.
        - **max_seq_length** = (phase1) 128 tokens
        - **global_batch_size** = (phase1) 65,536 sequences
        - **steps** = (phase1) 7,038 steps
        - **number of tokens processed in pre-training** :  
          128 tokens x 65,536 sequences x 7,038 steps = **59B tokens**
  - ouBioBERT-Base, Uncased (full)
    - **Vocabulary**: custom 32k vocabulary
    - **Pre-training**:
      - initialized from [ouBioBERT-Base, Uncased (demo)](./BioMed-Base-Uncased_P_demo) and run with additional steps on max_seq_length=512.
      - pre-trained on PubMed abstracts.
      - **setting**: almost same as [BERT For PyTorch](https://github.com/NVIDIA/DeepLearningExamples/blob/master/PyTorch/LanguageModeling/BERT/scripts/run_pretraining.sh) by NVIDIA, but modified to allow us to run on our local machine.
        - **max_seq_length** = (phase1) 128 tokens/ (phase2) 512 tokens
        - **global_batch_size** = (phase1) 65,536 sequences/ (phase2) 32,768 sequences
        - **steps** = (phase1) 7,038 steps/ (phase2) 1,563 steps
        - **Note that the phase1 has already done on ouBioBERT-Base, Uncased (demo).**
        - **number of tokens processed in pre-training** :  
          128 tokens x 65,536 sequences x 7,038 steps  
          +512 tokens x 32,768 sequences x 1,563 steps = **85B tokens**
  - BERT (sP+B+W), Uncased
    - **Vocabulary**: custom 32k vocabulary
    - **Pre-training**:
      - trained from scratch.
      - pre-trained on a small corpus of PubMed abstracts (200MB) with BooksCorpus (5GB) + Wikipedia (13GB).
      - **setting**:  
        - **max_seq_length** = 128 tokens
        - **global_batch_size** = 2,048 sequences
        - **steps** = 125K steps
        - **number of tokens processed in pre-training** :  
          128 tokens x 2,048 sequences x 125K steps = **33B tokens**
- BERT-Base (L-12_H-768_A-12), Uncased ([Devlin et al., (2019)](#bert))
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
- BlueBERT ([Peng et al., 2019](#bluebert))
  - [BlueBERT-Base, Uncased, PubMed](./BlueBERT-Base-Uncased_P) 
    - **Vocabulary**: BERT-Base
    - **Pre-training**:
      - initialized from BERT-Base-Uncased.
      - pre-trained on PubMed abstracts.
      - **setting**: *using the same vocabulary, sequence length, and other configurations provided by ([Devlin et al., (2019)](#bert)).*
        - **max_seq_length** = 128 tokens
        - **global_batch_size** = 256 sequences, or 32?
        - **steps** = 5M steps
        - **number of tokens processed in pre-training** :  
          128 tokens x 256(32?) sequences x 5M steps = **164B(21B?) tokens**
  - [BlueBERT-Base, Uncased, PubMed+MIMIC-III](./BlueBERT-Base-Uncased_PM)
    - **Vocabulary**: BERT-Base
    - **Pre-training**:
      - initialized from BERT-Base-Uncased.
      - pre-trained on PubMed abstracts and MIMIC-III.
      - **setting**: *using the same vocabulary, sequence length, and other configurations provided by ([Devlin et al., (2019)](#bert)).*
        - **max_seq_length** = 128 tokens
        - **global_batch_size** = 256 sequences, or 32?
        - **steps** = (PubMed) 5M steps + (MIMIC-III) 0.2M steps
        - **number of tokens processed in pre-training** :  
          128 tokens x 256(32?) sequences x 5.2M steps = **170B(21B?) tokens**
- SciBERT-Base, Uncased ([Beltagy et al., 2019](#scibert))
  - **Vocabulary**: custom 30k vocabulary named as scivocab
  - **Pre-training**:
    - trained from scratch.
    - pre-trained on a random sample of 1.14M full text papers from Semantic Scholar, which consists of 18% papers from the computer science domain and 82% from the broad biomedical domain.
    - **setting**: We guess the hyperparameters as follows.  
      *We use the original BERT code to train SCIBERT on our corpus with the same configuration and size as BERT-Base. We set a maximum sentence length of 128 tokens, and train the model until the training loss stops decreasing. We then continue training the model allowing sentence lengths up to 512 tokens. ([Beltagy et al., 2019](#scibert))*
      - **max_seq_length** = (phase1) 128 tokens/ (phase2) 512 tokens
      - **global_batch_size** = 256 sequences
      - **steps**: not clearly stated. *"until the training loss stops decreasing"*
      - **number of tokens processed in pre-training** :   
        **unknown**
- clinicalBERT ([Alsentzer et al., 2019](#clinicalbert))
  - clinicalBERT-Base, Cased (Bio+Clinical BERT)
    - **Vocabulary**: BERT-Base
    - **Pre-training**:
      - initialized from [BioBERT-Base v1.0 (+ PubMed 200K + PMC 270K)](https://github.com/naver/biobert-pretrained).
      - pre-trained on MIMIC notes: text from all note types.
      - **setting**:
        - **max_seq_length** = 128 tokens
        - **global_batch_size** = 32 sequences
        - **steps** = 150K steps
        - **number of tokens processed in pre-training** :  
          128 tokens x 32 sequences x 150K steps = **614M tokens**
  - clinicalBERT-Base, Cased (Bio+Discharge Summary BERT)
    - **Vocabulary**: BERT-Base
    - **Pre-training**:
      - initialized from [BioBERT-Base v1.0 (+ PubMed 200K + PMC 270K)](https://github.com/naver/biobert-pretrained).
      - pre-trained on MIMIC notes: only discharge summaries. 
      - **setting**:
        - **max_seq_length** = 128 tokens
        - **global_batch_size** = 32 sequences
        - **steps** = 150K steps
        - **number of tokens processed in pre-training** :  
          128 tokens x 32 sequences x 150K steps = **614M tokens**
- [BioBERT-Base, Cased v1.1 (+ PubMed 1M)](./BioBERT-Base-Cased_v1.1) ([Lee et al., 2019](#biobert))
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
- <a id="scibert"></a>Beltagy I, Lo K, Cohan A, editors. [SciBERT: A pretrained language model for scientific text](https://www.aclweb.org/anthology/D19-1371/). In *Proceedings of the 2019 Conference on Empirical Methods in Natural Language Processing and the 9th International Joint Conference on Natural Language Processing (EMNLP-IJCNLP)*. 2019;3606-3611.
- <a id="clinicalbert"></a>Alsentzer E, Murphy J, Boag W, Weng W-H, Jindi D, Naumann T, et al.. [Publicly Available Clinical BERT Embeddings](https://www.aclweb.org/anthology/W19-1909/). In *Proceedings of the 2nd Clinical Natural Language Processing Workshop*. 2019;72-78.
- <a id="biobert"></a>Lee J, Yoon W, Kim S, Kim D, Kim S, So CH, et al. [BioBERT: a pre-trained biomedical language representation model for biomedical text mining](https://www.ncbi.nlm.nih.gov/pubmed/31501885). 2019;36(4):1234-40.
