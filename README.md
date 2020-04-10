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
* [Total score](#total-score)
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

Note: Some of the figures in the table may differ from those in [the reference](#ypeng). We will explain them later.  

## Sentence similarity
- The sentence similarity task is to predict similarity scores based on sentence pairs [(ref)](#ypeng).
- **Metrics**: Pearson correlation coefficients
  + We used [scipy.stats.pearsonr()](https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.pearsonr.html).
### MedSTS
![MedSTS_hist](./img/medsts.png)  
[MedSTS](https://mayoclinic.pure.elsevier.com/en/publications/medsts-a-resource-for-clinical-semantic-textual-similarity) is a corpus of sentence pairs selected from Mayo Clinics clinical data warehouse and was used in the BioCreative/OHNLP Challenge 2018 Task 2 ([Wang et al., 2018](#ywang), also called [ClinicalSTS](https://sites.google.com/view/ohnlp2018/home)).  
Please visit the website or contact to the 1st author to obtain a copy of the dataset.
### BIOSSES
![BIOSSES_hist](./img/biosses.png)  
[BIOSSES](https://tabilab.cmpe.boun.edu.tr/BIOSSES/) is a corpus of sentence pairs selected from the Biomedical Summarization Track Training Dataset in the biomedical domain ([Soğancıoğlu et al., 2017](#gsogancioglu)).  
#### Known problems
The BIOSSES dataset is very small (only 16 sentence pairs in the development set). It causes unstable performance of fine-tuning, so we tried multiple random seeds and adopted the best model on the development set following the practice in [Peng et al. (2019)](#ypeng).
## Named entity recognition
- The aim of the named entity recognition task is to predict mention spans given in the text.
- **Metrics**: strict version of F1-score (exact phrase matching).
  + We used [conlleval.py](https://github.com/ncbi-nlp/bluebert/blob/master/bluebert/conlleval.py) from [the BlueBERT repository](https://github.com/ncbi-nlp/bluebert).
### Known problems
There are some irregular patterns:
- **starting with I**: caused by long phrases split in the middle [(example)](#startingwithi).
- **I next to O**: due to discontinuous mentions. It is often observed in [ShARe/CLEFE](#shareclefe) [(example)](#inexttoo).  

[conlleval.py](https://github.com/ncbi-nlp/bluebert/blob/master/bluebert/conlleval.py) appears to count them as different phrases.  
### BC5CDR-disease
|tag of tokens   | Train |  Dev |  Test |
|----------------|------:|-----:|------:|
|starting with B |  4182 | 4244 |  4424 |
|starting with I |     0 |    0 |     0 |
|I next to O     |     0 |    0 |     0 |
|**Total**       |  4182 | 4244 |  4424 |

[BC5CDR](https://biocreative.bioinformatics.udel.edu/tasks/biocreative-v/track-3-cdr/) is a collection of 1,500 PubMed titles and abstracts selected from the CTD-Pfizer corpus and was used in the BioCreative V chemical-disease relation task ([Li et al., 2016](#jli)).
### BC5CDR-chemical
|tag of tokens   | Train |  Dev |  Test |
|----------------|------:|-----:|------:|
|starting with B |  5203 | 5347 |  5385 |
|starting with I |     2 |    0 |     1 |
|I next to O     |     0 |    0 |     0 |
|**Total**       |  5205 | 5347 |  5386 |

An example of *starting with I*: `test.tsv#L78550-L78598`<a id="startingwithi"></a>
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

[ShARe/CLEFE](https://physionet.org/content/shareclefehealth2013/1.0/) eHealth Task 1 Corpus is a collection of 299 deidentified clinical free-text notes from the MIMIC II database ([Suominen et al.,2013](#hsuominen)).  
Please visit the website and sign up to obtain a copy of the dataset.  
An example of *I next to O*: `Test.tsv#L112-L118`<a id="inexttoo"></a>  
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
- The aim of the relation extraction task is to predict relations and their types between the two entities mentioned in the sentences. The relations with types were compared to annotated data.
- Following the implimentation of [BLUE benchmark](https://github.com/ncbi-nlp/BLUE_Benchmark), we treated the relation extraction task as a sentence classification by replacing two named entity mentions of interest in the sentence with predefined tags ([Lee et al., 2020](#jlee)).
  + **ORIGINAL**: *Citalopram* protected against the RTI-76-induced inhibition of *SERT* binding.
  + **REPLACED**: *@CHEMICAL$* protected against the RTI-76-induced inhibition of *@GENE$* binding.
  + **RELATION**: *citalopram* and *SERT* has **a chemical-gene relation**.
- **Evaluation**:
  1. predict classes containing "false".
  1. aggregate TP, FN, and FP in each class.
  1. calculate metrics excluding the "false" class.
- **Metrics**: micro-average F1-score.
  + We used [pmetrics.py](https://github.com/ncbi-nlp/BLUE_Benchmark/blob/master/blue/ext/pmetrics.py) from [the BLUE benchmark repository](https://github.com/ncbi-nlp/BLUE_Benchmark) to calculate micro-average.

### DDI
| class        | Train |  Dev |  Test |
|--------------|------:|-----:|------:|
|DDI-advise    |   633 |  193 |   221 |
|DDI-effect    |  1212 |  396 |   360 |
|DDI-int       |   146 |   42 |    96 |
|DDI-mechanism |   946 |  373 |   302 |
|DDI-false     | 15842 | 6240 |  4782 |
|**Total**     |  2937<br>+15842| 1004<br>+6240 |  979<br>+4782 |

[DDI](http://labda.inf.uc3m.es/ddicorpus) extraction 2013 corpus is a collection of 792 texts selected from the DrugBank database and other 233 Medline abstracts ([Herrero-Zazo et al., 2013](#hzazo)).

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

[ChemProt](https://biocreative.bioinformatics.udel.edu/news/corpora/) consists of 1,820 PubMed abstracts with chemical-protein interactions and was used in the BioCreative VI text mining chemical-protein interactions shared task ([Krallinger et al, 2017](#mkrallinger)).

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

[i2b2 2010](https://www.i2b2.org/NLP/DataSets/) shared task collection consists of 170 documents for training and 256 documents for testing, which is the subset of the original dataset ([Uzuner et al., 2011](#ouzuner)).

## Document multilabel classification
- The multilabel classification task predicts multiple labels from the texts.

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

[HoC](https://github.com/sb895/Hallmarks-of-Cancer) (the Hallmarks of Cancers corpus) consists of 1,580 PubMed publication abstracts manually annotated with ten currently known hallmarks of cancer([Baker et al., 2016](#sbaker)).
- **Evaluation**:
  1. predict multi-labels for each sentence in the document.
  1. combine the labels in one document and compare them with the gold-standard.
- **Metrics**: example-based F1-score on the abstract level ([Zhang and Zhou, 2014](#zz); [Du et al., 2019](#jdu)).
 + We used [pmetrics.py](https://github.com/ncbi-nlp/BLUE_Benchmark/blob/master/blue/ext/pmetrics.py) and [eval_hoc.py](https://github.com/ncbi-nlp/BLUE_Benchmark/blob/master/blue/eval_hoc.py) from [the BLUE benchmark repository](https://github.com/ncbi-nlp/BLUE_Benchmark) to calculate the metrics.

## Inference task
- The aim of the inference task is to predict whether the premise sentence entails or contradicts the hypothesis sentence.
- **Metrics**: overall accuracy
  + We used classification_report() in [pmetrics.py](https://github.com/ncbi-nlp/BLUE_Benchmark/blob/master/blue/ext/pmetrics.py).

### MedNLI
| class        | Train |  Dev |  Test |
|--------------|------:|-----:|------:|
|contradiction |  3744 |  465 |   474 |
|entailment    |  3744 |  465 |   474 |
|neutral       |  3744 |  465 |   474 |
|**Total**     | 11232 | 1395 |  1422 |

[MedNLI](https://physionet.org/content/mednli/1.0.0/) is a collection of sentence pairs selected from MIMIC-III.  
Please visit the website and sign up to obtain a copy of the dataset.  

## Total score
Following the practice in  [the BLUE benchmark](https://github.com/ncbi-nlp/BLUE_Benchmark), we use a macro-average of Pearson scores, F1-scores and an overall accuracy score to determine a pre-trained model's position.  
The results are [above](#results).

## Citing
currently being prepared...  
## Acknowledgments
We are grateful to the authors of BERT to make the data and codes publicly available. We thank the NVIDIA team because their implimentation of [BERT for PyTorch](https://github.com/NVIDIA/DeepLearningExamples/tree/master/PyTorch/LanguageModeling/BERT) enables us to train our BERT models locally. I would also like to take this opportunity to thank Yifan Peng and shared task organizers for publishing BLUE benchmark.  
This work was supported by Council for Science, Technology and Innovation (CSTI), cross-ministerial Strategic Innovation Promotion Program (SIP), "Innovative AI Hospital System" (Funding Agency: National Instisute of Biomedical Innovation, Health and Nutrition (NIBIOHN)).

## References
- <a id="ypeng"></a>Peng Y, Yan S, Lu Z. [Transfer Learning in Biomedical Natural Language Processing: An
Evaluation of BERT and ELMo on Ten Benchmarking Datasets](https://www.aclweb.org/anthology/W19-5006/). In *Proceedings of the Workshop on Biomedical Natural Language Processing (BioNLP)*. 2019.
- <a id="ywang"></a>Wang Y, Afzal N, Fu S, Wang L, Shen F, Rastegar-Mojarad M, Liu H. [MedSTS: a resource for clinical semantic textual similarity](https://link.springer.com/article/10.1007/s10579-018-9431-1). *Language Resources and Evaluation*. 2018 Jan 1:1-6.
- <a id="gsogancioglu"></a>Sog˘ancıog˘lu G, Öztürk H, Özgü A. [BIOSSES: a semantic sentence similarity estimation system for the biomedical domain](https://www.ncbi.nlm.nih.gov/pubmed/28881973). *Bioinformatics*. 2017 Jul 15; 33(14): i49–i58.
- <a id="jli"></a>Li J, Sun Y, Johnson RJ, Sciaky D, Wei CH, Leaman R, Davis AP, et al.. [BioCreative V CDR task corpus: a resource for chemical disease relation extraction](https://academic.oup.com/database/article/doi/10.1093/database/baw068/2630414). *Database: the journal of biological databases and curation*. 2016.
- <a id="hsuominen"></a>Suominen H, Salanterä S, Velupillai S, Chapman WW, Savova G, Elhadad N, Pradhan S, et al.. [Overview of the ShARe/CLEF eHealth evaluation lab 2013](https://link.springer.com/chapter/10.1007%2F978-3-642-40802-1_24). *Information Access Evaluation Multilinguality, Multimodality, and Visualization*. 2013. Springer. 212–231.
- <a id="jlee"></a>Jinhyuk Lee, Wonjin Yoon, Sungdong Kim, Donghyeon Kim, Sunkyu Kim, Chan Ho So, and Jaewoo Kang. 2019. [BioBERT: a pre-trained biomedical language representation model for biomedical text mining](https://www.ncbi.nlm.nih.gov/pubmed/31501885). *Bioinformatics*. 2020 Feb 15; 36(4): 1234-1240.
- <a id="hzazo"></a>Herrero-Zazo M, Segura-Bedmar I, Martínez P, Declerck T. [The DDI corpus: an annotated corpus with pharmacological substances and drug-drug interactions](https://www.ncbi.nlm.nih.gov/pubmed/23906817). *Journal of biomedical informatics*. 2013 46: 914–920.
- <a id="mkrallinger"></a>Krallinger M, Rabal O, Akhondi SA, Pérez MP, Santamaría JL, Rodríguez GP, Tsatsaronis G, et al.. [Overview of the BioCreative VI chemical-protein interaction track](https://biocreative.bioinformatics.udel.edu/tasks/biocreative-vi/track-5/). In Proceedings of *BioCreative*. 2017. 141–146.
- <a id="ouzuner"></a>Uzuner Ö,South BR,Shen S, DuVall SL. [2010 i2b2/va challenge on concepts, assertions, and relations in clinical text](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC3168320/). *Journal of the American Medical Informatics Association (JAMIA)*. 2011 18: 552–556.
- <a id="sbaker"></a>Baker S, Silins I, Guo Y, Ali I, Högberg J, Stenius U, Korhonen A. [Automatic semantic classification of scientific literature according to the hallmarks of cancer](https://www.ncbi.nlm.nih.gov/pubmed/26454282). *Bioinformatics (Oxford, England)*. 2016 32: 432–440.
- <a id="zz"></a>Zhang ML, Zhou ZH. [A review on multi-label learning algorithms](https://ieeexplore.ieee.org/document/6471714). *IEEE Transactions on Knowledge and Data Engineering*. 2014 26(8): 1819–1837.
- <a id="jdu"></a>Du J, Chen Q, Peng Y, Xiang Y, Tao C, Lu Z. [ML-Net: multilabel classification of biomedical texts with deep neural networks](https://academic.oup.com/jamia/article/26/11/1279/5522430). *Journal of the American Medical Informatics Association (JAMIA)*. 2019 Nov; 26(11); 1279–1285.
