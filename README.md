# GLADIS
GLADIS: A General and Large Acronym Disambiguation Benchmark (Long paper at EACL 23)


The tree structure of this folder is shown as follows

```
│   README.md
│   requirements.txt
│
├───evaluation
│   ├───dict
│   │       bio_umls_dict.json
│   │       sciad_dict.json
│   │       uad_dict.json
│   │
│   └───test_set
│           bio_umls_test.json
│           sciad_test.json
│           uad_test.json
│
├───input
│   │   acrobert.pt
│   │   acronym_kb.json
│   │   pre_train_sample.txt
│   │
│   └───dataset
│       ├───biomedical
│       │       dev.json
│       │       test.json
│       │       train.json
│       │
│       ├───general
│       │       dev.json
│       │       test.json
│       │       train.json
│       │
│       └───scientific
│               dev.json
│               test.json
│               train.json
│
├───output
└───source
        acrobert.py
        evaluation.py
        utils.py
        __init__.py
```
## Benchmark
The new benchmark constructed in this paper is located in `input/dataset`.
This benchmark covers three domains: general, scientific and biomedical.
The acronym dictionary is stored in this file: `input/dataset/acronym_kb.json`, which includes 1.5M acronyms
and 6.4M long forms.
However, due to the size limit of the upload files, you have to download the dictionary (with the AcroBERT model together) from this dropbox link:
[dictionary and model](https://zenodo.org/record/7568921#.Y9FA5XaZNPY). 
After downloading, decompress it and put the two files to this path `input/`

## Pre-training
The pre-training corpus can be downloaded from this [link](https://zenodo.org/record/7562925#.Y87_3naZNPY), which contains 160 million sentences with acronyms.

Here we use python3.6 and the Transformers library to implement the model. 
```
pip install -r requirements.txt
```
### Data preparation
In total, there are 160 million samples in the pre-training corpus, covering various domains.
Here, we put 10K samples in this path for testing `input/pre_train_sample.txt`

### Training
First you can use `-help` to show the arguments
```
python train.py -help
```
Once completing the data preparation and environment setup, we can train the model via `acrobert.py`.

```
python acrobert.py -pre_train_path ../input/pre_train_sample.txt
```
The entire pre-training needs two weeks on a single NVIDIA Tesla V100S PCIe 32 GB Specs.
## Evaluation
### Quick Reproduction
We provide a one-line command to reproduce the scores in Table A1,
which is the easiest one to reproduce, and you can see the scores after only several minutes. 
The needed test sets are store in this path `/evaluation/test_set`, and you can find three evaluation sets.
The corresponding dictionaries are in this path `/evaluation/dict`.
We also provided the AcroBERT model file in this path `/input/acrobert.pt`.
Then the scores can be obtained by using the following command:
```
python acrobert.py -mode evaluation
```
Finally, you can see the results
```
F1: [88.8, 58.0, 67.5], ACC: [93.7, 72.0, 65.3]
```
