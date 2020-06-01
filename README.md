# Explicit Representation of the Translation Space: Automatic Paraphrasing for Machine Translation Evaluation

## Requirements and related code

* python3 and the following packages (installable using `pip install`): sacrebleu==1.4.10, numpy, scipy, apted, nltk, sklearn
     - `pip3 install -r requirements.txt`
* Sentencepiece (https://github.com/google/sentencepiece)
* Moses scripts (https://github.com/moses-smt/mosesdecoder)
* TreeKernel (from https://github.com/JDonner/TreeKernel, reimplementation of Alessandro Moschitti's 2006 paper, "Making Tree Kernels Practical for Natural Language Learning")
     - With modifications (found in `tools/`)
     ```
     cd tools/TreeKernel/
     cd tree-parser; make; cd ..
     cd tree-kernel; make
     ```
* Diverse paraphrasing using sentence codes:
    - Laser embeddings: https://github.com/facebookresearch/LASER
    - Parsing of inputs, using the Berkeley Parser 1.1 (Petrov et al., 2006)
    - Treelstm sentence encoder in `treelstm-autoencoder/`
    - Modified beam search in Marian, found in this [modified version](ANON) (link anonymised for submission)
* Constrained decoding with n-grams:
    - Sockeye MT toolkit: https://github.com/awslabs/sockeye

## Contents of this repository

Outputs and results (for WMT18/19 into-English test sets):

* `paraphrases/newstest201{8,9}/` - paraphrased references (human and automatic methods)
* `paraphrases-parses/newstest201{8,9}/` - paraphrased references (human and automatic methods)
* `diversity-results/` - raw diversity scores for WMT19
* `metric-scores/newstest201{8,9}/` - BLEU (and Meteor) scores for baselines and multi-Bleu (or Meteor) metrics using automatic paraphrases
* `metrics-task/` - gold (human) quality assessment files for WMT18 and WMT19
* `latex-correlation-results/newstest201{8,9}/` - raw and relative metric correlations for all methods tested - TODO

Code and scripts:

* scripts in `scripts/`
* code either in this directory or linked in this README

## Instructions for reproducing results: 

### Data and pre-processing

Our paraphrase data comes from [Parabank2](http://decomp.io/projects/parabank2/) (Hu et al., 2019)
We take one paraphrase for each sentences, resulting in approximately 20M sentences associated with a paraphrase.

Pre-processing applied:

* Deduplication, remove sentences longer than 100 raw tokens (no tokenisation), segment into subwords using [SentencePiece](https://github.com/google/sentencepiece) with a unigram model of size 16k.
* Randomly shuffle and split into train/dev/test: 3k for dev and test each, the rest for training

Pre-processing applied to newstest data:

* Normalisation of punctuation using the [Moses](https://github.com/moses-smt/mosesdecoder) script (to be inline with the normalisation of apostrophes in Parabank) and subword segmentation as above


---

### Producing the sentence codes

Producing cluster codes for each sentence requires:
1. Encoding each sentence using either the lexical or syntactic method
2. Clustering the resulting representation into 256 codes and assigning each a number

The resulting cluster codes are found in `train-dev-test-data.tgz` in the folder `sentence-codes`

#### Lexical codes (using Laser)

1. Get LASER embeddings for data file like (train.en). Before using, set environment variable DATA, where your data is located. Then run the script. More info in the script. Also have train/valid/test data separated before extracting laser embeddings. NB! These can take a lot of hard drive space. 
```
bash scripts/extract_laser_embeds.sh train.en
```

#### Syntactic codes (Using a Treelstm autoencoder)

1. Parse the training data using the Berkeley parser
2. Prune the trees to depth 4 and remove leaves
```
cat PARSE_TREE_FILE | python scripts/prune_trees.py --depth DEPTH [--remove_leaves]
```
3. Encode each parse tree using the treelstm-autoencoder. More detailed instructions of this step are given in `treelstm-autoencoder/README`

#### Clustering the representations

Cluster the vectorial representation using the following script:
```
python scripts/cluster.py --n_clusters 256 --algorithm kmeans \
            --model_file MODEL_FILE --in_file IN_FILE --out_file OUT_FILE [--seed SEED]
```

The input should be in the following format: a ".laser" file (if created by laser as a numpy array), a ".tl" file (as a Torch format data) or as a numpy array saved in a file. 
The model can be used to apply the clusters to new data with the following script.

```
python scripts/cluster.py --model_file MODEL_FILE --in_file NEW_FILE --out_file OUT_FILE --seed SEED --predict
```

---

### Training the paraphrase models

The models are too large to be stored in this repository, but training and paraphrase scripts are provided in `paraphrase-models/`. Training and validation data is created by pasting the relevant sentence codes (separated by a space) to the training and validation sets on the target side of the data only. 

E.g. Training the laser model:
```
cd paraphrase-models/laser
bash train.sh "0 1 2 3"
```
The models are trained with early stopping of 10 validations and the model to be used is the one with the best BLEU score on the validation set (see `validate.sh`).

Producing the paraphrases:
```
bash produce_paraphrase_outputs.sh MODEL NUM_PARAS TESTSET GPUS

    MODEL: path to model (E.g. model/model.npz.best_translation.npz)
    NUM_PARAS: 0-20
    TESTSET: newstest2018 or newstest2019
    GPUS: list of devices of the format "0 1 2 3"
```
The paraphrases are output to `paraphrases/MODEL_TYPE/`

Sockeye models:

TODO? (some light instructions here?)

---

### Calculate diversity of a set of paraphrases

There are 2 diversity metrics (lexical (BOW) adn syntactic (syntax), calculated using the following script:

```
python scripts/calculate_diversity.py PARA_FOLDER TYPE -n NUM

    PARA_FOLDER: folder containing parses of paraphrased reference (e.g. paraphrases-parses/newstest2019/laser)
    TYPE: syntax or bow
    [-n NUM]: number of paraphrased references to include in the set (2-20)
```

E.g. Calculating syntactic diversity of the top 2 laser 2019 paraphrases:

```
python3 scripts/calculate_diversity.py paraphrases-parses/newstest2019/laser syntax -n 2
```

N.B. The syntax metric requires TreeKernel to be compiled

---

### Calculate BLEU (or METEOR) scores:

#### For each diverse paraphrase-augmented metric:
```
bash scripts/produce-system-scores-{seg,sys}level.sh TESTSET EVAL_TOOL NAME PARAHPRASE_FOLDER NUM_PARAPHRASES

    TESTSET: newstest2018 or newstest2019
    EVAL_TOOL: scripts/sacreBLEU-{seg,sys}level.sh or scripts/Meteor-{seg,sys}level.sh
    NAME: the metric name to be written into the output file
    PARAPHRASE_FOLDER: the specific folder containing paraphrased references (a folder in paraphrases/)
    NUM_PARAPHRASES: the number of additional paraphrased references (1-20) to use
```

E.g. Calculating BLEU scores for 5 sampled paraphrased references (out of a maximum of 20) in addition to the original one, for the newstest2019 test set at the system level:
```
bash scripts/produce-system-scores-syslevel.sh \
    newstest2019 \
    scripts/sacreBLEU-syslevel.sh  \
    sampled-5-syslevel paraphrases/newstest2019/sampled 5 \
     > metric-scores/newstest2019/sampled/parbleu-sampled.num\=5-syslevel.tsv
```

#### For each MT-specific paraphrase-augmented metric:
(i.e. making one paraphrased referencethat is specific to each MT output)

```
bash scripts/produce-metric-scores-output-specific-{seg,sys}level.sh TESTSET METHOD EVAL_TOOL
       
    TESTSET: newstest2018 or newstest2019
    METHOD: laser or treelstm
    EVAL_TOOL: scripts/sacreBLEU-{seg,sys}level.sh or scripts/Meteor-{seg,sys}level.sh
```

E.g. Calculating BLEU scores using Laser-guided paraphrases at the system level for newstest2019:

```
bash scripts/produce-metric-scores-output-specific-syslevel.sh newstest2019 laser scripts/sacreBLEU-syslevel.sh \
    > metric-scores/newstest2019/sampled/parbleu-laser-constrained.syslevel.tsv
 ```
 
#### For the 500 de-en subsample:

Diverse paraphrases: 
```
bash scripts/produce-metric-scores-500-subsample-{seg,sys}level.sh EVAL_TOOL NAME PARAHPRASE_FOLDER NUM_PARAPHRASES

    EVAL_TOOL: scripts/sacreBLEU-{seg,sys}level-truncate.sh or scripts/Meteor-{seg,sys}level-truncate.sh
    NAME: metric name to be written into the output file
    PARAPHRASE_FOLDER: folder in paraphrases/newstest2019 containing the paraphrased references
    NUM_PARAPHRASES: number of additional, paraphrased references to use (up to 5)
```

Output-guided paraphrases:
```
bash scripts/produce-metric-scores-output-specific-500-subsample-{seg,sys}level.sh EVAL_TOOL NAME MODEL_TYPE

    EVAL_TOOL: scripts/sacreBLEU-seglevel-truncate.sh 
    NAME: metric name to be written into the output file 
    MODEL_TYPE: 'laser' or 'treelstm'
```

---

### Calculate correlation of metric scores with human judgments:

```
python3 scripts/metric_correlation-syslevel.py HUMAN_ASSESSMENTS BASELINE_SCORES SYSTEM_SCORES

           HUMAN_ASSESSMENTS: metrics-task/{RR-seglevel,DA-syslevel}-newstest201{8,9}.csv
           BASELINE_SCORES: metric-scores/newstest2019/{Meteor,sacreBLEU}-{seg,sys}level.tsv
           SYSTEM_SCORES: metric-scores output (produced in previous step)
```

E.g. Calculating system-level correlations for the metric score file produced above, calculating significance against the sacreBLEU baseline:
```
python3 scripts/metric_correlation-syslevel.py \
            metrics-task/DA-syslevel-newstest2019.csv \
            metric-scores/newstest2019/sacreBLEU-syslevel.tsv \
            metric-scores/newstest2019/sampled/parbleu-sampled.num\=5-syslevel.tsv
```


Re-create results tables:
(Outputs found in `latex-correlation-results/`)
``` 
bash scripts/metric_correlation-create-raw-latex-table.py {bleu,meteor} {seg,sys}level newstest201{89} \
     > latex-correlation-results/par{bleu,meteor}-newstest2019-summary-{seg,sys}level.tex
bash scripts/metric_correlation-create-summary-latex-table.py {bleu,meteor} newstest201{89}
     > latex-correlation-results/par{bleu,meteor}-newstest2019-raw-{seg,sys}level.tex
python3 scripts/metric_correlation-create-raw-500-latex-table.py \
     > latex-correlation-results/parbleu-newstest2019-raw-500-subsample.tex
```




