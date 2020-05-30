# Explicit Representation of the Translation Space: Automatic Paraphrasing for Machine Translation Evaluation

## Requirements

python3 and the following packages (installable using `pip install`): sacrebleu, numpy, scipy

`pip3 install -r requirements.txt`

## Contents of this repository

Outputs and results (for WMT18/19 into-English test sets):

* `paraphrases/newstest201{8,9}/` - paraphrased references (human and automatic methods)
* `metrics-scores/newstest201{8,9}/` - BLEU (and Meteor) scores for baselines and multi-Bleu (or Meteor) metrics using automatic paraphrases
* `metric-correlations/newstest201{8,9}/` - raw and relative metric correlations for all methods tested - TODO

Code:

* Diverse paraphrasing using sentence codes:
    - Treelstm sentence encoder, found 
    - Modified beam search in Marian, found in this [modified version](https://github.com/rbawden/marian-dev-diverse-beam)
    - Clustering
* `scripts/`
   - list scripts here
   - diversity metrics

Scripts to reproduce results:

Calculate diversity:

TODO

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
bash scripts/produce-metric-scores-output-specific-syslevel.sh newstest2019 laser scripts/sacreBLEU-syslevel.sh
 ```

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

Re-create raw results table:

```
bash scripts/metric_correlation-create-latex-table.sh newstest201{89} {seg,sys} {bleu,meteor}
```


