# Explicit Representation of the Translation Space:Automatic Paraphrasing for Machine Translation Evaluation

This is the github repository corresponding to the following article:

Rachel Bawden, Biao Zhang, Lisa Yankovskaya, Andre Tättar and Matt Post. 2020. Explicit Representation of the Translation Space: Automatic Paraphrasing for Machine Translation Evaluation. arXiv.

## Citation and licence

Please cite the following article:
```
@misc{bawden2020explicit,
    title={Explicit Representation of the Translation Space: Automatic Paraphrasing for Machine Translation Evaluation},
    author={Rachel Bawden and Biao Zhang and Lisa Yankovskaya and Andre Tättar and Matt Post},
    year={2020},
    eprint={2004.14989},
    archivePrefix={arXiv},
    primaryClass={cs.CL}
}
```
The content of this repository is available under TODO licence.

## Contents of this repository

Outputs and results (for WMT18/19 into-English test sets):

* `paraphrases/newstest201{8,9}/` - paraphrased references (human and automatic methods)
* `metrics-scores/newstest201{8,9}/` - BLEU and Meteor scores for baselines and multi-bleu metrics
* `metric-correlations/newstest201{8,9}/` - raw and relative metric correlations for all methods tested - TODO

Code:

* Diverse paraphrasing using sentence codes:
    - Treelstm sentence encoder, found here
    - Modified beam search in Marian, found here
    - Clustering
* `scripts/`
   - list scripts here
   - diversity metrics

Scripts to reproduce results:

Calculate diversity:

TODO

Calculate metric correlations:

TODO


