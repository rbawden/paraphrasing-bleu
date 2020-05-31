# Tree-LSTM autoencoder

Reimplementation of treelstm autoencoder for sentence codes generation in https://www.aclweb.org/anthology/P19-1177/.

Source code is inspired by [treelstm.pytorch](https://github.com/dasguptar/treelstm.pytorch).

## Environment setup
* Python version 3.5+
* Pytorch 1.2.0

## Preprocessing
1. prepare vocabulary
    - For parsing trees: `python vocab.py --tree input.tree vocab.txt` (output is vocab.txt)
    - For source sentence: `python vocab.py input.src src_vocab.txt` (output is src_vocab.txt)
2. Folder format
    ```
    data/
    ├── dev             # development set
    │   ├── src         # source input, default name `src`
    │   └── tree        # tree structure, default name `tree`
    ├── src_vocab.txt   # source vocabulary
    ├── train           # training set
    │   ├── src
    │   └── tree
    └── vocab.txt       # tree parsing vocabulary
    ```
 
## Usage

We support multiple-gpu training and gradient accumulation. 
`--device` indicates the gpu index list, and `--num_grad_agg` denotes the accumulation step. 
For example, `--device 0 3 --num_grad_agg 2` means using gpu0 and gpu3 and accumulating gradients of 2 steps for optimizer

1. Train a basic treelstm with syntactic parsing tree alone
    ```
    python3 run.py --lr 0.025 --wd 0.0001 --optim adagrad --cuda --data data/ --expname model --mode train --batchsize 25 
    ```
2. Train a treelstm with improved semantic hashing, with syntactic parsing tree alone
    ```
    python3 run.py --lr 0.025 --wd 0.0001 --optim adagrad --cuda --data data/ --expname model --mode train --batchsize 25 --use_bottleneck
    ```
3. Train a treelstm with improved semantic hashing plus source input
    ```
    python3 run.py --lr 0.025 --wd 0.0001 --optim adagrad --cuda --data data/ --expname model --mode train --batchsize 25 --use_bottleneck --use_src
    ```
