# coding: utf-8

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import random
import logging

import torch
import torch.optim as optim

import zglobal
import parallel
from model import TreeLSTMAutoEncoder
from vocab import Vocab
from trainer import Trainer
from dataset import TreeDataset
from config import parse_args


# MAIN BLOCK
def main():
    global args
    args = parse_args()

    # local directory
    if not os.path.exists(args.save):
        os.makedirs(args.save)

    # global logger
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.DEBUG)
    formatter = logging.Formatter("[%(asctime)s] %(levelname)s:%(name)s:%(message)s")

    # file logger
    fh = logging.FileHandler(os.path.join(args.save, args.expname)+'.log', mode='w')
    fh.setLevel(logging.INFO)
    fh.setFormatter(formatter)
    logger.addHandler(fh)

    # console logger
    ch = logging.StreamHandler()
    ch.setLevel(logging.DEBUG)
    ch.setFormatter(formatter)
    logger.addHandler(ch)

    # argument validation
    # single gpu running, if too slow switch multiple-gpu running
    args.cuda = args.cuda and torch.cuda.is_available()
    device = parallel.get_device(args.device) if args.cuda else torch.device("cpu")
    args.shard_size = len(args.device) if args.cuda else 1

    # control randomness
    logger.debug(args)
    torch.manual_seed(args.seed)
    random.seed(args.seed)
    if args.cuda:
        torch.cuda.manual_seed(args.seed)
        torch.backends.cudnn.benchmark = True

    train_dir = os.path.join(args.data, 'train/')
    dev_dir = os.path.join(args.data, 'dev/')
    test_dir = os.path.join(args.data, 'test/')

    # prepare vocabularies
    vocab_file = os.path.join(args.data, 'vocab.txt')
    assert os.path.isfile(vocab_file)
    src_vocab_file = os.path.join(args.data, 'src_vocab.txt')
    if args.use_src:
        assert os.path.isfile(src_vocab_file)

    # get vocab object from vocab file previously written
    vocab = Vocab(vocab_file)
    logger.debug('==> vocabulary size : %d ' % vocab.size())
    if args.use_src:
        src_vocab = Vocab(src_vocab_file)
        logger.debug('==> source vocabulary size : %d ' % src_vocab.size())

    # initialize model, criterion/loss_function, optimizer
    model = TreeLSTMAutoEncoder(
        vocab.size(),       # vocabulary size, word embeddings
        args.input_dim,     # word embedding siz
        args.mem_dim,       # hidden size in tree

        bits_number=args.bit_number,                # the number of hashing bits
        filter_size=args.filter_size,               # the internal state size for unbottleneck
        max_num_children=args.max_num_children,     # maximum allowed children number

        noise_dev=args.noise_dev,                   # the deviation of injected noise
        startup_steps=args.startup_size,            # warmup step
        discrete_mix=args.discrete_mix,             # mix ratio for discretization
        use_bottleneck=args.use_bottleneck,         # whether use the discretization model

        src_vocab_size=src_vocab.size() if args.use_src else None,  # source vocabulary size
        atn_num_layer=args.num_layer,                               # transformer layer number
        atn_num_heads=args.num_head,                                # attention heads
        atn_dp=args.atn_dp,                                         # dropout for transformer
    )
    logger.info(model)

    model.to(device)

    if args.optim == 'adam':
        optimizer = optim.Adam(filter(lambda p: p.requires_grad,
                                      model.parameters()), lr=args.lr, weight_decay=args.wd)
    elif args.optim == 'adagrad':
        optimizer = optim.Adagrad(filter(lambda p: p.requires_grad,
                                         model.parameters()), lr=args.lr, weight_decay=args.wd)
    elif args.optim == 'sgd':
        optimizer = optim.SGD(filter(lambda p: p.requires_grad,
                                     model.parameters()), lr=args.lr, weight_decay=args.wd)
    else:
        raise Exception("Unrecognized/Unsupported model optimizer {}".format(args.optim))

    start_epoch = 0
    global_step = 0
    best = -float('inf')

    # backup from saved checkpoints
    saved_checkpoints = '%s.pt' % os.path.join(args.save, args.expname)
    tmp_saved_checkpoints = '%s.tmp.pt' % os.path.join(args.save, args.expname)
    if os.path.isfile(saved_checkpoints):
        saved_states = torch.load(saved_checkpoints, map_location=device)

        logger.info("detected and loaded")
        model.load_state_dict(saved_states['model'])
        optimizer.load_state_dict(saved_states['optim'])
        start_epoch = saved_states['epoch'] + 1
        global_step = saved_states['global_step']
        best = saved_states['best']
    elif os.path.isfile(tmp_saved_checkpoints):
        saved_states = torch.load(tmp_saved_checkpoints, map_location=device)

        logger.info("temporary checkpoint detected and loaded")
        model.load_state_dict(saved_states['model'])
        optimizer.load_state_dict(saved_states['optim'])
        start_epoch = saved_states['epoch'] + 1
        global_step = saved_states['global_step']

    zglobal.global_update("global_step", global_step)

    # create trainer object for training and testing
    trainer = Trainer(args, model, optimizer, device, logger, epoch=start_epoch)

    if args.mode == "train":

        # load dataset splits
        train_dataset = TreeDataset(
            train_dir,
            vocab,
            src_path=train_dir if args.use_src else None,
            src_vocab=src_vocab if args.use_src else None,
            tree_depth_limit=args.max_depth,
            tree_size_limit=args.max_tree_size,
            src_len_limit=args.max_src_len,
        )
        logger.debug('==> Loading train data')

        dev_dataset = TreeDataset(
            dev_dir,
            vocab,
            src_path=dev_dir if args.use_src else None,
            src_vocab=src_vocab if args.use_src else None,
            tree_depth_limit=args.max_depth,
            tree_size_limit=args.max_tree_size,
            src_len_limit=args.max_src_len,
        )
        logger.debug('==> Loading dev data')

        logger.debug('Start training from epoch {}'.format(start_epoch))

        for epoch in range(start_epoch, args.epochs):
            train_loss = trainer.train(train_dataset)
            dev_loss, dev_acc, dev_pred, dev_repr = trainer.test(dev_dataset)

            global_step = zglobal.global_get('global_step')
            logger.info('==> Epoch {}, Step {}, Train \tLoss: {}'.format(epoch, global_step, train_loss))
            logger.info('==> Epoch {}, Step {}, Dev \tLoss: {}, ACC: {}'.format(epoch, global_step, dev_loss, dev_acc))

            # select best model according to accuracy, rather than development loss
            dev_score = dev_acc    # - dev_loss

            if best < dev_score:
                best = dev_score
                checkpoint = {
                    'model': trainer.model.state_dict(),
                    'optim': trainer.optimizer.state_dict(),
                    'args': args, 'epoch': epoch,
                    'global_step': zglobal.global_get('global_step'),
                    'best': best,
                }
                logger.debug('==> New optimum found, checkpointing everything now...')
                torch.save(checkpoint, '%s.pt' % os.path.join(args.save, args.expname))
                torch.save(checkpoint, '%s.%spt' % (os.path.join(args.save, args.expname), epoch))
                torch.save(dev_pred, '%s.dev.pred.%spt' % (os.path.join(args.save, args.expname), epoch))
                torch.save(dev_repr, '%s.dev.repr.%spt' % (os.path.join(args.save, args.expname), epoch))

    elif args.mode == "eval":

        test_dataset = TreeDataset(
            test_dir,
            vocab,
            src_path=test_dir if args.use_src else None,
            src_vocab=src_vocab if args.use_src else None,
            tree_depth_limit=args.max_depth,
            tree_size_limit=args.max_tree_size,
            src_len_limit=args.max_src_len,
        )
        logger.debug('==> Loading test data')

        # evaluating the test set
        test_loss, test_acc, test_pred, test_repr = trainer.test(test_dataset)
        torch.save(test_pred, '%s.test.pred.th' % os.path.join(args.save, args.expname))
        torch.save(test_repr, '%s.test.repr.th' % os.path.join(args.save, args.expname))

    else:
        raise Exception("Invalid training mode {}".format(args.mode))

    logger.debug('Ending')


if __name__ == "__main__":
    main()
