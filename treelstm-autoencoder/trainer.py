# coding: utf-8

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import torch

import os
import util
import zglobal
import parallel


class Trainer(object):
    def __init__(self, args, model, optimizer, device, logger, epoch=0):
        super(Trainer, self).__init__()
        self.args = args
        self.model = model
        self.optimizer = optimizer
        self.device = device
        self.logger = logger
        self.epoch = epoch

    def save(self):
        checkpoint = {
            'model': self.model.state_dict(),
            'optim': self.optimizer.state_dict(),
            'args': self.args, 'epoch': self.epoch,
            'global_step': zglobal.global_get('global_step'),
        }
        self.logger.debug('==> New optimum found, checkpointing everything now...')
        torch.save(checkpoint, '%s.tmp.pt' % os.path.join(self.args.save, self.args.expname))

    # helper function for training
    def train(self, dataset):
        self.model.train()
        self.optimizer.zero_grad()
        total_loss, total_instance = 0.0, 0.0

        data_shards = []
        agg_counter = 0

        for dict_batch in dataset.batcher(
                self.args.batchsize,
                buffer_size=self.args.batchsize * self.args.buffersize,
                shuffle=True,
                train=True
        ):

            # put data on gpu0
            for key in dict_batch:
                if isinstance(dict_batch[key], torch.Tensor):
                    dict_batch[key] = dict_batch[key].to(self.device)

            # collect data shards
            data_shards.append(dict_batch)
            if len(data_shards) < self.args.shard_size:
                continue

            # start the parallel mode to process all the shards
            paral_outputs = parallel.parallel_model(self.model, data_shards, self.args.device)
            logits, tree_codes, tree_repr = list(zip(*paral_outputs))

            # collect losses
            loss = []
            acc = []
            for gpu_logits, gpu_sample in zip(logits, data_shards):
                gpu_loss = util.masked_loss(gpu_logits, gpu_sample['y'], mask=gpu_sample['y_m'])
                loss.append(gpu_loss)
                gpu_cc, gpu_tc, gpu_acc = util.masked_acc(gpu_logits, gpu_sample['y'], mask=gpu_sample['y_m'])
                acc.append(gpu_acc)
            loss = torch.cat(loss, 0).mean()

            # squash the loss, and compute gradient
            total_loss += loss.item()
            total_instance += sum([len(dict_batch['id'].tolist()) for dict_batch in data_shards])
            loss.backward()

            data_shards = []
            agg_counter += 1

            # update the model parameters
            if agg_counter % self.args.num_grad_agg == 0:
                self.optimizer.step()
                self.optimizer.zero_grad()

                global_step = zglobal.global_get('global_step')
                zglobal.global_update('global_step', global_step + 1)

                if global_step % self.args.disp_freq == 0 and global_step > 0:
                    acc = sum(acc) / len(acc)
                    self.logger.info(
                        "Epoch {} Step {} Loss {:.3f} ACC {:.3f}".format(self.epoch, global_step, loss, acc))

                if global_step % self.args.save_freq == 0 and global_step > 0:
                    self.save()

        self.epoch += 1
        return total_loss / total_instance

    # function used to process data shards, with focus of obtaining model outputs
    def shard_eval(self, data_shards):
        with torch.no_grad():
            paral_outputs = parallel.parallel_model(self.model, data_shards, self.args.device)
            logits, tree_codes, tree_repr = list(zip(*paral_outputs))

            loss = []
            scc, stc = 0., 0.
            predictions = []
            dataindices = []
            encodings = []
            for gpu_logits, gpu_codes, gpu_repr, gpu_sample in zip(logits, tree_codes, tree_repr, data_shards):
                gpu_loss = util.masked_loss(gpu_logits, gpu_sample['y'], mask=gpu_sample['y_m'])
                loss.append(gpu_loss)

                gpu_cc, gpu_tc, gpu_acc = util.masked_acc(gpu_logits, gpu_sample['y'], mask=gpu_sample['y_m'])
                scc += gpu_cc
                stc += gpu_tc

                for eidx, (idx, tc) in enumerate(zip(gpu_sample['id'].tolist(), gpu_codes.tolist())):
                    predictions.append(tc)
                    dataindices.append(int(idx))
                    encodings.append(gpu_repr[eidx])

            loss = torch.cat(loss, 0).sum()
            num_instance = sum([len(dict_batch['id'].tolist()) for dict_batch in data_shards])

            return loss, scc, stc, num_instance, predictions, dataindices, encodings

    # helper function for testing
    def test(self, dataset):
        self.model.eval()
        with torch.no_grad():
            total_loss, corr_count, total_count, total_instance = 0.0, 0.0, 0.0, 0.0
            predictions = []
            encodings = []
            dataindices = []

            data_shards = []

            for dict_batch in dataset.batcher(
                    self.args.batchsize,
                    buffer_size=self.args.batchsize * self.args.buffersize,
                    shuffle=False,
                    train=False
            ):
                # put data on gpu0
                for key in dict_batch:
                    if isinstance(dict_batch[key], torch.Tensor):
                        dict_batch[key] = dict_batch[key].to(self.device)

                data_shards.append(dict_batch)
                if len(data_shards) < self.args.shard_size:
                    continue

                shard_output = self.shard_eval(data_shards)

                total_loss += shard_output[0]
                corr_count += shard_output[1]
                total_count += shard_output[2]
                total_instance += shard_output[3]
                predictions += shard_output[4]
                dataindices += shard_output[5]
                encodings += shard_output[6]

                data_shards = []

        if len(data_shards) > 0:
            shard_output = self.shard_eval(data_shards)

            total_loss += shard_output[0]
            corr_count += shard_output[1]
            total_count += shard_output[2]
            total_instance += shard_output[3]
            predictions += shard_output[4]
            dataindices += shard_output[5]
            encodings += shard_output[6]

        # get input-order-based outputs based on data indices
        sorted_outputs = sorted(zip(dataindices, predictions, encodings), key=lambda x: x[0])
        _, predictions, encodings = list(zip(*sorted_outputs))
        self.logger.info("Processing {} elements".format(len(predictions)))
        encodings = [enc.unsqueeze(0) for enc in encodings]

        return total_loss / total_instance, corr_count * 100. / total_count, predictions, torch.cat(encodings, 0)
