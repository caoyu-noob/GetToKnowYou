import logging
import math
import random

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import CrossEntropyLoss
from torch.utils.data import DataLoader
from torch.utils.data import RandomSampler
from tqdm import tqdm
from itertools import chain

from model.seq2seq import TransformerSeq2Seq
from model.seq2seq_vocab import Seq2seqTokenizer
from .loss import LabelSmoothingLoss
from .optim import Adam
from .optim import NoamOpt
from .utils import pad_sequence

SPECIAL_TOKENS = ['<bos>', '<eos>', '<pad>', '<talker1_bos>', '<talker2_bos>', '<talker1_eos>', '<talker2_eos>',
                  '<info_bos>', '<info_eos>', '.', ',', '?', '!', ':']
MIX_IGNORE_TOKENS = ['.', ',', '!', '?', ';', ':', '-', '*', '=', ')', '(', '\'', '"', ]


class Trainer:
    def __init__(self, model, train_dataset, trainer_config, writer, logger=None, test_dataset=None, valid_dataset=None,
                 n_jobs=0, label_smoothing=0, device=torch.device('cuda'), evaluate_full_sequences=False,
                 ignore_idxs=[], full_input=False, max_length=511, max_y_length=80, new_dataset=False,
                 best_model_path='',
                 no_persona=False, mixup=False, mixup_mode='alternate', mixup_dataset=None,
                 mixup_ratio=0.15, bert_mixup=False, replace=False, pointer_gen=False):
        if logger is None:
            self.logger = logging.getLogger(__file__)
        else:
            self.logger = logger

        self.train_batch_size = trainer_config.train_batch_size
        self.test_batch_size = trainer_config.test_batch_size
        self.lr = trainer_config.lr
        self.lr_warmup = trainer_config.lr_warmup
        self.weight_decay = trainer_config.weight_decay
        self.batch_split = trainer_config.batch_split
        self.s2s_weight = 1
        self.single_input = True
        self.clip_grad = trainer_config.clip_grad
        self.n_epochs = trainer_config.n_epochs
        self.linear_schedule = trainer_config.linear_schedule
        self.patience = trainer_config.patience
        self.model_saving_interval = trainer_config.model_saving_interval
        self.device = device
        self.no_persona = no_persona
        self.evaluate_full_sequences = evaluate_full_sequences
        self.global_step = 0
        self.full_input = full_input
        self.max_length = max_length
        self.max_y_length = max_y_length
        self.new_dataset = new_dataset
        self.best_loss = 1e5
        self.best_model_path = best_model_path
        self.model_type = 'pretrain'
        self.patience_cnt = 0
        self.stop_training = False
        self.pointer_gen = pointer_gen

        self.loss_lambda = trainer_config.loss_lambda

        self.model = model.to(device)

        self.criterion = LabelSmoothingLoss(n_labels=self.model.n_embeddings, smoothing=label_smoothing,
                                            ignore_index=self.model.padding_idx).to(device)

        param_optimizer = list(self.model.named_parameters())
        # Here we should remove parameters which are not used during to avoid breaking apex with None grads
        self.loss_weight = None
        no_decay = ['bias', 'loss']
        optimizer_grouped_parameters = [
            {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)],
             'weight_decay': self.weight_decay},
            {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
        ]

        base_optimizer = Adam(optimizer_grouped_parameters, lr=self.lr)

        if not self.linear_schedule:
            self.optimizer = NoamOpt(self.model.embeddings_size, self.lr_warmup, base_optimizer, lr=self.lr,
                                     linear_schedule=False, loss_weight=self.loss_weight)
        else:
            total_steps = len(train_dataset) * self.n_epochs // self.train_batch_size
            self.optimizer = NoamOpt(self.model.embeddings_size, self.lr_warmup, base_optimizer, linear_schedule=True,
                                     lr=self.lr, total_steps=total_steps, loss_weight=self.loss_weight)

        train_sampler = RandomSampler(train_dataset)
        self.train_dataloader = DataLoader(train_dataset, batch_size=self.train_batch_size // self.batch_split,
                                           sampler=train_sampler,
                                           num_workers=n_jobs, collate_fn=self.collate_func)
        self.train_dataset = train_dataset  # used to sample negative examples
        if test_dataset is not None:  # only do evaluation on main process
            self.test_dataloader = DataLoader(test_dataset, batch_size=self.test_batch_size, shuffle=False,
                                              num_workers=n_jobs, collate_fn=self.collate_func)
        if valid_dataset is not None:
            self.valid_dataloader = DataLoader(valid_dataset, batch_size=self.test_batch_size, shuffle=False,
                                               num_workers=n_jobs, collate_fn=self.collate_func)

        self.tokenizer = train_dataset.tokenizer
        self.writer = writer

        if isinstance(self.model, TransformerSeq2Seq):
            self.model_type = 'seq2seq'

    def state_dict(self):
        return {'model': self.model.state_dict(),
                'optimizer': self.optimizer.state_dict(),
                'global_step': self.global_step}

    def load_state_dict(self, state_dict):
        if state_dict.__contains__('model') and state_dict.__contains__('optimizer'):
            self.model.load_state_dict(state_dict['model'], strict=False)
            self.optimizer.load_state_dict(state_dict['optimizer'])
            self.global_step = state_dict['global_step']
        else:
            self.model.load_state_dict(state_dict, strict=False)

    def collate_func(self, data):
        utterances, relations, targets = zip(*data)

        contexts = [torch.tensor(u, dtype=torch.long) for u in utterances]

        decoder_index, y_out = [], []
        predicate_x_index, predicate_y_index = [], []
        for i in range(len(targets)):
            if len(targets[i]) > 0:
                decoder_index.extend([i] * len(targets[i]))
                y_out.extend([torch.tensor(d, dtype=torch.long) for d in targets[i]])
            predicate_x_index.extend([i] * len(relations[i]))
            predicate_y_index.extend(relations[i])
        y_out = pad_sequence(y_out, batch_first=True, padding_value=self.model.padding_idx)
        input_ids = pad_sequence(contexts, batch_first=True, padding_value=self.model.padding_idx)
        predicate_labels = torch.sparse_coo_tensor(torch.tensor([predicate_x_index, predicate_y_index]),
                                       torch.tensor([1 for _ in range(len(predicate_y_index))], dtype=torch.float32),
                                       [len(relations), self.tokenizer.n_relations - 1]).to_dense()
        decoder_index = torch.tensor(decoder_index, dtype=torch.long)
        return input_ids, predicate_labels, decoder_index, y_out

        # contexts = []
        #
        # if max(map(len, persona_info)) > 0:
        #     persona_info = [torch.tensor(d, dtype=torch.long) for d in persona_info]
        #     contexts.append(persona_info)
        #
        # if max(map(len, h)) > 0:
        #     h = [torch.tensor(d, dtype=torch.long) for d in h]
        #     contexts.append(h)
        #
        # y_out = [torch.tensor(d, dtype=torch.long) for d in y]
        #
        # if self.no_persona:
        #     for c in contexts[1]:
        #         c[0][0] = self.vocab.bos_id
        #     y_out = [torch.cat(pieces, dim=0) for pieces in zip(*([contexts[1]] + [y_out]))]
        #     lengths = [(contexts[1][i].size(0), y_out[i].size(0)) for i in range(len(y_out))]
        #     contexts = lengths
        # else:
        #     y_out1 = [torch.cat(pieces, dim=0) for pieces in zip(*(contexts))]
        #     lengths = [(contexts[0][i].size(0) + contexts[1][i].size(0), y_out[i].size(0)) for i in
        #                range(len(y_out))]
        #     y_out = (y_out1, y_out)
        #     contexts = lengths
        #
        # # Pad now so we pad correctly when we have only a single input (context concatenated with y)
        # if isinstance(y_out, tuple):
        #     y_out = (
        #     [y[-(self.max_length - 1):] for y in y_out[0]], [y[:(self.max_y_length - 1)] for y in y_out[1]])
        # else:
        #     y_out = [y[-(self.max_length - 1):] for y in y_out]
        # contexts = [c if c[1] <= self.max_length - 1 else (c[0] - (c[1] - self.max_length + 1), self.max_length - 1)
        #             for c in contexts]
        # if isinstance(y_out, tuple):
        #     y_out = (pad_sequence(y_out[0], batch_first=True, padding_value=self.model.padding_idx),
        #              pad_sequence(y_out[1], batch_first=True, padding_value=self.model.padding_idx))
        # else:
        #     y_out = pad_sequence(y_out, batch_first=True, padding_value=self.model.padding_idx)
        #
        # return contexts, y_out

    def _s2s_loss(self, targets, enc_contexts, negative_samples):
        hidden_state, padding_mask = None, None

        nexts = targets[:, 1:].contiguous() if targets.dim() == 2 else targets[:, 1:, 0].contiguous()
        outputs, _, _ = self.model.decode(targets[:, :-1].contiguous(), enc_contexts)
        if self.full_input:
            for i in range(targets.shape[0]):
                for j in range(targets.shape[1]):
                    if targets[i][j][1] == self.vocab.sent_dialog_id:
                        nexts[i][: j] = self.model.padding_idx
                        break

        outputs = outputs.view(-1, outputs.shape[-1]).float()
        nexts = nexts.view(-1)

        loss = self.criterion(F.log_softmax(outputs, dim=-1), nexts) if self.model.training \
            else self.lm_criterion(outputs, nexts)
        return loss, hidden_state, padding_mask

    def optimizer_step(self, s2s_loss, pred_loss, full_loss):
        if self.clip_grad is not None:
            for group in self.optimizer.param_groups:
                nn.utils.clip_grad_norm_(group['params'], self.clip_grad)

        self.optimizer.step()
        self.optimizer.zero_grad()

        global_step = max(self.global_step, 0)
        self.writer.add_scalar("training/s2s_loss", s2s_loss, global_step=global_step)
        self.writer.add_scalar("training/pred_loss", pred_loss, global_step=global_step)
        self.writer.add_scalar("training/full_loss", full_loss, global_step=global_step)
        self.writer.add_scalar("training/lr", self.optimizer.get_lr(), global_step=global_step)

        self.global_step += 1

    def _eval_train(self, epoch, risk_func=None):  # add ppl and hits@1 evaluations
        self.model.train()

        tqdm_data = tqdm(self.train_dataloader, desc='Train (epoch #{})'.format(epoch))
        s2s_loss = 0
        pred_loss = 0
        for i, batch in enumerate(tqdm_data):
            input_ids, predicate_labels, decoder_index, targets = batch[0].to(self.device), \
                    batch[1].to(self.device), batch[2].to(self.device), batch[3].to(self.device)
            loss = self.model(input_ids, decoder_index, predicate_labels, targets)
            s2s_loss = (i * s2s_loss + loss[0].item()) / (i + 1)
            pred_loss = (i * pred_loss + loss[1].item()) / (i + 1)
            loss = (1 - self.loss_lambda) * loss[0] + self.loss_lambda * loss[1]
            full_loss = loss / self.batch_split
            tqdm_data.set_postfix({'s2s_loss': s2s_loss, 'pred_loss': pred_loss})

            # optimization
            full_loss = self.optimizer.backward(full_loss)
            if self.pointer_gen and (torch.isnan(self.model.generator.p_gen_linear._parameters['weight']._grad[0][0]) or \
                                     torch.isinf(self.model.generator.p_gen_linear._parameters['weight']._grad[0][0])):
                self.optimizer.zero_grad()
                self.logger.info('Abnormal gradient')

            if (i + 1) % self.batch_split == 0:
                self.optimizer_step(s2s_loss, pred_loss, full_loss)
        if (i + 1) % self.batch_split != 0:
            self.optimizer_step(s2s_loss, pred_loss, full_loss)

    def _get_eval_loss(self, input_ids, decoder_index, predicate_labels, targets, metrics, index):
        results = self.model(input_ids, decoder_index, predicate_labels, targets, output_logits=True, output_encoder=True)

        metrics['s2s_loss'] = (metrics['s2s_loss'] * index + results[0].item()) / (index + 1)
        metrics['pred_loss'] = (metrics['pred_loss'] * index + results[1].item()) / (index + 1)
        metrics['full_loss'] = (metrics['full_loss'] * index +
                        (self.loss_lambda * results[1].item() + ((1 - self.loss_lambda) * results[0].item()))) / (index + 1)
        predict_predicate = (results[3] >= 0.5).float()
        predict_predicate_num = torch.sum(predict_predicate).item()
        predict_predicate_acc_num = torch.sum((predict_predicate == predicate_labels) * predicate_labels).item()
        predicate_label_num = torch.sum(predicate_labels).item()
        metrics['pred_num'] += predict_predicate_num
        metrics['pred_acc_num'] += predict_predicate_acc_num
        metrics['label_num'] += predicate_label_num
        metrics['pred_acc'] = metrics['pred_acc_num'] / metrics['pred_num'] if metrics['pred_num'] != 0 else 0
        recall = metrics['pred_acc_num'] / metrics['label_num'] if metrics['label_num'] != 0 else 0
        f1 = 2 * metrics['pred_acc'] * recall / (metrics['pred_acc'] + recall) if metrics['pred_acc'] + recall != 0 else 0
        metrics['pred_f1'] = f1
        encoder_outputs_tuple = results[4:]
        return metrics, predict_predicate, encoder_outputs_tuple

    def _get_eval_predictions(self, input_ids, encoder_outputs_list, predict_predicate, decoder_index, targets):
        references, predictions, predictions_given_predicate = \
            [[] for _ in range(predict_predicate.size(0))], [[] for _ in range(predict_predicate.size(0))], \
            [[] for _ in range(predict_predicate.size(0))]
        ## Obtain the references for each sample, each reference is a list of tuple
        for i in range(decoder_index.size(0)):
            relation_string = self.tokenizer.decode([targets[i][0].item()])
            string = self.tokenizer.decode(targets[i][1:].tolist(), skip_special_tokens=True,
                                           clean_up_tokenization_spaces=False)
            subject_string = string.split(';')[0].strip()
            object_string = string.split(';')[1].strip()
            references[decoder_index[i].item()].append((subject_string, relation_string, object_string))

        ## Obtain the predictions for each sample if given the correct predicate
        if targets.size(0) > 0:
            start_ids = targets[:, :1]
            model_predictions = self.model.inference(input_ids, encoder_outputs_list, decoder_index, start_ids)
            for i in range(len(model_predictions)):
                relation_string = self.tokenizer.decode([start_ids[i].item()])
                string = self.tokenizer.decode(model_predictions[i], skip_special_tokens=True,
                                               clean_up_tokenization_spaces=False)
                spit_items = string.split(';')
                if len(spit_items) >= 2:
                    subject_string = spit_items[0].strip()
                    object_string = spit_items[1].strip()
                else:
                    if len(string) > 0:
                        split_items = string.split(' ')
                        subject_string = spit_items[0]
                        object_string = ' '.join(split_items[1:])
                    else:
                        subject_string, object_string = '', ''
                predictions_given_predicate[decoder_index[i].item()].append((subject_string, relation_string, object_string))

        ## Obtain the predictions for each sample using the predicted predicate
        nonzero_index = predict_predicate.nonzero()
        if nonzero_index.size(0) > 0:
            start_ids = nonzero_index[:, 1:] + self.tokenizer.no_relation_id + 1
            predict_decoder_index = nonzero_index[:, 0]
            model_predictions = self.model.inference(input_ids, encoder_outputs_list, predict_decoder_index, start_ids)
            for i in range(len(model_predictions)):
                relation_string = self.tokenizer.decode([start_ids[i].item()])
                string = self.tokenizer.decode(model_predictions[i], skip_special_tokens=True,
                                               clean_up_tokenization_spaces=False)
                spit_items = string.split(';')
                if len(spit_items) >= 2:
                    subject_string = spit_items[0].strip()
                    object_string = spit_items[1].strip()
                else:
                    if len(string) > 0:
                        split_items = string.split(' ')
                        subject_string = spit_items[0]
                        object_string = ' '.join(split_items[1:])
                    else:
                        subject_string, object_string = '', ''
                predictions[predict_decoder_index[i].item()].append((subject_string, relation_string, object_string))

        return references, predictions, predictions_given_predicate

    def _eval_test(self, metric_funcs={}, external_metrics_func=None, epoch=-1, inference=False, is_best=False,
                   raw_entail_data=None):
        with torch.no_grad():
            self.model.eval()
            if epoch == -1:
                tqdm_data = tqdm(self.test_dataloader, desc='Test')
                self.logger.info('Starting testing on Test dataset')
            else:
                tqdm_data = tqdm(self.valid_dataloader, desc='Test')
                self.logger.info('Starting testing on Valid dataset')
            metrics = {name: 0 for name in
                       ('s2s_loss', 'pred_loss', 'full_loss', 'pred_acc', 'pred_f1', 'pred_num', 'pred_acc_num', 'label_num')
                       + tuple(metric_funcs.keys())}
            full_predictions, full_references, full_predictions_given_predicate = [], [], []
            for i, batch in enumerate(tqdm_data):
                '''Get the loss, ppl for each batch'''
                input_ids, predicate_labels, decoder_index, targets = batch[0].to(self.device), \
                              batch[1].to(self.device), batch[2].to(self.device), batch[3].to(self.device)
                metrics, predict_predicate, encoder_outputs_tuple = self._get_eval_loss(input_ids, decoder_index,
                        predicate_labels, targets, metrics, i)
                # full sequence loss
                cur_references, cur_predictions, cur_predictions_given_predicate = self._get_eval_predictions(
                        input_ids, encoder_outputs_tuple, predict_predicate, decoder_index, targets)
                full_predictions.extend(cur_predictions)
                full_predictions_given_predicate.extend(cur_predictions_given_predicate)
                full_references.extend(cur_references)
                tqdm_data.set_postfix({'s2s_loss': metrics['s2s_loss'], 'pred_loss': metrics['pred_loss'],
                                       'full_loss': metrics['full_loss'], 'pred_acc': metrics['pred_acc'],
                                       'pred_f1': metrics['pred_f1']})

            if external_metrics_func and self.evaluate_full_sequences:
                external_metrics = external_metrics_func(full_references, full_predictions,
                                                         full_predictions_given_predicate, epoch, is_best)
                metrics.update(external_metrics)

            # logging
            global_step = max(self.global_step, 0)
            if self.writer is not None:
                for key, value in metrics.items():
                    self.writer.add_scalar("eval/{}".format(key), value, global_step=global_step)
            self.logger.info(metrics)

            if epoch != -1:
                if metrics['full_loss'] < self.best_loss:
                    self.logger.info('Current loss BEATS the previous best one, previous best is %.5f', self.best_loss)
                    self.best_loss = metrics['full_loss']
                    torch.save(self.model.state_dict(), self.best_model_path)
                    self.logger.info('Best model is saved on epoch %d', epoch)
                else:
                    self.patience_cnt += 1
                    self.logger.info('Current ppl CANNOT BEATS the previous best one, previous best is %.5f',
                                     self.best_loss)
                    if self.patience > 0 and self.patience_cnt > self.patience:
                        self.stop_training = True
            if epoch % self.model_saving_interval == 0 and epoch >= self.model_saving_interval and \
                    self.model_type in ['seq2seq']:
                torch.save(self.model.state_dict(), self.best_model_path + '_' + str(epoch))

    def _clip_grad_norm(self, grads, max_norm, norm_type=2):
        max_norm = float(max_norm)
        norm_type = float(norm_type)
        if norm_type == float('inf'):
            total_norm = max(grad.data.abs().max() for grad in grads)
        else:
            total_norm = 0
            for grad in grads:
                grad_norm = grad.data.norm(norm_type)
                total_norm += grad_norm ** norm_type
            total_norm = total_norm ** (1. / norm_type)
        clip_coef = max_norm / (total_norm + 1e-6)
        if clip_coef < 1:
            for grad in grads:
                grad.data.mul_(clip_coef)
        return total_norm

    def test(self, metric_funcs={}, external_metrics_func=None, epoch=-1, inference=False, raw_entail_data=None):
        if hasattr(self, 'valid_dataloader') or hasattr(self, 'test_dataloader'):
            self._eval_test(metric_funcs, external_metrics_func, epoch, inference, raw_entail_data=raw_entail_data)
            if epoch == -1 and not inference:
                self.logger.info('Loading the best model...')
                state_dict = torch.load(self.best_model_path, map_location=self.device)
                if state_dict.__contains__('model'):
                    self.model.load_state_dict(state_dict['model'], strict=False)
                else:
                    self.model.load_state_dict(state_dict)
                self._eval_test(metric_funcs, external_metrics_func, epoch, inference, is_best=True)

    def train(self, after_epoch_funcs=[], risk_func=None):
        for epoch in range(1, self.n_epochs + 1):
            self.logger.info('===============================')
            self.logger.info('Start training on Epoch %d', epoch)
            self._eval_train(epoch, risk_func)
            # self._eval_test()

            for func in after_epoch_funcs:
                func(epoch)
            self.logger.info('End training on Epoch %d', epoch)
            self.logger.info('===============================')
            if self.stop_training:
                self.logger.info('Training will be STOPPED in advance due to exceeding patience number')
                break

        for func in after_epoch_funcs:
            func(-1)
