#  transformer_chatbot
#  Copyright (C) 2018 Golovanov, Tselousov
#
#  This program is free software: you can redistribute it and/or modify
#  it under the terms of the GNU Affero General Public License as published by
#  the Free Software Foundation, either version 3 of the License, or
#  (at your option) any later version.
#
#  This program is distributed in the hope that it will be useful,
#  but WITHOUT ANY WARRANTY; without even the implied warranty of
#  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
#  GNU Affero General Public License for more details.
#
#  You should have received a copy of the GNU Affero General Public License
#  along with this program.  If not, see <http://www.gnu.org/licenses/>.

import json
import math
import os
import pickle
import random

import fasttext
import torch
from torch.utils.data import Dataset
from tqdm import tqdm

from model.seq2seq_vocab import Seq2seqTokenizer

SPECIAL_TOKENS = ['.', ',', '?', '!', ':']

class FacebookDataset(Dataset):
    @staticmethod
    def parse_data(path):
        with open(path, 'r', encoding='utf-8') as file:
            data = []
            for line in file.readlines():
                line = line.strip()

                if len(line) == 0:
                    continue

                space_idx = line.find(' ')
                if space_idx == -1:
                    dialog_idx = int(line)
                else:
                    dialog_idx = int(line[:space_idx])

                if int(dialog_idx) == 1:
                    data.append({'persona_info': [], 'dialog': []})

                dialog_line = line[space_idx + 1:].split('\t')
                dialog_line = [l.strip() for l in dialog_line]

                if dialog_line[0].startswith('your persona:'):
                    persona_info = dialog_line[0].replace('your persona: ', '')
                    if persona_info[-1] == '.' and persona_info[-2] != ' ':
                        persona_info = persona_info[:-1] + ' .'
                    data[-1]['persona_info'].append(persona_info)
                if dialog_line[0].startswith('partner\'s person'):
                    if not data[-1].__contains__('partner_persona_info'):
                        data[-1]['partner_persona_info'] = []
                    persona_info = dialog_line[0].replace('partner\'s persona: ', '')
                    if persona_info[-1] == '.' and persona_info[-2] != ' ':
                        persona_info = persona_info[:-1] + ' .'
                    data[-1]['partner_persona_info'].append(persona_info)

                elif len(dialog_line) > 1:
                    data[-1]['dialog'].append(dialog_line[0])
                    data[-1]['dialog'].append(dialog_line[1])

            return data

    @staticmethod
    def parse_data_emoji(path):
        with open(path, 'r', encoding='utf-8') as f:
            data = []
            for line in f.readlines():
                line = line.strip()
                items = line.split('\t')
                data.append({'persona_info': [], 'dialog': []})
                data[-1]['persona_info'].append(items[0])
                data[-1]['dialog'].append(items[1])
                data[-1]['dialog'].append(items[2])
            return data

    @staticmethod
    def parse_data_daily(path):
        with open(path, 'r', encoding='utf-8') as f:
            data = []
            for line in f.readlines():
                line = line.strip()
                items = line.split('\t')
                data.append({'persona_info': [], 'dialog': []})
                data[-1]['persona_info'].append(items[0])
                for i in range(1, len(items)):
                    data[-1]['dialog'].append(items[i])
            return data

    @staticmethod
    def parse_data_triple(path):
        with open(path, 'r', encoding='utf-8') as f:
            data = []
            for line in f.readlines():
                line = line.strip()
                if line.startswith('your persona:') or line.startswith('partner\'s persona:'):
                    continue
                items = line.split('\t')
                labels = []
                triples = []
                if len(items) > 2:
                    for i in range(2, len(items)):
                        string = items[i]
                        string = string.replace('\', \'', '\", \"')
                        string = string.replace('\',\'', '\",\"')
                        string = string.replace('\', \"', '\",\"')
                        string = string.replace('\", \'', '\",\"')
                        string = string.replace('[\'', '[\"')
                        string = string.replace('\']', '\"]')
                        triple = eval(string)
                        labels.append(triple[1])
                        triples.append(triple)
                data.append([items[1], labels, triples])
        return data

    @staticmethod
    def make_dataset(data, tokenizer):
        dataset = []
        for u in tqdm(data):
            utterance = tokenizer.encode(tokenizer.tokenize(u[0]))
            labels = []
            for label in u[1]:
                labels.append(tokenizer.convert_relation_to_label("<" + label + ">"))
            if len(labels) == 0:
                labels = []
            targets = []
            for triple in u[2]:
                targets.append(tokenizer.encode(tokenizer.tokenize(
                    ' '.join(["<" + triple[1] + ">", triple[0], ';', triple[2]]))))
            dataset.append([utterance, labels, targets])

        return dataset

    def __init__(self, paths, tokenizer, *, max_lengths=512,  max_y_length=80, min_infos=2,
                 dialog_embeddings=False, use_start_end=True, limit_size=-1,
                 cache=None, augment=False, aug_syn_proba=0.1, aug_vary_length=True, max_history_size=-1,
                 single_input=True, data_type='persona', parsed_data=None):
        assert min_infos > 0

        if isinstance(paths, str):
            paths = [paths]

        self.augment = augment
        self.aug_syn_proba = aug_syn_proba
        self.aug_vary_length = aug_vary_length

        self.tokenizer = tokenizer
        self.max_lengths = max_lengths
        self.max_y_length = max_y_length
        self.min_infos = min_infos
        self.dialog_embeddings = dialog_embeddings
        self.use_start_end = use_start_end
        self.max_history_size = max_history_size
        self.single_input = single_input
        self.data_type = data_type

        if cache and os.path.exists(cache):
            self.data = torch.load(cache)
        else:
            self.data = self._parse_data(paths, tokenizer, data_type, parsed_data)
            if cache:
                torch.save(self.data, cache)

        if limit_size > 0:
            self.data = self.data[:limit_size]

    def __len__(self):
        return len(self.data)

    def _parse_data(self, paths, tokenizer, data_type, parsed_data):
        data = None
        if data_type == 'triple':
            if not parsed_data:
                parsed_data = sum([FacebookDataset.parse_data_triple(path) for path in paths], [])
                data = FacebookDataset.make_dataset(parsed_data, tokenizer)
        return data

    def __getitem__(self, idx):
        d = self.data[idx]
        if self.use_start_end:
            d[0] = [self.tokenizer.bos_id] + d[0] + [self.tokenizer.eos_id]
            for i in range(len(d[2])):
                d[2][i] = d[2][i] + [self.tokenizer.eos_id]
        return d[0], d[1], d[2]

    # def __getitem__(self, idx):
    #     persona_info, dialog = self.data[idx]
    #
    #     if len(persona_info):
    #         persona_info = sum(persona_info, [])
    #         if self.single_input:
    #             persona_info = [self.vocab.bos_id] + persona_info
    #             if self.dialog_embeddings:
    #                 persona_info = [[tok, self.vocab.talker1_bos_id] for tok in persona_info]
    #         elif not self.single_input and not self.dialog_embeddings:
    #             persona_info = [self.vocab.bos_id] + persona_info[:self.max_lengths-2]
    #         else:
    #             persona_info = [self.vocab.info_bos_id] + persona_info[:self.max_lengths-2] + \
    #                            [self.vocab.info_eos_id] if self.use_start_end else persona_info[:self.max_lengths]
    #             if self.dialog_embeddings:
    #                 persona_info = [[tok, self.vocab.info_dialog_id] for tok in persona_info]
    #
    #     h = []
    #     history_start = 0
    #     if self.max_history_size != -1:
    #         history_start = -1 - self.max_history_size
    #     dialog_history = dialog[history_start: -1]
    #     if self.single_input:
    #         for i, ids in enumerate(dialog_history):
    #             if (len(dialog_history) - i) % 2 == 0:
    #                 ids = [self.vocab.talker1_bos_id] + ids
    #             else:
    #                 ids = [self.vocab.talker2_bos_id] + ids
    #             if self.dialog_embeddings:
    #                 ids = [[tok, self.vocab.talker1_bos_id if (len(dialog_history) - i) % 2 == 0
    #                         else self.vocab.talker2_bos_id] for tok in ids]
    #             h.extend(ids)
    #     elif not self.single_input and not self.dialog_embeddings:
    #         for i, ids in enumerate(dialog_history):
    #             if (len(dialog_history) - i) % 2 == 0:
    #                 ids = [self.vocab.talker1_bos_id] + ids
    #             else:
    #                 ids = [self.vocab.talker2_bos_id] + ids
    #             h.extend(ids)
    #     else:
    #         for i, ids in enumerate(dialog_history):
    #             if (len(dialog_history) - i) % 2 == 0 and self.use_start_end:
    #                 ids = [self.vocab.talker1_bos_id] + ids + [self.vocab.talker1_eos_id]
    #             elif self.use_start_end:
    #                 ids = [self.vocab.talker2_bos_id] + ids + [self.vocab.talker2_eos_id]
    #             if self.dialog_embeddings:
    #                 ids = [[tok, self.vocab.talker1_dialog_id if (len(dialog_history) - i) % 2 == 0
    #                         else self.vocab.talker2_dialog_id] for tok in ids]
    #             h.extend(ids)
    #         h = h[-self.max_lengths:]
    #
    #     sentences = []
    #     for y in (dialog[-1:]):
    #         if self.single_input:
    #             y = [self.vocab.talker1_bos_id] + y + [self.vocab.eos_id]
    #             if self.dialog_embeddings:
    #                 y = [[tok, self.vocab.talker1_bos_id] for tok in y]
    #             sentences.append(y)
    #         elif not self.single_input and not self.dialog_embeddings:
    #             y = [self.vocab.talker1_bos_id] + y + [self.vocab.eos_id]
    #             sentences.append(y)
    #         else:
    #             y = [self.vocab.bos_id] + y + [self.vocab.eos_id]
    #             if self.dialog_embeddings:
    #                 y = [[tok, self.vocab.sent_dialog_id] for tok in y]
    #             sentences.append(y)
    #
    #     return persona_info, h, sentences[0], sentences[1:]
