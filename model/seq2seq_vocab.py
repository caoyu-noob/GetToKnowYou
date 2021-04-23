import os
import pickle
import json

class Seq2seqTokenizer:
    def __init__(self, is_relation=False):
        self.word2idx = {"<unk>": 0, "<pad>": 1, "<bos>": 2, "<eos>": 3}
        self.idx2word = {0: "<unk>", 1: "<pad>", 2: "<bos>", 3: "<eos>"}
        self.n_words = 4
        self.all_special_ids = [0, 1, 2, 3]
        self.pad_id = 1
        self.bos_id = 2
        self.eos_id = 3
        self.unk_id = 0
        self.NO_RELATION = "<no_relation>"
        self.n_relations = 0
        self.no_relation_id = -1

    def tokenize(self, str):
        res = str.strip().split(' ')
        res = [x.lower() for x in res]
        return res

    def encode(self, tokenized_str):
        res = []
        for token in tokenized_str:
            if self.word2idx.__contains__(token):
                res.append(self.word2idx[token])
            else:
                res.append(self.unk_id)
        return res

    def decode(self, ids, skip_special_tokens=True, clean_up_tokenization_spaces=False):
        res = []
        for id in ids:
            if skip_special_tokens and id in self.all_special_ids:
                continue
            res.append(self.idx2word[id])
        text = ' '.join(res)
        return text

    def index_words(self, sentence):
        if isinstance(sentence, str):
            for word in sentence.split(' '):
                self.index_word(word)
        if isinstance(sentence, list):
            for word in sentence:
                self.index_word(word)

    def index_word(self, word):
        if not self.word2idx.__contains__(word):
            self.word2idx[word] = self.n_words
            self.idx2word[self.n_words] = word
            self.n_words += 1

    def update_relation(self, relations):
        self.index_words(relations)
        self.n_relations = len(relations)
        self.no_relation_id = self.word2idx[self.NO_RELATION]

    def convert_relation_to_label(self, relation):
        return self.word2idx[relation] - self.no_relation_id - 1

class Seq2seqVocab:
    def __init__(self, train_dataset_path, valid_dataset_path, test_dataset_path, vocab_path, data_type='triple'):
        if (os.path.exists(vocab_path)):
            with open(vocab_path, 'rb') as f:
                cached_data = pickle.load(f)
            self.vocab = cached_data[0]
        else:
            self.vocab = Seq2seqTokenizer()
            self.all_data = self._parse_data(train_dataset_path, valid_dataset_path, test_dataset_path, data_type)
            self.parse_vocab(self.all_data, self.vocab)
            with open(vocab_path, 'wb') as f:
                pickle.dump([self.vocab], f)

    def _parse_data(self, train_dataset_path, valid_dataset_path, test_dataset_path, data_type):
        data = None
        if data_type == 'triple':
            data = self.parse_data_persona(train_dataset_path, valid_dataset_path, test_dataset_path)
        return data

    def parse_data_persona(self, train_dataset_path, valid_dataset_path, test_dataset_path):
        subsets = [train_dataset_path, valid_dataset_path, test_dataset_path]
        all_data = []
        for subset in subsets:
            data = []
            if subset is None or len(subset) == 0:
                all_data.append(data)
                continue
            with open(subset, 'r', encoding='utf-8') as f:
                for line in f.readlines():
                    line = line.strip()
                    if line.startswith('your persona:') or line.startswith('partner\'s persona:'):
                        continue
                    items = line.split('\t')
                    labels = []
                    triples = []
                    if len(items) > 2:
                        for i in range(2, len(items)):
                            # triple_items = items[i].split(',')
                            # for j, triple_item in enumerate(triple_items):
                            #     if triple_item.startswith("\'") or triple_item.startswith(" \'") or \
                            #             triple_item.startswith("[\'"):
                            #         triple_items[j] = triple_item.replace("'", "\"")
                            # triple = json.loads(','.join(triple_items))
                            triple = eval(items[i])
                            labels.append(triple[1])
                            triples.append(triple)
                    data.append([items[1], labels, triples])
            all_data.append(data)
        return all_data

    def parse_vocab(self, all_data, vocab):
        relation_labels = set()
        for data in all_data:
            for d in data:
                vocab.index_words(d[0])
                for label in d[1]:
                    relation_labels.add("<" + label + ">")
                for triple in d[2]:
                    vocab.index_words(triple[0])
                    vocab.index_words(triple[2])
        relation_labels = list(relation_labels)
        relation_labels = [vocab.NO_RELATION] + relation_labels
        vocab.update_relation(relation_labels)