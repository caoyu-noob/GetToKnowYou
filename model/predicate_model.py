import torch

import torch.nn as nn
import torch.nn.functional as F


class PredicateModel(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(PredicateModel, self).__init__()
        self.linear = nn.Linear(input_dim, output_dim)

    def forward(self, input_state):
        output_state = self.linear(input_state)
        return F.sigmoid(output_state)


class PredicateClassifier(nn.Module):
    ''' Source: https://github.com/HLTCHKUST/Mem2Seq '''

    '''
    query: encoder_hidden_state
    memory: persona_sentences_ids
    '''

    def __init__(self, input_dim, output_dim, hop=3, dropout_p=0.1):
        super(PredicateClassifier, self).__init__()
        self.max_hops = hop
        self.hidden_dim = input_dim

        self.dropout = nn.Dropout(dropout_p)

        for hop in range(self.max_hops + 1):
            C = nn.Embedding(output_dim, self.hidden_dim)
            # C = nn.Embedding(self.n_relation, self.hidden_dim)  # TODO: dim 1 need to be vocab_size?
            C.weight.data.normal_(0, 0.1)
            self.add_module("C_{}".format(hop), C)
        self.C = AttrProxy(self, "C_")
        self.softmax = nn.Softmax(dim=1)

        self.classifier = nn.Linear(self.hidden_dim, output_dim)
        self.sigmoid = nn.Sigmoid()

    def init_hidden(self, bsz):
        weight = next(self.parameters())
        device = weight.device
        weight = weight.data
        hidden = weight.new(bsz, self.hidden_dim).to(device)
        return hidden

    def forward(self, input_ids, hidden_states):
        '''

        :param input_ids:  (bsz, len)
        :param hidden_states:  (bsz, hidden_size)
        :return:
        '''

        # u = [self.init_hidden(input_ids.size(0))]  # [(bsz, hidden_size)]
        u = [hidden_states]  # [(bsz, hidden_size)]
        for hop in range(self.max_hops):
            m_A = self.C[hop](input_ids.long())  # (bsz, len, hidden_size).

            u_temp = u[-1].unsqueeze(1).expand_as(m_A)  # (bsz, len, hidden_size)
            prob = self.softmax(torch.sum(m_A * u_temp, 2))  # (bsz, len). eq (1).

            m_C = self.C[hop + 1](input_ids.long())  # (bsz, len, hidden_size).
            prob = prob.unsqueeze(2).expand_as(m_C)

            o_k = torch.sum(m_C * prob, 1)  # eq (2).
            u_k = u[-1] + o_k  # eq (3).
            u.append(u_k)

        output_state = u[-1]  # (bsz, hidden_size)
        output_state = self.classifier(output_state)  # (bsz, n_relation)
        return F.sigmoid(output_state)


class AttrProxy(object):
    """
    Translates index lookups into attribute lookups.
    To implement some trick which able to use list of nn.Module in a nn.Module
    see https://discuss.pytorch.org/t/list-of-nn-module-in-a-nn-module/219/2
    """

    def __init__(self, module, prefix):
        self.module = module
        self.prefix = prefix

    def __getitem__(self, i):
        return getattr(self.module, self.prefix + str(i))