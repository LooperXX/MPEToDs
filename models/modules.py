import torch
import torch.nn as nn
from utils.config import *
from utils.utils_general import _cuda


class ExternalKnowledge(nn.Module):
    def __init__(self, vocab, embedding_dim, hop, dropout):
        super(ExternalKnowledge, self).__init__()
        self.max_hops = hop
        self.embedding_dim = embedding_dim
        self.dropout = dropout
        self.dropout_layer = nn.Dropout(dropout)
        for hop in range(self.max_hops + 1):
            C = nn.Embedding(vocab, embedding_dim, padding_idx=PAD_token)
            C.weight.data.normal_(0, 0.1)
            self.add_module("C_{}".format(hop), C)
        self.C = AttrProxy(self, "C_")
        self.softmax = nn.Softmax(dim=1)
        self.sigmoid = nn.Sigmoid()
        self.W = nn.Linear(args['embedding_dim'], embedding_dim)

    def add_lm_embedding(self, full_memory, kb_len, conv_len, hiddens):
        for bi in range(full_memory.size(0)):
            start, end = kb_len[bi], kb_len[bi] + conv_len[bi]
            full_memory[bi, start:end, :] = full_memory[bi, start:end, :] + hiddens[bi, :conv_len[bi], :]
        return full_memory

    def load_memory(self, story, kb_len, conv_len, hidden, dh_outputs):
        # TODO: ablationH ?
        # Forward multiple hop mechanism
        hidden = self.W(hidden)
        u = [hidden]
        story_size = story.size()
        self.m_story = []
        for hop in range(self.max_hops):
            embed_A = self.C[hop](story.contiguous().view(story_size[0], -1))  # .long()) # b * (m * s) * e
            embed_A = embed_A.view(story_size + (embed_A.size(-1),))  # b * m * s * e
            embed_A = torch.sum(embed_A, 2).squeeze(2)  # b * m * e
            if not args["ablationH"]:
                embed_A = self.add_lm_embedding(embed_A, kb_len, conv_len, dh_outputs)
            embed_A = self.dropout_layer(embed_A)

            if (len(list(u[-1].size())) == 1):
                u[-1] = u[-1].unsqueeze(0)  ## used for bsz = 1.
            u_temp = u[-1].unsqueeze(1).expand_as(embed_A)
            prob_logit = torch.sum(embed_A * u_temp, 2)
            prob_ = self.softmax(prob_logit)

            embed_C = self.C[hop + 1](story.contiguous().view(story_size[0], -1).long())
            embed_C = embed_C.view(story_size + (embed_C.size(-1),))
            embed_C = torch.sum(embed_C, 2).squeeze(2)
            if not args["ablationH"]:
                embed_C = self.add_lm_embedding(embed_C, kb_len, conv_len, dh_outputs)

            prob = prob_.unsqueeze(2).expand_as(embed_C)
            o_k = torch.sum(embed_C * prob, 1)
            u_k = u[-1] + o_k
            u.append(u_k)
            self.m_story.append(embed_A)
        self.m_story.append(embed_C)
        return self.sigmoid(prob_logit), u[-1]

    def forward(self, query_vector, global_pointer):
        query_vector = self.W(query_vector)
        u = [query_vector]
        for hop in range(self.max_hops):
            m_A = self.m_story[hop]
            if not args["ablationG"]:
                m_A = m_A * global_pointer.unsqueeze(2).expand_as(m_A)
            if (len(list(u[-1].size())) == 1):
                u[-1] = u[-1].unsqueeze(0)  ## used for bsz = 1.
            u_temp = u[-1].unsqueeze(1).expand_as(m_A)
            prob_logits = torch.sum(m_A * u_temp, 2)
            prob_soft = self.softmax(prob_logits)
            m_C = self.m_story[hop + 1]
            if not args["ablationG"]:
                m_C = m_C * global_pointer.unsqueeze(2).expand_as(m_C)
            prob = prob_soft.unsqueeze(2).expand_as(m_C)
            o_k = torch.sum(m_C * prob, 1)
            u_k = u[-1] + o_k
            u.append(u_k)
        return prob_soft, prob_logits


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
