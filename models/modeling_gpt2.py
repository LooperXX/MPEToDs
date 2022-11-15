# coding=utf-8
# Copyright 2018 The OpenAI Team Authors and HuggingFace Inc. team.
# Copyright (c) 2018, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""PyTorch OpenAI GPT-2 model."""

from __future__ import absolute_import, division, print_function, unicode_literals

import torch
import torch.nn as nn
from torch.nn import CrossEntropyLoss
from transformers.modeling_gpt2 import GPT2PreTrainedModel, GPT2Model
import math

class GPT2LMHeadModel(GPT2PreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        self.transformer = GPT2Model(config)
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)
        self.classifier = nn.Linear(config.n_embd, 1)
        self.init_weights()

    def get_output_embeddings(self):
        return self.lm_head

    def prepare_inputs_for_generation(self, input_ids, past, **kwargs):
        # only last token for inputs_ids if past is defined in kwargs
        if past:
            input_ids = input_ids[:, -1].unsqueeze(-1)

        return {"input_ids": input_ids, "past": past}

    def forward(
            self,
            input_ids=None,
            past=None,
            attention_mask=None,
            token_type_ids=None,
            position_ids=None,
            head_mask=None,
            inputs_embeds=None,
            labels=None,
            type_loss=False
    ):
        transformer_outputs = self.transformer(
            input_ids,
            past=past,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
        )
        hidden_states = transformer_outputs[0]

        lm_logits = self.lm_head(hidden_states)
        classifier_output = None
        if type_loss is not None:
            classifier_output = self.classifier(hidden_states[:, :, :])

        outputs = (lm_logits,) + transformer_outputs[1:]
        if labels is not None:
            loss_fct1 = CrossEntropyLoss(ignore_index=-1, reduction='none')
            loss1 = loss_fct1(lm_logits.view(-1, lm_logits.size(-1)),
                              labels.view(-1))
            loss1 = loss1.view(labels.size(0), labels.size(1))
            label_size = torch.sum(labels != -1, dim=1).type(loss1.type())
            loss = torch.sum(loss1) / torch.sum(label_size)
            # ppl = torch.exp(torch.mean(torch.sum(loss1, dim=1).float()
            #                            / label_size.float()))
            # return loss, ppl
            return hidden_states, classifier_output, loss
        return lm_logits, outputs[1]
