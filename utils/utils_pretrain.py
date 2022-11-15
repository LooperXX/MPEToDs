import json
import torch
import torch.utils.data as data
import torch.nn as nn
from utils.config import *
import ast
import pickle


def _cuda(x):
    if args['gpu']:
        return x.cuda()
    else:
        return x


class PretrainDataset(data.Dataset):
    """Custom data.Dataset compatible with data.DataLoader."""

    def __init__(self, data_info):
        """Reads source and target sequences from txt files."""
        self.data_info = {}
        for k in data_info.keys():
            self.data_info[k] = data_info[k]

        self.num_total_seqs = len(data_info['conv_GPT2'])

    def __getitem__(self, index):
        """Returns one data pair (source and target)."""
        binary_labels = _cuda(torch.Tensor(self.data_info['binary_labels'][index]).contiguous())
        conv_GPT2 = _cuda(torch.LongTensor(self.data_info['conv_GPT2'][index]).contiguous())
        input_ids = _cuda(torch.Tensor(self.data_info['input_ids'][index]).contiguous())

        # processed information
        data_info = {}
        for k in self.data_info.keys():
            try:
                data_info[k] = locals()[k]
            except:
                data_info[k] = self.data_info[k][index]

        return data_info

    def __len__(self):
        return self.num_total_seqs

    def collate_fn(self, data):
        def merge_index(sequences):
            lengths = [len(seq) for seq in sequences]
            padded_seqs = torch.zeros(len(sequences), max(lengths)).float()
            for i, seq in enumerate(sequences):
                end = lengths[i]
                padded_seqs[i, :end] = seq[:end]
            return padded_seqs, lengths

        # sort a list by sequence length (descending order) to use pack_padded_sequence
        data.sort(key=lambda x: len(x['input_ids']), reverse=True)
        item_info = {}
        for key in data[0].keys():
            item_info[key] = [d[key] for d in data]

        # merge sequences
        binary_labels, _ = merge_index(item_info['binary_labels'])
        input_ids, input_ids_lengths = merge_index(item_info['input_ids'])
        input_ids = input_ids.long()

        # convert to contiguous and cuda
        binary_labels = _cuda(binary_labels.contiguous())
        input_ids = _cuda(input_ids.contiguous())

        # processed information
        data_info = {}
        for k in item_info.keys():
            try:
                data_info[k] = locals()[k]
            except:
                data_info[k] = item_info[k]

        # additional plain information
        data_info['input_ids_lengths'] = input_ids_lengths
        data_info['conv_GPT2_lengths'] = [len(sample) for sample in data_info['conv_GPT2']]
        return data_info


def read_langs(file_name, tokenizer, max_line=None):
    mylogger.info(("Reading lines from {}".format(file_name)))
    data, conv_GPT2, input_ids = [], [], []
    max_resp_len = 128

    with open(file_name, encoding='utf-8') as fin:
        cnt_lin, sample_counter = 1, 1
        for line in fin:
            line = line.strip()
            if line:
                if line[-1] == line[0] == "#":
                    task_type = line[1:-1]
                    continue

                if '\t' in line:
                    u, r, gold_ent, gold_type = line.split('\t')
                    cur_u = tokenizer.encode(u.strip())
                    # drop the long conv
                    if len(conv_GPT2) + len(cur_u) > max_resp_len:
                        conv_GPT2 = cur_u
                    else:
                        conv_GPT2 += cur_u

                        # Get gold entity for each domain
                    gold_ent = ast.literal_eval(gold_ent)
                    gold_type = ast.literal_eval(gold_type)

                    sketch_response_plain = generate_template(r, gold_ent, gold_type)
                    sketch_response_plain_ids = tokenizer.encode(sketch_response_plain.strip())
                    binary_labels = [0] * len(sketch_response_plain_ids)
                    for sr_iid, sr_id in enumerate(sketch_response_plain_ids):
                        if sr_id > 52258:
                            binary_labels[sr_iid] = 1

                    input_ids = conv_GPT2 + sketch_response_plain_ids
                    # drop the long dialog
                    if len(input_ids) > max_resp_len:
                        conv_GPT2 = []
                        continue

                    data_detail = {
                        'conv_GPT2': list(conv_GPT2),
                        'input_ids': list(input_ids),
                        'binary_labels': binary_labels,
                        'id': int(sample_counter),
                        'ID': int(cnt_lin),
                        'domain': task_type}
                    data.append(data_detail)

                    conv_GPT2 += tokenizer.encode(r.strip())

                    sample_counter += 1
            else:
                cnt_lin += 1
                conv_GPT2 = []
                if (max_line and cnt_lin >= max_line):
                    break

    return data, max_resp_len


def generate_template(sentence, sent_ent, sent_type):
    sentence = sentence.split()
    sketch_response = []
    # binary_labels = [0] * len(sentence)
    if sent_ent == []:
        sketch_response = sentence
    else:
        for word_id, word in enumerate(sentence):
            if word not in sent_ent:
                sketch_response.append(word)
            else:
                ent_type = sent_type[sent_ent.index(word)]
                sketch_response.append("@{}@".format(ent_type))
                # binary_labels[word_id] = 1
    sketch_response = " ".join(sketch_response)
    return sketch_response


def get_seq(pairs):
    data_info = {}
    for k in pairs[0].keys():
        data_info[k] = []

    for pair in pairs:
        for k in pair.keys():
            data_info[k].append(pair[k])
    dataset = PretrainDataset(data_info)
    return dataset


def prepare_data_seq(tokenizer, batch_size=100):
    file_train = './data/pre-train/pretrain_data.txt'
    file_dev = './data/pre-train/pretrain_dev_data.txt'
    if os.path.exists('./data/pre-train/dataset.pkl'):
        [train, dev, max_resp_len] = pickle.load(open('./data/pre-train/dataset.pkl', 'rb'))
        mylogger.info(" %s Loaded" % './data/pre-train/dataset.pkl')
    else:
        pair_train, train_max_len = read_langs(file_train, tokenizer, max_line=args['max_line'])
        pair_dev, dev_max_len = read_langs(file_dev, tokenizer, max_line=args['max_line'])
        max_resp_len = max(train_max_len, dev_max_len) + 1
        train = get_seq(pair_train)
        dev = get_seq(pair_dev)
        pickle.dump([train, dev, max_resp_len], open('./data/pre-train/dataset.pkl', 'wb'))
        exit(1)
    mylogger.info("Read %s sentence pairs train" % len(train))
    mylogger.info("Read %s sentence pairs dev" % len(dev))

    train = torch.utils.data.DataLoader(dataset=train,
                                        batch_size=batch_size,
                                        shuffle=True,
                                        collate_fn=train.collate_fn)
    dev = torch.utils.data.DataLoader(dataset=dev,
                                      batch_size=batch_size,
                                      shuffle=False,
                                      collate_fn=dev.collate_fn)
    mylogger.info("Max. length of system response: %s " % max_resp_len)
    mylogger.info("USE_CUDA={}".format(args['gpu']))

    return train, dev, max_resp_len
