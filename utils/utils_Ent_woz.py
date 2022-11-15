import json
import torch
import torch.utils.data as data
import torch.nn as nn
from utils.config import *
import ast
from transformers import GPT2Tokenizer
from utils.utils_general import *
import pickle

def read_langs(file_name, tokenizer, max_line=None):
    mylogger.info(("Reading lines from {}".format(file_name)))
    data, context_arr, conv_arr, kb_arr = [], [], [], []
    conv_GPT2, input_ids = [], []
    max_resp_len = 0

    with open('./data/fine-tune/MULTIWOZ2.1/global_entities.json') as f:
        global_entity = json.load(f)

    with open(file_name) as fin:
        cnt_lin, sample_counter = 1, 1
        for line in fin:
            line = line.strip()
            if line:
                if line[-1] == line[0] == "#":
                    line = line.replace("#", "")
                    task_type = line
                    continue

                nid, line = line.split(' ', 1)
                if '\t' in line:
                    u, r, gold_ent = line.split('\t')
                    gen_u = generate_memory(u, "$u", str(nid))
                    context_arr += gen_u
                    conv_arr += gen_u

                    conv_GPT2 += tokenizer.encode(USR + u.strip() + EOS)

                    # Get gold entity for each domain
                    gold_ent = ast.literal_eval(gold_ent)
                    ent_idx_restaurant, ent_idx_attraction, ent_idx_hotel = [], [], []
                    if task_type == "restaurant":
                        ent_idx_restaurant = gold_ent
                    elif task_type == "attraction":
                        ent_idx_attraction = gold_ent
                    elif task_type == "hotel":
                        ent_idx_hotel = gold_ent
                    ent_index = list(set(ent_idx_restaurant + ent_idx_attraction + ent_idx_hotel))

                    # Get local pointer position for each word in system response
                    ptr_index = []
                    for key in r.split():
                        index = [loc for loc, val in enumerate(context_arr) if (val[0] == key and key in ent_index)]
                        if (index):
                            index = max(index)
                        else:
                            index = len(context_arr)
                        ptr_index.append(index)

                    # Get global pointer labels for words in system response, the 1 in the end is for the NULL token
                    selector_index = [1 if (word_arr[0] in ent_index or word_arr[0] in r.split()) else 0 for word_arr in
                                      context_arr] + [1]

                    sketch_response_plain = generate_template(global_entity, r, gold_ent, kb_arr, task_type)
                    sketch_response_plain_ids = tokenizer.encode(SYS + sketch_response_plain.strip() + EOS)
                    binary_labels = [0] * len(sketch_response_plain_ids)
                    for sr_iid, sr_id in enumerate(sketch_response_plain_ids):
                        if sr_id > 52258:
                            binary_labels[sr_iid] = 1

                    counts = []
                    for i, token in enumerate(sketch_response_plain.split()):
                        if i == 0:
                            count = len(tokenizer.encode(token + ' '))
                        else:
                            count = len(tokenizer.encode(' ' + token))
                        if count > 1:
                            counts.append((i, count, ptr_index[i]))
                    index_add = 0
                    for item in counts:
                        index = item[0] + index_add
                        for i in range(item[1] - 1):
                            ptr_index.insert(index, item[2])
                            index_add += 1
                    ptr_index = [len(context_arr)] + ptr_index + [len(context_arr)]

                    input_ids = conv_GPT2 + sketch_response_plain_ids
                    if len(ptr_index) != len(sketch_response_plain_ids):
                        print('error in preprocessing')
                        exit(1)

                    data_detail = {
                        'context_arr': list(context_arr + [['$$$$'] * MEM_TOKEN_SIZE]),  # $$$$ is NULL token
                        'response': r,
                        'sketch_response_plain': sketch_response_plain,
                        'ptr_index': ptr_index,
                        'selector_index': selector_index,
                        'ent_index': ent_index,
                        'ent_idx_restaurant': list(set(ent_idx_restaurant)),
                        'ent_idx_attraction': list(set(ent_idx_attraction)),
                        'ent_idx_hotel': list(set(ent_idx_hotel)),
                        'conv_arr': list(conv_arr),
                        'conv_GPT2': list(conv_GPT2),
                        'input_ids': list(input_ids),
                        'binary_labels': binary_labels,
                        'kb_arr': list(kb_arr),
                        'id': int(sample_counter),
                        'ID': int(cnt_lin),
                        'domain': task_type}
                    data.append(data_detail)

                    gen_r = generate_memory(r, "$s", str(nid))
                    context_arr += gen_r
                    conv_arr += gen_r
                    conv_GPT2 += tokenizer.encode(SYS + r.strip() + EOS)

                    if max_resp_len < len(sketch_response_plain_ids):
                        max_resp_len = len(sketch_response_plain_ids)
                    sample_counter += 1
                else:
                    r = line
                    kb_info = generate_memory(r, "", str(nid))
                    context_arr = kb_info + context_arr
                    kb_arr += kb_info
            else:
                cnt_lin += 1
                context_arr, conv_arr, kb_arr = [], [], []
                conv_GPT2 = []
                if (max_line and cnt_lin >= max_line):
                    break

    return data, max_resp_len


def generate_template(global_entity, sentence, sent_ent, kb_arr, domain):
    """
    Based on the system response and the provided entity table, the output is the sketch response.
    """
    sketch_response = []
    if sent_ent == []:
        sketch_response = sentence.split()
    else:
        for word in sentence.split():
            if word not in sent_ent:
                sketch_response.append(word)
            else:
                ent_type = None
                for kb_item in kb_arr:
                    if word == kb_item[0]:
                        ent_type = kb_item[1]
                        break
                if ent_type is None:
                    for key in global_entity.keys():
                        global_entity[key] = [x.lower() for x in global_entity[key]]
                        if word in global_entity[key] or word.replace('_', ' ') in global_entity[key]:
                            ent_type = key
                            break
                assert ent_type is not None
                sketch_response.append('@{}@'.format(ent_type))
    sketch_response = " ".join(sketch_response)
    return sketch_response


def generate_memory(sent, speaker, time):
    sent_new = []
    sent_token = sent.split(' ')
    if speaker == "$u" or speaker == "$s":
        for idx, word in enumerate(sent_token):
            temp = [word, speaker, 'turn' + str(time), 'word' + str(idx)] + ["PAD"] * (MEM_TOKEN_SIZE - 4)
            sent_new.append(temp)
    else:
        sent_token = sent_token[::-1] + ["PAD"] * (MEM_TOKEN_SIZE - len(sent_token))
        sent_new.append(sent_token)
    return sent_new

def prepare_data_seq(tokenizer, batch_size=100, data_augmentation_file=None):
    file_train = './data/fine-tune/MULTIWOZ2.1/train.txt'
    file_test = './data/fine-tune/MULTIWOZ2.1/test.txt'
    if data_augmentation_file is not None:
        file_dev = './data/pre-train/MULTIWOZ2.1/dev.txt'
        pkl_name = './data/pre-train/MULTIWOZ2.1/pretrain_dataset.pkl'
    else:
        file_dev = './data/fine-tune/MULTIWOZ2.1/dev.txt'
        pkl_name = './data/fine-tune/MULTIWOZ2.1/fine_tune_dataset.pkl'

    if os.path.exists(pkl_name):
        [train, dev, test, lang, max_resp_len] = pickle.load(open(pkl_name, 'rb'))
        mylogger.info(" %s Loaded" % pkl_name)
    else:
        lang = Lang()
        pair_train, train_max_len = read_langs(file_train, tokenizer, max_line=args['max_line'])
        update_lang(pair_train, lang)
        pair_pretrain, pretrain_max_len = read_langs('./data/pre-train/MULTIWOZ2.1/' + args['data_augmentation_file'],
                                                     tokenizer,
                                                     max_line=args['max_line'])
        update_lang(pair_pretrain, lang)
        mylogger.info("Generate Lang")

        pair_dev, dev_max_len = read_langs(file_dev, tokenizer, max_line=args['max_line'])
        if data_augmentation_file is not None:
            pair_train, train_max_len = pair_pretrain, pretrain_max_len
            max_resp_len = max(train_max_len, dev_max_len) + 1
        else:
            pair_test, test_max_len = read_langs(file_test, tokenizer, max_line=args['max_line'])
            max_resp_len = max(train_max_len, dev_max_len, test_max_len) + 1

        train = get_seq(pair_train, lang, False)
        dev = get_seq(pair_dev, lang, False)
        if data_augmentation_file is not None:
            test = None
        else:
            test = get_seq(pair_test, lang, False)
        pickle.dump([train, dev, test, lang, max_resp_len], open(pkl_name, 'wb'))
        exit(1)

    mylogger.info("Read %s sentence pairs train" % len(train))
    mylogger.info("Read %s sentence pairs dev" % len(dev))
    if data_augmentation_file is None:
        mylogger.info("Read %s sentence pairs test" % len(test))

    train = torch.utils.data.DataLoader(dataset=train,
                                        batch_size=batch_size,
                                        shuffle=True,
                                        collate_fn=train.collate_fn)

    dev = torch.utils.data.DataLoader(dataset=dev,
                                      batch_size=batch_size,
                                      shuffle=False,
                                      collate_fn=dev.collate_fn)
    if data_augmentation_file is  None:
        test = torch.utils.data.DataLoader(dataset=test,
                                           batch_size=batch_size,
                                           shuffle=False,
                                           collate_fn=test.collate_fn)

    mylogger.info("Vocab_size: %s " % lang.n_words)
    mylogger.info("Max. length of system response: %s " % max_resp_len)
    mylogger.info("USE_CUDA={}".format(args['gpu']))

    return train, dev, test, lang, max_resp_len