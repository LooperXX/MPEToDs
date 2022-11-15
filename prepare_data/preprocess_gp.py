import os, json, logging, ast
import numpy as np
import matplotlib.pyplot as plt

logging.basicConfig(
    format='%(asctime)s - %(levelname)s - %(name)s -\t%(message)s',
    datefmt='%Y/%d/%m %H:%M:%S', level=logging.INFO)
logger = logging.getLogger(__name__)

EOS = '<|endoftext|>'
SYS = '<|SYS|>'
USR = '<|USR|>'

train_file = './data/pre-train/pretrain_data.txt'
dev_file = './data/pre-train/pretrain_dev_data.txt'

length, length_dev = [], []
total, total_dev = [], []
max_seq_len = 128
slot_types = set()

types_replace_woz = {
    '@addr@': '@address@', '@post@': '@postcode@', '@price@': '@pricerange@'
}

types_bad_MSR_E2E = ['@other@', '@closing@', '@result@', '@greeting@']


def divide_dialogue(input):
    result = []
    sens_len = sum([len(sen.split('\t')[0].split(' ')) for sen in input])
    if sens_len <= max_seq_len:
        result.append(input)
    else:
        if len(input) == 2:
            return []
        divide = len(input) // 2
        if divide % 2 == 1:
            divide -= 1
        result.extend(divide_dialogue(input[:divide]))
        result.extend(divide_dialogue(input[divide:]))
    return result

def blank(sen):
    for token in [',', '.', '?', '!', "'", '-', ':']:
        if token in sen:
            sen = sen.replace(token, " {} ".format(token))
    return sen.replace('  ', ' ').replace('  ', ' ')


def trans(word):
    return word.replace(' ', '_')


def clean(text):
    return blank(text.replace('\t', ' ').replace('\n', ' ').replace('\r', ' ').replace('  ', ' ').replace('  ',
                                                                                                          ' ')).lower().strip()


################## Schema #################################

def get_data_Schema(file_path, output_file):
    result = []
    for name in os.listdir(file_path):
        if name == 'schema.json':
            continue
        fullname = os.path.join(file_path, name)
        with open(fullname, 'r') as f:
            data = json.load(f)
            for dialog in data:
                sens = [turn['utterance'] for turn in dialog['turns']]
                speakers = [turn['speaker'] for turn in dialog['turns']]
                if len(sens) < 2:
                    continue
                assert len(sens) % 2 == 0
                assert speakers[0] == 'USER'
                # assert the format of 'user agent user agent user agent...'
                flag = False
                for spe in speakers:
                    if flag:
                        assert spe == 'SYSTEM'
                        flag = False
                    else:
                        assert spe == 'USER'
                        flag = True

                gold_entity, gold_type = [], []
                for turn_id, turn in enumerate(dialog['turns']):
                    gold_entity.append([])
                    gold_type.append([])
                    for frame in turn['frames']:
                        slots = frame['slots']
                        if slots != []:
                            for slot in slots:
                                slot_type = '@{}@'.format(slot['slot'].strip()).lower()
                                slot_types.add(slot_type)
                                slot_value = clean(sens[turn_id][slot['start']:slot['exclusive_end']])
                                if slot_value not in gold_entity[-1]:
                                    gold_entity[-1].append(slot_value)
                                    gold_type[-1].append(slot_type)
                assert len(sens) == len(gold_entity)

                for i in range(0, len(sens)):
                    sens[i] = clean(sens[i])
                    if speakers[i] == "USER":
                        sens[i] = '{} {} {}\t'.format(USR, sens[i], EOS)
                    else:
                        for entity_id, entity in enumerate(gold_entity[i]):
                            if entity not in sens[i]:
                                gold_entity[i].pop(entity_id)
                                gold_type[i].pop(entity_id)
                            else:
                                sens[i] = sens[i].replace(entity, trans(entity))
                                gold_entity[i][entity_id] = trans(entity)
                        sens[i] = '{} {} {}\t{}\t{}\n'.format(SYS, sens[i], EOS, gold_entity[i], gold_type[i])
                sens = divide_dialogue(sens)
                for dialog in sens:
                    result.append('#Schema#\n' + ''.join(dialog) + '\n')

    logger.info('Number of dialogues in the Training File:\t%s' % len(result))
    total.append(len(result))
    length.extend([sum([len(turn.split('\t')[0].split(' ')) + len(turn.split('\t')[1].split(' ')) for turn in i.split('\n')[1:-2]]) for i in result])
    with open(output_file, 'a') as f:
        f.writelines(result)


def get_dev_data_Schema(file_path, output_file):
    result = []
    for name in os.listdir(file_path):
        if name == 'schema.json':
            continue
        fullname = os.path.join(file_path, name)
        with open(fullname, 'r') as f:
            data = json.load(f)
            for dialog in data:
                sens = [turn['utterance'] for turn in dialog['turns']]
                speakers = [turn['speaker'] for turn in dialog['turns']]
                if len(sens) < 2:
                    continue
                assert len(sens) % 2 == 0
                assert speakers[0] == 'USER'
                # assert the format of 'user agent user agent user agent...'
                flag = False
                for spe in speakers:
                    if flag:
                        assert spe == 'SYSTEM'
                        flag = False
                    else:
                        assert spe == 'USER'
                        flag = True

                gold_entity, gold_type = [], []
                for turn_id, turn in enumerate(dialog['turns']):
                    gold_entity.append([])
                    gold_type.append([])
                    for frame in turn['frames']:
                        slots = frame['slots']
                        if slots != []:
                            for slot in slots:
                                slot_type = '@{}@'.format(slot['slot'].strip()).lower()
                                slot_types.add(slot_type)
                                slot_value = clean(sens[turn_id][slot['start']:slot['exclusive_end']])
                                if slot_value not in gold_entity[-1]:
                                    gold_entity[-1].append(slot_value)
                                    gold_type[-1].append(slot_type)
                assert len(sens) == len(gold_entity)

                for i in range(0, len(sens)):
                    sens[i] = clean(sens[i])
                    if speakers[i] == "USER":
                        sens[i] = '{} {} {}\t'.format(USR, sens[i], EOS)
                    else:
                        for entity_id, entity in enumerate(gold_entity[i]):
                            if entity not in sens[i]:
                                gold_entity[i].pop(entity_id)
                                gold_type[i].pop(entity_id)
                            else:
                                sens[i] = sens[i].replace(entity, trans(entity))
                                gold_entity[i][entity_id] = trans(entity)
                        sens[i] = '{} {} {}\t{}\t{}\n'.format(SYS, sens[i], EOS, gold_entity[i], gold_type[i])
                sens = divide_dialogue(sens)
                for dialog in sens:
                    result.append('#Schema#\n' + ''.join(dialog) + '\n')
    logger.info('Number of dialogues in the Dev File:\t%s' % len(result))
    total_dev.append(len(result))
    length_dev.extend([sum([len(turn.split('\t')[0].split(' ')) + len(turn.split('\t')[1].split(' ')) for turn in i.split('\n')[1:-2]]) for i in result])
    with open(output_file, 'a') as f:
        f.writelines(result)


################## Taskmaster-1 #################################

def get_id_list_Taskmaster(file_path):
    names = ['train', 'dev', 'test']
    result = []
    for name in names:
        fullname = os.path.join(file_path, name + '.csv')
        with open(fullname, 'r') as f:
            lines = f.readlines()
            lines = [line.strip()[:-1] for line in lines]
            result.append(lines)
    return result


def get_data_Taskmaster(oneperson_path, twoperson_path, output_file, ids):
    result = []
    names = [oneperson_path, twoperson_path]
    for name in names:
        with open(name, 'r') as f:
            data = json.load(f)
            for dialog in data:
                if name == oneperson_path and dialog['conversation_id'] not in ids:
                    continue
                if len(dialog['utterances']) < 2:
                    continue
                if dialog['utterances'][0]['speaker'] == 'ASSISTANT':
                    dialog['utterances'].insert(0, {'index': -1, 'speaker': 'USER', 'text': 'START.'})
                sens = [clean(turn['text']) for turn in dialog['utterances']]
                pop_ids, offset = [], 0
                for sen_id, sen in enumerate(sens):
                    if len(sen) == 0:
                        pop_ids.append(sen_id)
                for sen_id in pop_ids:
                    sens.pop(sen_id - offset)
                    dialog['utterances'].pop(sen_id - offset)
                    offset += 1

                speakers = [turn['speaker'] for turn in dialog['utterances']]
                segments = [turn['segments'] if 'segments' in turn else [] for turn in dialog['utterances']]
                # merge continuous turn
                prev_spe = 'ASSISTANT'
                pop_ids, count = [], 1
                for spe_id, spe in enumerate(speakers):
                    # assert count != 3
                    if spe == prev_spe:
                        pop_ids.append(spe_id)
                        sens[spe_id - count] = sens[spe_id - count] + ' ' + sens[spe_id]
                        if segments[spe_id] != []:
                            segments[spe_id - count].extend(segments[spe_id])
                        count += 1
                    else:
                        count = 1
                    prev_spe = spe

                offset = 0
                for pop_id in pop_ids:
                    sens.pop(pop_id - offset)
                    speakers.pop(pop_id - offset)
                    segments.pop(pop_id - offset)
                    offset += 1

                if len(sens) % 2 != 0:
                    sens = sens[:-1]
                    speakers = speakers[:-1]
                    segments = segments[:-1]

                # assert the format of 'user agent user agent user agent...'
                flag = False
                for spe in speakers:
                    if flag:
                        assert spe == 'ASSISTANT'
                        flag = False
                    else:
                        assert spe == 'USER'
                        flag = True

                gold_entity, gold_type = [], []
                for segment in segments:
                    gold_entity.append([])
                    gold_type.append([])
                    if segment != []:
                        for frame in segment:
                            slot_value = frame['text']
                            slot_type = '@{}@'.format(frame['annotations'][0]['name'].replace(
                                '.accept', '').replace('.reject', '').strip()).lower()
                            slot_types.add(slot_type)
                            gold_entity[-1].append(clean(slot_value))
                            gold_type[-1].append(slot_type)
                assert len(sens) == len(gold_entity)

                for i in range(0, len(sens)):
                    if speakers[i] == "USER":
                        sens[i] = '{} {} {}\t'.format(USR, sens[i], EOS)
                    else:
                        for entity_id, entity in enumerate(gold_entity[i]):
                            if entity not in sens[i]:
                                gold_entity[i].pop(entity_id)
                                gold_type[i].pop(entity_id)
                            else:
                                sens[i] = sens[i].replace(entity, trans(entity))
                                gold_entity[i][entity_id] = trans(entity)
                        sens[i] = '{} {} {}\t{}\t{}\n'.format(SYS, sens[i], EOS, gold_entity[i], gold_type[i])
                sens = divide_dialogue(sens)
                for dialog in sens:
                    result.append('#Taskmaster-1#\n' + ''.join(dialog) + '\n')
    logger.info('Number of dialogues in the Training File:\t%s' % len(result))
    total.append(len(result))
    length.extend([sum([len(turn.split('\t')[0].split(' ')) + len(turn.split('\t')[1].split(' ')) for turn in i.split('\n')[1:-2]]) for i in result])
    with open(output_file, 'a') as f:
        f.writelines(result)


def get_dev_data_Taskmaster(oneperson_path, output_file, ids):
    result = []
    names = [oneperson_path]
    for name in names:
        with open(name, 'r') as f:
            data = json.load(f)
            for dialog in data:
                if name == oneperson_path and dialog['conversation_id'] not in ids:
                    continue
                if len(dialog['utterances']) < 2:
                    continue
                if dialog['utterances'][0]['speaker'] == 'ASSISTANT':
                    dialog['utterances'].insert(0, {'index': -1, 'speaker': 'USER', 'text': 'START.'})
                sens = [clean(turn['text']) for turn in dialog['utterances']]
                pop_ids, offset = [], 0
                for sen_id, sen in enumerate(sens):
                    if len(sen) == 0:
                        pop_ids.append(sen_id)
                for sen_id in pop_ids:
                    sens.pop(sen_id - offset)
                    dialog['utterances'].pop(sen_id - offset)
                    offset += 1

                speakers = [turn['speaker'] for turn in dialog['utterances']]
                segments = [turn['segments'] if 'segments' in turn else [] for turn in dialog['utterances']]
                # merge continuous turn
                prev_spe = 'ASSISTANT'
                pop_ids, count = [], 1
                for spe_id, spe in enumerate(speakers):
                    # assert count != 3
                    if spe == prev_spe:
                        pop_ids.append(spe_id)
                        sens[spe_id - count] = sens[spe_id - count] + ' ' + sens[spe_id]
                        if segments[spe_id] != []:
                            segments[spe_id - count].extend(segments[spe_id])
                        count += 1
                    else:
                        count = 1
                    prev_spe = spe

                offset = 0
                for pop_id in pop_ids:
                    sens.pop(pop_id - offset)
                    speakers.pop(pop_id - offset)
                    segments.pop(pop_id - offset)
                    offset += 1

                if len(sens) % 2 != 0:
                    sens = sens[:-1]
                    speakers = speakers[:-1]
                    segments = segments[:-1]

                # assert the format of 'user agent user agent user agent...'
                flag = False
                for spe in speakers:
                    if flag:
                        assert spe == 'ASSISTANT'
                        flag = False
                    else:
                        assert spe == 'USER'
                        flag = True

                gold_entity, gold_type = [], []
                for segment in segments:
                    gold_entity.append([])
                    gold_type.append([])
                    if segment != []:
                        for frame in segment:
                            slot_value = frame['text']
                            slot_type = '@{}@'.format(frame['annotations'][0]['name'].replace(
                                '.accept', '').replace('.reject', '').strip()).lower()
                            slot_types.add(slot_type)
                            gold_entity[-1].append(clean(slot_value))
                            gold_type[-1].append(slot_type)
                assert len(sens) == len(gold_entity)

                for i in range(0, len(sens)):
                    if speakers[i] == "USER":
                        sens[i] = '{} {} {}\t'.format(USR, sens[i], EOS)
                    else:
                        for entity_id, entity in enumerate(gold_entity[i]):
                            if entity not in sens[i]:
                                gold_entity[i].pop(entity_id)
                                gold_type[i].pop(entity_id)
                            else:
                                sens[i] = sens[i].replace(entity, trans(entity))
                                gold_entity[i][entity_id] = trans(entity)
                        sens[i] = '{} {} {}\t{}\t{}\n'.format(SYS, sens[i], EOS, gold_entity[i], gold_type[i])
                sens = divide_dialogue(sens)
                for dialog in sens:
                    result.append('#Taskmaster-1#\n' + ''.join(dialog) + '\n')
    logger.info('Number of dialogues in the Dev File:\t%s' % len(result))
    total_dev.append(len(result))
    length_dev.extend([sum([len(turn.split('\t')[0].split(' ')) + len(turn.split('\t')[1].split(' ')) for turn in i.split('\n')[1:-2]]) for i in result])
    with open(output_file, 'a') as f:
        f.writelines(result)


################## Taskmaster-2 #################################

def get_data_Taskmaster2(file_path, output_file):
    result = []
    for name in os.listdir(file_path):
        fullname = os.path.join(file_path, name)
        with open(fullname, 'r') as f:
            data = json.load(f)
            for dialog in data:
                if len(dialog['utterances']) < 2:
                    continue
                if dialog['utterances'][0]['speaker'] == 'ASSISTANT':
                    dialog['utterances'].insert(0, {'index': -1, 'speaker': 'USER', 'text': 'START.'})
                sens = [clean(turn['text']) for turn in dialog['utterances']]
                pop_ids, offset = [], 0
                for sen_id, sen in enumerate(sens):
                    if len(sen) == 0:
                        pop_ids.append(sen_id)
                for sen_id in pop_ids:
                    sens.pop(sen_id - offset)
                    dialog['utterances'].pop(sen_id - offset)
                    offset += 1

                speakers = [turn['speaker'] for turn in dialog['utterances']]
                segments = [turn['segments'] if 'segments' in turn else [] for turn in dialog['utterances']]
                # merge continuous turn
                prev_spe = 'ASSISTANT'
                pop_ids, count = [], 1
                for spe_id, spe in enumerate(speakers):
                    # assert count != 3
                    if spe == prev_spe:
                        pop_ids.append(spe_id)
                        sens[spe_id - count] = sens[spe_id - count] + ' ' + sens[spe_id]
                        if segments[spe_id] != []:
                            segments[spe_id - count].extend(segments[spe_id])
                        count += 1
                    else:
                        count = 1
                    prev_spe = spe

                offset = 0
                for pop_id in pop_ids:
                    sens.pop(pop_id - offset)
                    speakers.pop(pop_id - offset)
                    segments.pop(pop_id - offset)
                    offset += 1

                if len(sens) % 2 != 0:
                    sens = sens[:-1]
                    speakers = speakers[:-1]
                    segments = segments[:-1]

                # assert the format of 'user agent user agent user agent...'
                flag = False
                for spe in speakers:
                    if flag:
                        assert spe == 'ASSISTANT'
                        flag = False
                    else:
                        assert spe == 'USER'
                        flag = True

                gold_entity, gold_type = [], []
                for segment in segments:
                    gold_entity.append([])
                    gold_type.append([])
                    if segment != []:
                        for frame in segment:
                            slot_value = frame['text']
                            slot_type = '@{}@'.format(frame['annotations'][0]['name'].strip()).lower()
                            if slot_type == '@hotel_search. num.beds@':
                                slot_type = '@hotel_search.num.beds@'
                            slot_types.add(slot_type)
                            gold_entity[-1].append(clean(slot_value))
                            gold_type[-1].append(slot_type)
                assert len(sens) == len(gold_entity)

                for i in range(0, len(sens)):
                    if speakers[i] == "USER":
                        sens[i] = '{} {} {}\t'.format(USR, sens[i], EOS)
                    else:
                        for entity_id, entity in enumerate(gold_entity[i]):
                            if entity not in sens[i]:
                                gold_entity[i].pop(entity_id)
                                gold_type[i].pop(entity_id)
                            else:
                                sens[i] = sens[i].replace(entity, trans(entity))
                                gold_entity[i][entity_id] = trans(entity)
                        sens[i] = '{} {} {}\t{}\t{}\n'.format(SYS, sens[i], EOS, gold_entity[i], gold_type[i])
                sens = divide_dialogue(sens)
                for dialog in sens:
                    result.append('#Taskmaster-2#\n' + ''.join(dialog) + '\n')
    logger.info('Number of dialogues in the Training File:\t%s' % len(result))
    total.append(len(result))
    length.extend([sum([len(turn.split('\t')[0].split(' ')) + len(turn.split('\t')[1].split(' ')) for turn in i.split('\n')[1:-2]]) for i in result])
    with open(output_file, 'a') as f:
        f.writelines(result)


################## Taskmaster-3 #################################

def get_data_Taskmaster3(file_path, output_file):
    result = []
    for name in os.listdir(file_path):
        fullname = os.path.join(file_path, name)
        with open(fullname, 'r') as f:
            data = json.load(f)
            for dialog in data:
                if len(dialog['utterances']) < 2:
                    continue
                if dialog['utterances'][0]['speaker'] == 'assistant':
                    dialog['utterances'].insert(0, {'index': -1, 'speaker': 'user', 'text': 'START.'})
                sens = [clean(turn['text']) for turn in dialog['utterances']]
                pop_ids, offset = [], 0
                for sen_id, sen in enumerate(sens):
                    if len(sen) == 0:
                        pop_ids.append(sen_id)
                for sen_id in pop_ids:
                    sens.pop(sen_id - offset)
                    dialog['utterances'].pop(sen_id - offset)
                    offset += 1

                speakers = [turn['speaker'] for turn in dialog['utterances']]
                segments = [turn['segments'] if 'segments' in turn else [] for turn in dialog['utterances']]
                # merge continuous turn
                prev_spe = 'assistant'
                pop_ids, count = [], 1
                for spe_id, spe in enumerate(speakers):
                    # assert count != 3
                    if spe == prev_spe:
                        pop_ids.append(spe_id)
                        sens[spe_id - count] = sens[spe_id - count] + ' ' + sens[spe_id]
                        if segments[spe_id] != []:
                            segments[spe_id - count].extend(segments[spe_id])
                        count += 1
                    else:
                        count = 1
                    prev_spe = spe

                offset = 0
                for pop_id in pop_ids:
                    sens.pop(pop_id - offset)
                    speakers.pop(pop_id - offset)
                    segments.pop(pop_id - offset)
                    offset += 1

                if len(sens) % 2 != 0:
                    sens = sens[:-1]
                    speakers = speakers[:-1]
                    segments = segments[:-1]
                # assert the format of 'user agent user agent user agent...'
                flag = False
                for spe in speakers:
                    if flag:
                        assert spe == 'assistant'
                        flag = False
                    else:
                        assert spe == 'user'
                        flag = True

                gold_entity, gold_type = [], []
                for segment in segments:
                    gold_entity.append([])
                    gold_type.append([])
                    if segment != []:
                        for frame in segment:
                            slot_value = frame['text']
                            slot_type = '@{}@'.format(frame['annotations'][0]['name'].strip()).lower()
                            slot_types.add(slot_type)
                            gold_entity[-1].append(clean(slot_value))
                            gold_type[-1].append(slot_type)
                assert len(sens) == len(gold_entity)

                for i in range(0, len(sens)):
                    if speakers[i] == "user":
                        sens[i] = '{} {} {}\t'.format(USR, sens[i], EOS)
                    else:
                        for entity_id, entity in enumerate(gold_entity[i]):
                            if entity not in sens[i]:
                                gold_entity[i].pop(entity_id)
                                gold_type[i].pop(entity_id)
                            else:
                                sens[i] = sens[i].replace(entity, trans(entity))
                                gold_entity[i][entity_id] = trans(entity)
                        sens[i] = '{} {} {}\t{}\t{}\n'.format(SYS, sens[i], EOS, gold_entity[i], gold_type[i])
                sens = divide_dialogue(sens)
                for dialog in sens:
                    result.append('#Taskmaster-3#\n' + ''.join(dialog) + '\n')
    logger.info('Number of dialogues in the Training File:\t%s' % len(result))
    total.append(len(result))
    length.extend([sum([len(turn.split('\t')[0].split(' ')) + len(turn.split('\t')[1].split(' ')) for turn in i.split('\n')[1:-2]]) for i in result])
    with open(output_file, 'a') as f:
        f.writelines(result)


################## MWOZ #################################

def get_id_list_MWOZ(valList_path, testList_path):
    names = [valList_path, testList_path]
    result = []
    for name in names:
        with open(name, 'r') as f:
            lines = f.readlines()
            lines = [line.strip() for line in lines]
            result.append(lines)
    return result


def get_data_MWOZ(data_path, output_file, ids):
    result = []
    with open(data_path, 'r') as f:
        data = json.load(f)
        for dialog in data:
            if dialog in ids[0] or dialog in ids[1]:
                continue
            turns = data[dialog]['log']
            sens = [clean(turn['text']) for turn in turns]

            pop_ids, offset = [], 0
            for sen_id, sen in enumerate(sens):
                if len(sen) == 0:
                    pop_ids.append(sen_id)
            for sen_id in pop_ids:
                sens.pop(sen_id - offset)
                turns.pop(sen_id - offset)
                offset += 1

            if len(sens) < 2:
                continue
            assert len(sens) % 2 == 0

            gold_entity, gold_type = [], []
            for turn_id, turn in enumerate(turns):
                gold_entity.append([])
                gold_type.append([])
                if 'span_info' in turn:
                    for slot in turn['span_info']:
                        slot_type = '@{}@'.format(slot[1].strip()).lower()
                        if slot_type in types_replace_woz:
                            slot_type = types_replace_woz[slot_type]
                        slot_types.add(slot_type)
                        slot_value = clean(slot[2])
                        if slot_value not in gold_entity[-1]:
                            gold_entity[-1].append(slot_value)
                            gold_type[-1].append(slot_type)
            assert len(sens) == len(gold_entity)

            flag = False
            if len(sens) < 2:
                continue
            for i in range(0, len(sens)):
                if flag:
                    for entity_id, entity in enumerate(gold_entity[i]):
                        if entity not in sens[i]:
                            gold_entity[i].pop(entity_id)
                            gold_type[i].pop(entity_id)
                        else:
                            sens[i] = sens[i].replace(entity, trans(entity))
                            gold_entity[i][entity_id] = trans(entity)
                    sens[i] = '{} {} {}\t{}\t{}\n'.format(SYS, sens[i], EOS, gold_entity[i], gold_type[i])
                    flag = False
                else:
                    sens[i] = '{} {} {}\t'.format(USR, sens[i], EOS)
                    flag = True
            sens = divide_dialogue(sens)
            for dialog in sens:
                result.append('#MultiWOZ_2.1#\n' + ''.join(dialog) + '\n')
    logger.info('Number of dialogues in the Training File:\t%s' % len(result))
    total.append(len(result))
    length.extend([sum([len(turn.split('\t')[0].split(' ')) + len(turn.split('\t')[1].split(' ')) for turn in i.split('\n')[1:-2]]) for i in result])
    with open(output_file, 'a') as f:
        f.writelines(result)


def get_dev_data_MWOZ(data_path, output_file, ids):
    result = []
    with open(data_path, 'r') as f:
        data = json.load(f)
        for dialog in data:
            if dialog in ids:
                turns = data[dialog]['log']
                sens = [clean(turn['text']) for turn in turns]
                pop_ids, offset = [], 0
                for sen_id, sen in enumerate(sens):
                    if len(sen) == 0:
                        pop_ids.append(sen_id)

                for sen_id in pop_ids:
                    sens.pop(sen_id - offset)
                    turns.pop(sen_id - offset)
                    offset += 1

                if len(sens) < 2:
                    continue
                assert len(sens) % 2 == 0

                gold_entity, gold_type = [], []
                for turn_id, turn in enumerate(turns):
                    gold_entity.append([])
                    gold_type.append([])
                    if 'span_info' in turn:
                        for slot in turn['span_info']:
                            slot_type = '@{}@'.format(slot[1].strip()).lower()
                            if slot_type in types_replace_woz:
                                slot_type = types_replace_woz[slot_type]
                            slot_types.add(slot_type)
                            slot_value = clean(slot[2])
                            if slot_value not in gold_entity[-1]:
                                gold_entity[-1].append(slot_value)
                                gold_type[-1].append(slot_type)
                assert len(sens) == len(gold_entity)

                flag = False
                if len(sens) < 2:
                    continue
                for i in range(0, len(sens)):
                    if flag:
                        for entity_id, entity in enumerate(gold_entity[i]):
                            if entity not in sens[i]:
                                gold_entity[i].pop(entity_id)
                                gold_type[i].pop(entity_id)
                            else:
                                sens[i] = sens[i].replace(entity, trans(entity))
                                gold_entity[i][entity_id] = trans(entity)
                        sens[i] = '{} {} {}\t{}\t{}\n'.format(SYS, sens[i], EOS, gold_entity[i], gold_type[i])
                        flag = False
                    else:
                        sens[i] = '{} {} {}\t'.format(USR, sens[i], EOS)
                        flag = True
                sens = divide_dialogue(sens)
                for dialog in sens:
                    result.append('#MultiWOZ_2.1#\n' + ''.join(dialog) + '\n')
    logger.info('Number of dialogues in the Dev File:\t%s' % len(result))
    total_dev.append(len(result))
    length_dev.extend([sum([len(turn.split('\t')[0].split(' ')) + len(turn.split('\t')[1].split(' ')) for turn in i.split('\n')[1:-2]]) for i in result])
    with open(output_file, 'a') as f:
        f.writelines(result)


################## MSR_E2E #################################

def get_data_MSR_E2E(file_path, output_file):
    result = []
    for name in os.listdir(file_path):
        fullname = os.path.join(file_path, name)
        with open(fullname, 'r') as f:
            lines = f.readlines()
            lines.pop(0)
            IDs = [int(line.split('\t')[0]) for line in lines]
            speakers = np.array([line.split('\t')[3] for line in lines])
            sentences = np.array([clean(line.split('\t')[4]) for line in lines])
            annotations = np.array([line.split('\t')[5] for line in lines])  # only use the first annotation
            i, line = 1, 0
            ids = [[]]
            for ID in IDs:
                if ID == i:
                    ids[-1].append(line)
                else:
                    i = ID
                    ids.append([line])
                line += 1
            for id in ids:
                if len(id) < 2:
                    continue
                spes = list(speakers[id])
                sens = list(sentences[id])
                ans = list(annotations[id])

                pop_ids, offset = [], 0
                for sen_id, sen in enumerate(sens):
                    if len(sen) == 0:
                        pop_ids.append(sen_id)
                for sen_id in pop_ids:
                    sens.pop(sen_id - offset)
                    spes.pop(sen_id - offset)
                    ans.pop(sen_id - offset)
                    offset += 1

                if len(sens) < 2:
                    continue

                assert spes[0] == 'user'
                if len(sens) % 2 != 0:
                    sens = sens[:-1]
                    spes = spes[:-1]
                    ans = ans[:-1]
                # assert the format of 'user agent user agent user agent...'
                flag, drop = False, False
                for spe in spes:
                    if flag:
                        if spe != 'agent':
                            drop = True
                            break
                        flag = False
                    else:
                        if spe != 'user':
                            drop = True
                            break
                        flag = True
                if drop:
                    continue

                gold_entity, gold_type = [], []
                for annotation_id, annotation in enumerate(ans):
                    gold_entity.append([])
                    gold_type.append([])
                    annotation = annotation[annotation.find('(') + 1:annotation.find(')')]
                    if annotation != '':
                        annotation = annotation.split(';')
                        for frame in annotation:
                            if '=' in frame:
                                slots = frame.split('=')
                                slot_type = '@{}@'.format(slots[0].strip()).lower()
                                if slot_type in types_bad_MSR_E2E:
                                    continue
                                slot_types.add(slot_type)
                                slot_value = slots[1].strip()
                                if not slot_value:
                                    continue
                                if '#' in slot_value:
                                    slot_values = slot_value[1:-1].split('#')
                                else:
                                    slot_values = [slot_value]
                                for slot_value in slot_values:
                                    slot_value = clean(slot_value)
                                    if slot_value not in gold_entity[-1]:
                                        gold_entity[-1].append(clean(slot_value))
                                        gold_type[-1].append(slot_type)
                assert len(sens) == len(gold_entity)

                for i in range(0, len(sens)):
                    if spes[i] == "user":
                        sens[i] = '{} {} {}\t'.format(USR, sens[i], EOS)
                    else:
                        for entity_id, entity in enumerate(gold_entity[i]):
                            if entity not in sens[i]:
                                gold_entity[i].pop(entity_id)
                                gold_type[i].pop(entity_id)
                            else:
                                sens[i] = sens[i].replace(entity, trans(entity))
                                gold_entity[i][entity_id] = trans(entity)
                        sens[i] = '{} {} {}\t{}\t{}\n'.format(SYS, sens[i], EOS, gold_entity[i], gold_type[i])
                sens = divide_dialogue(sens)
                for dialog in sens:
                    result.append('#MSR_E2E#\n' + ''.join(dialog) + '\n')
    logger.info('Number of dialogues in the Training File:\t%s' % len(result))
    total.append(len(result))
    length.extend([sum([len(turn.split('\t')[0].split(' ')) + len(turn.split('\t')[1].split(' ')) for turn in i.split('\n')[1:-2]]) for i in result])
    with open(output_file, 'a') as f:
        f.writelines(result)


################## SMD #################################

def get_data_SMD_txt(file_path, output_file, entity_path):
    with open(entity_path, 'r') as f:
        global_entity = json.load(f)
        global_entity['address'] = set()
        global_entity['poi_temp'] = set()
        for poi in global_entity['poi']:
            global_entity['address'].add(poi['address'])
            global_entity['poi_temp'].add(poi['poi'])
        global_entity['address'] = list(global_entity['address'])
        global_entity['poi'] = list(global_entity['poi_temp'])
        global_entity.pop('poi_temp')
    with open(file_path, 'r') as f:
        count = 0
        lines = f.readlines()
        result = []
        for line in lines:
            line = line.strip()
            if line:
                if line[-1] == line[0] == '#':
                    result.append('#SMD#\n')
                    count += 1
                elif line[0] == '0':
                    continue
                else:
                    nid, line = line.split(' ', 1)
                    u, r, gold_ent = line.split('\t')
                    gold_ent = list(ast.literal_eval(gold_ent))
                    gold_type = []
                    for entity in gold_ent:
                        ent_type = None
                        for key in global_entity.keys():
                            global_entity[key] = [x.lower() for x in global_entity[key]]
                            if entity in global_entity[key] or entity.replace('_', ' ') in global_entity[key]:
                                ent_type = key
                                break
                        assert ent_type != None
                        slot_types.add('@{}@'.format(ent_type))
                        gold_type.append('@{}@'.format(ent_type))
                    result.append('{} {} {}\t{} {} {}\t{}\t{}\n'.format(USR, u, EOS, SYS, r, EOS, gold_ent, gold_type))
            else:
                result.append('\n')
    logger.info('Number of dialogues in the Training File:\t%s' % count)
    total.append(count)
    length.extend([sum([len(turn.split('\t')[0].split(' ')) + len(turn.split('\t')[1].split(' ')) for turn in i.split('\n')[1:-2]]) for i in result])
    with open(output_file, 'a') as f:
        f.writelines(result)


def get_dev_data_SMD_txt(file_path, output_file, entity_path):
    with open(entity_path, 'r') as f:
        global_entity = json.load(f)
        global_entity['address'] = set()
        global_entity['poi_temp'] = set()
        for poi in global_entity['poi']:
            global_entity['address'].add(poi['address'])
            global_entity['poi_temp'].add(poi['poi'])
        global_entity['address'] = list(global_entity['address'])
        global_entity['poi'] = list(global_entity['poi_temp'])
        global_entity.pop('poi_temp')
    with open(file_path, 'r') as f:
        count = 0
        lines = f.readlines()
        result = []
        for line in lines:
            line = line.strip()
            if line:
                if line[-1] == line[0] == '#':
                    result.append('#SMD#\n')
                    count += 1
                elif line[0] == '0':
                    continue
                else:
                    nid, line = line.split(' ', 1)
                    u, r, gold_ent = line.split('\t')
                    gold_ent = list(ast.literal_eval(gold_ent))
                    gold_type = []
                    for entity in gold_ent:
                        ent_type = None
                        for key in global_entity.keys():
                            global_entity[key] = [x.lower() for x in global_entity[key]]
                            if entity in global_entity[key] or entity.replace('_', ' ') in global_entity[key]:
                                ent_type = key
                                break
                        assert ent_type != None
                        slot_types.add('@{}@'.format(ent_type))
                        gold_type.append('@{}@'.format(ent_type))
                    result.append('{} {} {}\t{} {} {}\t{}\t{}\n'.format(USR, u, EOS, SYS, r, EOS, gold_ent, gold_type))
            else:
                result.append('\n')

    logger.info('Number of dialogues in the Dev File:\t%s' % count)
    total_dev.append(count)
    length_dev.extend([sum([len(turn.split('\t')[0].split(' ')) + len(turn.split('\t')[1].split(' ')) for turn in i.split('\n')[1:-2]]) for i in result])
    with open(output_file, 'a') as f:
        f.writelines(result)


################## WOZ #################################

def get_data_WOZ(file_path, output_file):
    result = []
    with open(file_path, 'r') as f:
        data = json.load(f)
        for dialog in data:
            sens, speakers, labels = [], [], []
            for turn in dialog['dialogue']:
                if turn['system_transcript'] != '':
                    sens.append(clean(turn['system_transcript']))
                    speakers.append('system')
                    labels.append(turn['turn_label'])
                if turn['transcript'] != '':
                    sens.append(clean(turn['transcript']))
                    speakers.append('user')
                    labels.append([])

            pop_ids, offset = [], 0
            for sen_id, sen in enumerate(sens):
                if len(sen) == 0:
                    pop_ids.append(sen_id)
            for sen_id in pop_ids:
                sens.pop(sen_id - offset)
                speakers.pop(sen_id - offset)
                labels.pop(sen_id - offset)
                offset += 1

            if len(sens) < 2:
                continue

            if len(sens) % 2 != 0:
                sens = sens[:-1]
                speakers = speakers[:-1]
                labels = labels[:-1]
            # assert the format of 'user agent user agent user agent...'
            flag = False
            for spe in speakers:
                if flag:
                    assert spe == 'system'
                    flag = False
                else:
                    assert spe == 'user'
                    flag = True

            gold_entity, gold_type = [], []
            for label in labels:
                gold_entity.append([])
                gold_type.append([])
                for slot in label:
                    if slot[0] != 'request':
                        slot_value = slot[1]
                        slot_type = '@{}@'.format(slot[0].strip()).lower()
                        if slot_type == '@price range@':
                            slot_type = '@pricerange@'
                        slot_types.add(slot_type)
                        gold_entity[-1].append(clean(slot_value))
                        gold_type[-1].append(slot_type)
            assert len(sens) == len(gold_entity)

            for i in range(0, len(sens)):
                if speakers[i] == "user":
                    sens[i] = '{} {} {}\t'.format(USR, sens[i], EOS)
                else:
                    for entity_id, entity in enumerate(gold_entity[i]):
                        if entity not in sens[i]:
                            gold_entity[i].pop(entity_id)
                            gold_type[i].pop(entity_id)
                        else:
                            sens[i] = sens[i].replace(entity, trans(entity))
                            gold_entity[i][entity_id] = trans(entity)
                    sens[i] = '{} {} {}\t{}\t{}\n'.format(SYS, sens[i], EOS, gold_entity[i], gold_type[i])
            sens = divide_dialogue(sens)
            for dialog in sens:
                result.append('#WOZ#\n' + ''.join(dialog) + '\n')
    logger.info('Number of dialogues in the Training File:\t%s' % len(result))
    total.append(len(result))
    length.extend([sum([len(turn.split('\t')[0].split(' ')) + len(turn.split('\t')[1].split(' ')) for turn in i.split('\n')[1:-2]]) for i in result])
    with open(output_file, 'a') as f:
        f.writelines(result)


def get_dev_data_WOZ(file_path, output_file):
    result = []
    with open(file_path, 'r') as f:
        data = json.load(f)
        for dialog in data:
            sens, speakers, labels = [], [], []
            for turn in dialog['dialogue']:
                if turn['system_transcript'] != '':
                    sens.append(clean(turn['system_transcript']))
                    speakers.append('system')
                    labels.append([])
                if turn['transcript'] != '':
                    sens.append(clean(turn['transcript']))
                    speakers.append('user')
                    labels.append(turn['turn_label'])

            pop_ids, offset = [], 0
            for sen_id, sen in enumerate(sens):
                if len(sen) == 0:
                    pop_ids.append(sen_id)
            for sen_id in pop_ids:
                sens.pop(sen_id - offset)
                speakers.pop(sen_id - offset)
                labels.pop(sen_id - offset)
                offset += 1

            if len(sens) < 2:
                continue
            if len(sens) % 2 != 0:
                sens = sens[:-1]
                speakers = speakers[:-1]
                labels = labels[:-1]
            # assert the format of 'user agent user agent user agent...'
            flag, drop = False, False
            for spe in speakers:
                if flag:
                    if spe != 'system':
                        drop = True
                        break
                    flag = False
                else:
                    if spe != 'user':
                        drop = True
                        break
                    flag = True
            if drop:
                continue

            gold_entity, gold_type = [], []
            for label in labels:
                gold_entity.append([])
                gold_type.append([])
                for slot in label:
                    if slot[0] != 'request':
                        slot_value = slot[1]
                        slot_type = '@{}@'.format(slot[0].strip()).lower()
                        if slot_type == '@price range@':
                            slot_type = '@pricerange@'
                        slot_types.add(slot_type)
                        gold_entity[-1].append(clean(slot_value))
                        gold_type[-1].append(slot_type)
            assert len(sens) == len(gold_entity)

            for i in range(0, len(sens)):
                if speakers[i] == "user":
                    sens[i] = '{} {} {}\t'.format(USR, sens[i], EOS)
                else:
                    for entity_id, entity in enumerate(gold_entity[i]):
                        if entity not in sens[i]:
                            gold_entity[i].pop(entity_id)
                            gold_type[i].pop(entity_id)
                        else:
                            sens[i] = sens[i].replace(entity, trans(entity))
                            gold_entity[i][entity_id] = trans(entity)
                    sens[i] = '{} {} {}\t{}\t{}\n'.format(SYS, sens[i], EOS, gold_entity[i], gold_type[i])
            sens = divide_dialogue(sens)
            for dialog in sens:
                result.append('#WOZ#\n' + ''.join(dialog) + '\n')
    logger.info('Number of dialogues in the Dev File:\t%s' % len(result))
    total_dev.append(len(result))
    length_dev.extend([sum([len(turn.split('\t')[0].split(' ')) + len(turn.split('\t')[1].split(' ')) for turn in i.split('\n')[1:-2]]) for i in result])
    with open(output_file, 'a') as f:
        f.writelines(result)


################## CameRest676 #################################

def get_data_CameRest676_txt(file_path, output_file, entity_path):
    with open(entity_path, 'r') as f:
        global_entity = json.load(f)

    with open(file_path, 'r') as f:
        count = 0
        lines = f.readlines()
        result = []
        for line in lines:
            line = line.strip()
            if line:
                if line[-1] == line[0] == '#':
                    result.append('#CameRest676#\n')
                    count += 1
                elif line[0] == '0':
                    continue
                else:
                    nid, line = line.split(' ', 1)
                    u, r, gold_ent = line.split('\t')
                    gold_ent = list(ast.literal_eval(gold_ent))
                    gold_type = []
                    for entity in gold_ent:
                        ent_type = None
                        for key in global_entity.keys():
                            global_entity[key] = [x.lower() for x in global_entity[key]]
                            if entity in global_entity[key] or entity.replace('_', ' ') in global_entity[key]:
                                ent_type = key
                                break
                        assert ent_type != None
                        slot_types.add('@{}@'.format(ent_type))
                        gold_type.append('@{}@'.format(ent_type))
                    result.append('{} {} {}\t{} {} {}\t{}\t{}\n'.format(USR, u, EOS, SYS, r, EOS, gold_ent, gold_type))
            else:
                result.append('\n')
    logger.info('Number of dialogues in the Training File:\t%s' % count)
    total.append(count)
    length.extend([sum([len(turn.split('\t')[0].split(' ')) + len(turn.split('\t')[1].split(' ')) for turn in i.split('\n')[1:-2]]) for i in result])
    with open(output_file, 'a') as f:
        f.writelines(result)


def get_dev_data_CameRest676_txt(file_path, output_file, entity_path):
    with open(entity_path, 'r') as f:
        global_entity = json.load(f)

    with open(file_path, 'r') as f:
        count = 0
        lines = f.readlines()
        result = []
        for line in lines:
            line = line.strip()
            if line:
                if line[-1] == line[0] == '#':
                    result.append('#CameRest676#\n')
                    count += 1
                elif line[0] == '0':
                    continue
                else:
                    nid, line = line.split(' ', 1)
                    u, r, gold_ent = line.split('\t')
                    gold_ent = list(ast.literal_eval(gold_ent))
                    gold_type = []
                    for entity in gold_ent:
                        ent_type = None
                        for key in global_entity.keys():
                            global_entity[key] = [x.lower() for x in global_entity[key]]
                            if entity in global_entity[key] or entity.replace('_', ' ') in global_entity[key]:
                                ent_type = key
                                break
                        assert ent_type != None
                        gold_type.append('@{}@'.format(ent_type))
                    result.append('{} {} {}\t{} {} {}\t{}\t{}\n'.format(USR, u, EOS, SYS, r, EOS, gold_ent, gold_type))
            else:
                result.append('\n')

    logger.info('Number of dialogues in the Dev File:\t%s' % count)
    total_dev.append(count)
    length_dev.extend([sum([len(turn.split('\t')[0].split(' ')) + len(turn.split('\t')[1].split(' ')) for turn in i.split('\n')[1:-2]]) for i in result])
    with open(output_file, 'a') as f:
        f.writelines(result)


############## Run #############

train_path = './data/original/dstc8-schema-guided-dialogue-master/train'
dev_path = './data/original/dstc8-schema-guided-dialogue-master/dev'
logger.info('Schema')
get_data_Schema(train_path, train_file)
get_dev_data_Schema(dev_path, dev_file)

id_path = './data/original/Taskmaster-master/TM-1-2019/train-dev-test'
oneperson_path = './data/original/Taskmaster-master/TM-1-2019/self-dialogs.json'
twoperson_path = './data/original/Taskmaster-master/TM-1-2019/woz-dialogs.json'
logger.info('Taskmaster-1')
id_list = get_id_list_Taskmaster(id_path)
logger.info(' '.join([str(len(ids)) for ids in id_list]))
get_data_Taskmaster(oneperson_path, twoperson_path, train_file, id_list[0])
get_dev_data_Taskmaster(oneperson_path, dev_file, id_list[1])

data_path = './data/original/Taskmaster-master/TM-2-2020/data'
logger.info('Taskmaster-2')
get_data_Taskmaster2(data_path, train_file)

data_path = './data/original/Taskmaster-master/TM-3-2020/data'
logger.info('Taskmaster-3')
get_data_Taskmaster3(data_path, train_file)

valList_path = './data/original/MultiWOZ_2.1/valListFile.txt'
testList_path = './data/original/MultiWOZ_2.1/testListFile.txt'
data_path = './data/original/MultiWOZ_2.1/data.json'
logger.info('MWOZ')
id_list = get_id_list_MWOZ(valList_path, testList_path)
logger.info(' '.join([str(len(ids)) for ids in id_list]))
get_data_MWOZ(data_path, train_file, id_list)
get_dev_data_MWOZ(data_path, dev_file, id_list[0])

data_path = './data/original/e2e_dialog_challenge'
logger.info('MSR_E2E')
get_data_MSR_E2E(data_path, train_file)

train_path = './data/original/SMD/train.txt'
dev_path = './data/original/SMD/dev.txt'
entity_path = './data/original/SMD/kvret_entities.json'
logger.info('SMD')
get_data_SMD_txt(train_path, train_file, entity_path)
get_dev_data_SMD_txt(dev_path, dev_file, entity_path)

# annotation not enough
train_path = './data/original/neural-belief-tracker-master/./data/woz/woz_train_en.json'
dev_path = './data/original/neural-belief-tracker-master/./data/woz/woz_validate_en.json'
logger.info('WOZ')
get_data_WOZ(train_path, train_file)
get_dev_data_WOZ(dev_path, dev_file)

train_path = './data/original/CamRest676/train.txt'
dev_path = './data/original/CamRest676/dev.txt'
entity_path = './data/original/CamRest676/CamRest676_entities.json'
logger.info('CameRest676')
get_data_CameRest676_txt(train_path, train_file, entity_path)
get_dev_data_CameRest676_txt(dev_path, dev_file, entity_path)

############## Summary #############
logger.info('Total number of dialogues in the Training File:\t%s' % sum(total))
print(total)
logger.info('Total number of dialogues in the Dev File:\t%s' % sum(total_dev))
print(total_dev)

logger.info('AvgLength of dialogues in the Training File:\t%s' % np.mean(length))
plt.hist(length)
plt.savefig("./data/original/AvgLength_Train.png")
plt.show()

logger.info('AvgLength of dialogues in the Dev File:\t%s' % np.mean(length_dev))
plt.hist(length_dev)
plt.savefig("./data/original/AvgLength_Dev.png")
plt.show()

# slot types should be added to ent_types in utils/config.py, so that the model can take them as special tokens.
print(slot_types)