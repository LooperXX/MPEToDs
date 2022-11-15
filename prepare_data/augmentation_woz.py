import json, ast, random, os

random_seed = 42
random.seed(random_seed)

entity_types = ['@address', '@area', '@choice', '@food', '@name', '@phone', '@postcode', '@pricerange', '@ref',
                '@stars', '@type']

# key type in KB | candidate types
entity_types_res = ['@address', '@area', '@food', '@name', '@phone', '@postcode', '@pricerange', '@type', '@ref', '@choice']  # @name | @address @area @phone @postcode @ref
entity_types_att = ['@address', '@area', '@name', '@phone', '@postcode', '@pricerange', '@type', '@choice']  # @name | @address @area @phone @postcode
entity_types_hot = ['@address', '@area', '@name', '@phone', '@postcode', '@pricerange', '@stars', '@type', '@ref', '@choice']  # @name | @address @area @phone @postcode @stars @ref

# candidate types: entity types that can be used for augmentation, filter by manual check
candidate_types = ['address', 'area', 'phone', 'postcode', 'ref', 'stars']
count = 0
count_ = [0, 0, 0]
repeat_time = 10
domains = {'#restaurant#': [], '#attraction#': [], '#hotel#': []}


def sort_set(input):
    output = list(set(input))
    output.sort(key=input.index)
    return output


with open('./data/fine-tune/MULTIWOZ2.1/global_entities.json') as f:
    global_entity = json.load(f)

for key in global_entity.keys():
    global_entity[key] = [x.lower().replace(' ', '_') for x in global_entity[key]]
    global_entity[key] = set(global_entity[key])
for item in global_entity.keys() - set(candidate_types):
    global_entity.pop(item)

with open('./data/fine-tune/MULTIWOZ2.1/train.txt') as fin:
    for line in fin:
        line = line.strip()
        if line:
            if line[-1] == line[0] == '#':
                line = line.replace("#", "")
                domain = line
                kb, context, gold = [], [], []
                candidates = dict.fromkeys(candidate_types)
                for key in candidates.keys():
                    candidates[key] = []
                continue

            _, line_ = line.split(' ', 1)
            if '\t' in line_:
                _, _, gold_ent = line_.split('\t')
                gold_ent = ast.literal_eval(gold_ent)
                gold.extend(gold_ent)
                context.append(line)
            else:
                kb.append(line_.split(' '))
        else:
            if gold != []:
                flag = False
                for g in gold:
                    for key in candidate_types:
                        if g in global_entity[key]:
                            candidates[key].append(g)
                            flag = True
                for key in candidate_types:
                    if len(candidates[key]) > 0:
                        candidates[key] = sort_set(candidates[key])
                if flag:
                    for i in range(repeat_time):
                        replace = dict.fromkeys(candidate_types)
                        for key in candidate_types:
                            replace[key] = []
                        for key in candidates.keys():
                            if len(candidates[key]) != 0:
                                if domain == 'restaurant':
                                    cur = set()
                                    if key == 'name':
                                        lens = 11
                                    else:
                                        lens = 3
                                    for item in kb:
                                        if len(item) == lens and item[-2] == key:
                                            cur.add(item[-1])
                                    remain = global_entity[key] - cur
                                    if len(remain) < len(candidates[key]):
                                        cur = set(candidates[key])
                                        remain = global_entity[key] - cur
                                    remain_list = list(remain)
                                    if len(remain) >= len(candidates[key]):
                                        while len(replace[key]) < len(candidates[key]):
                                            index = random.randint(0, len(remain) - 1)
                                            replace[key].append(remain_list[index])
                                            replace[key] = sort_set(replace[key])
                                    else:
                                        print('???restaurant???')
                                        flag = False
                                        break
                                elif domain == 'attraction':
                                    cur = set()
                                    if key == 'name':
                                        lens = 8
                                    else:
                                        lens = 3
                                    for item in kb:
                                        if len(item) == lens and item[-2] == key:
                                            cur.add(item[-1])
                                    remain = global_entity[key] - cur
                                    if len(remain) < len(candidates[key]):
                                        cur = set(candidates[key])
                                        remain = global_entity[key] - cur
                                    remain_list = list(remain)
                                    if len(remain) >= len(candidates[key]):
                                        while len(replace[key]) < len(candidates[key]):
                                            index = random.randint(0, len(remain) - 1)
                                            replace[key].append(remain_list[index])
                                            replace[key] = sort_set(replace[key])
                                    else:
                                        print('???attraction???')
                                        flag = False
                                        break
                                elif domain == 'hotel':
                                    cur = set()
                                    if key == 'name':
                                        lens = 10
                                    else:
                                        lens = 3
                                    for item in kb:
                                        if len(item) == lens and item[-2] == key:
                                            cur.add(item[-1])
                                    remain = global_entity[key] - cur
                                    if len(remain) < len(candidates[key]):
                                        cur = set(candidates[key])
                                        remain = global_entity[key] - cur
                                    remain_list = list(remain)
                                    if len(remain) >= len(candidates[key]):
                                        while len(replace[key]) < len(candidates[key]):
                                            index = random.randint(0, len(remain) - 1)
                                            replace[key].append(remain_list[index])
                                            replace[key] = sort_set(replace[key])
                                    else:
                                        print('???hotel???')
                                        flag = False
                                        break
                        if not flag:
                            continue

                        if domain == 'restaurant':
                            # break
                            count_[0] += 1
                        elif domain == 'attraction':
                            count_[1] += 1
                        else:
                            count_[2] += 1
                        cur_data = []
                        cur_data.append('#' + domain + '#')
                        previous, new_entity = [], []
                        for key in replace.keys():
                            if len(replace[key]) != 0:
                                for k, v in enumerate(replace[key]):
                                    previous.append(candidates[key][k])
                                    new_entity.append(v)
                        for item in kb:
                            cur = '0 ' + ' '.join(item)
                            for k, v in enumerate(previous):
                                if v in cur:
                                    cur = cur.replace(' %s ' % v, ' %s ' % new_entity[k])
                                    cur = cur.replace(' %s' % v, ' %s' % new_entity[k])
                            cur_data.append(cur)
                        for item in context:
                            temp = item.split('\t')
                            for k, v in enumerate(previous):
                                if v in item:
                                    for id in range(len(temp) - 1):
                                        if v in temp[id]:
                                            temp[id] = temp[id].replace(' %s ' % v, ' %s ' % new_entity[k])
                                            temp[id] = temp[id].replace('%s ' % v, '%s ' % new_entity[k])
                                            temp[id] = temp[id].replace(' %s' % v, ' %s' % new_entity[k])
                                    if '\'' in new_entity[k]:
                                        temp[-1] = temp[-1].replace('\'%s\'' % v, '"%s"' % new_entity[k])
                                    else:
                                        temp[-1] = temp[-1].replace('\'%s\'' % v, '\'%s\'' % new_entity[k])
                            cur_data.append('\t'.join(temp))
                        domains['#' + domain + '#'].append(cur_data)
                        count += 1

print(count, count_)

train, dev = [], []
ratios = [0.9, 0.1]
for domain in domains.keys():
    random.shuffle(domains[domain])
    nums = len(domains[domain])
    divider = round(nums * ratios[0])
    train.extend(domains[domain][:divider])
    dev.extend(domains[domain][divider:])

random.shuffle(train)
random.shuffle(dev)
os.makedirs('./data/pre-train/MULTIWOZ2.1', exist_ok=True)
with open('./data/pre-train/MULTIWOZ2.1/train_{}.txt'.format(str(repeat_time)), 'w') as fw:
    for lines in train:
        fw.write('\n'.join(lines))
        fw.write('\n\n')

with open('./data/pre-train/MULTIWOZ2.1/dev_{}.txt'.format(str(repeat_time)), 'w') as fw:
    for lines in dev:
        fw.write('\n'.join(lines))
        fw.write('\n\n')