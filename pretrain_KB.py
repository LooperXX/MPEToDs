from utils.config import *
from models.model import *
from transformers import GPT2Tokenizer

# configure models and load data
tokenizer = GPT2Tokenizer.from_pretrained(args['pretrain_path'])
special_tokens_dict = {'additional_special_tokens': [USR, SYS] + ent_types}
tokenizer.add_special_tokens(special_tokens_dict)

train, dev, _, lang, max_resp_len = prepare_data_seq(tokenizer, batch_size=int(args['batch']),
                                                     data_augmentation_file=args['data_augmentation_file'])

model = MPEToDs(tokenizer, max_resp_len, lang=lang)
mylogger.info('Training Start')
model.pretrain_KB(train, dev)
fitlog.finish()
