from utils.config import *
from models.model import *
from transformers import GPT2Tokenizer

# configure models and load data
tokenizer = GPT2Tokenizer.from_pretrained(args['pretrain_path'])
special_tokens_dict = {'additional_special_tokens': [USR, SYS] + ent_types}
tokenizer.add_special_tokens(special_tokens_dict)

train, dev, test, lang, max_resp_len = prepare_data_seq(tokenizer, batch_size=int(args['batch']))

avg_best, cnt, score, global_step, epoch = 0.0, 0, 0.0, 0, 0
model = MPEToDs(tokenizer, max_resp_len, lang=lang)
model.fine_tune(train, dev)
# test
model = MPEToDs(tokenizer, max_resp_len, lang=lang)
score_test = model.evaluate(test, 1e7, Test=True)
print(model.eval_log, file=test_logger)

eval_logger.close()
test_logger.close()
fitlog.finish()
