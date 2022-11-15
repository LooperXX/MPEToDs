import torch.nn as nn
from torch.optim import lr_scheduler
from torch import optim
from utils.measures import moses_multi_bleu
from utils.masked_cross_entropy import *
from models.modules import *
from utils.config import *
from transformers import GPT2Tokenizer, GPT2Config, get_linear_schedule_with_warmup
from models.modeling_gpt2 import GPT2LMHeadModel
from models.optim import warmup_linear, noam_decay, noamwd_decay

scaler = torch.cuda.amp.GradScaler(enabled=args['fp16'])


class MPEToDs(nn.Module):
    def __init__(self, tokenizer, max_resp_len, lang=None):
        super(MPEToDs, self).__init__()
        self.name = "MPEToDs"
        self.hidden_size = args['hidden']
        self.lr = args['learn']
        self.dropout = args['drop']
        self.max_resp_len = max_resp_len
        self.lang = lang

        self.tokenizer = tokenizer
        self.classifier = nn.Linear(args['embedding_dim'], 1)
        config = GPT2Config.from_json_file(os.path.join(args['pretrain_path'], 'config.json'))
        self.GPT2_model = GPT2LMHeadModel(config)
        if lang is not None:
            self.extKnow = ExternalKnowledge(lang.n_words, self.hidden_size, args['hop'], self.dropout)

        # load model for testing
        if args['path'] is not None:
            self.GPT2_model.resize_token_embeddings(len(self.tokenizer))
            if args['gpu']:
                mylogger.info("MODEL {} LOADED".format(args['path']))
                self.extKnow.load_state_dict(torch.load(os.path.join(args['path'], 'enc_kb.th')))
                self.GPT2_model.load_state_dict(torch.load(os.path.join(args['path'], 'GPT2.th')))
            else:
                mylogger.info("MODEL {} LOADED".format(args['path']))
                self.extKnow.load_state_dict(torch.load(os.path.join(args['path'], 'enc_kb.th'),
                                                        lambda storage, loc: storage))
                self.GPT2_model.load_state_dict(torch.load(os.path.join(args['path'], 'GPT2.th'),
                                                           lambda storage, loc: storage))
        # pretrain GPT, load pretrain models
        elif args['pretrain_GPT'] is None and args['pretrain_KB'] is None and not args['fine_tune']:
            if args['pretrain_name'] == 'pytorch_model.bin':  # use the GPT2 model
                model_dict = self.GPT2_model.state_dict()  # currently with random initialization
                state_dict = torch.load(os.path.join(args['pretrain_path'], args['pretrain_name']))
                old_keys = []
                new_keys = []
                for key in state_dict.keys():
                    if "mlp" in key:  # The hugging face state dict references the feedforward network as mlp, need to replace to `feedforward` be able to reuse these weights
                        new_key = key.replace("mlp", "feedforward")
                        new_keys.append(new_key)
                        old_keys.append(key)
                for old_key, new_key in zip(old_keys, new_keys):
                    state_dict[new_key] = state_dict.pop(old_key)
                pretrained_dict = {k: v for k, v in state_dict.items() if k in model_dict}
                model_dict.update(pretrained_dict)
                self.GPT2_model.load_state_dict(model_dict, strict=False)
            elif args['pretrain_name'] == 'medium_ft.pkl':  # use the DialoGPT model
                model_state_dict = torch.load(os.path.join(args['pretrain_path'], args['pretrain_name']))
                model_state_dict['lm_head.weight'] = model_state_dict['lm_head.decoder.weight']
                model_state_dict.pop('lm_head.decoder.weight')
                self.GPT2_model.load_state_dict(model_state_dict, strict=False)
            self.GPT2_model.resize_token_embeddings(len(self.tokenizer))
            mylogger.info("pretrain MODEL {} LOADED".format(args['pretrain_name']))
        # pretrain KB, load pretrain GPT models     or       fine tune, load pretrain GPT and KB models
        else:
            self.GPT2_model.resize_token_embeddings(len(self.tokenizer))
            if args['gpu']:
                if args['pretrain_GPT'] is not None and args['pretrain_KB'] is not None:
                    mylogger.info("KB MODEL {} LOADED".format(args['pretrain_KB']))
                    self.extKnow.load_state_dict(torch.load(os.path.join(args['pretrain_KB'], 'enc_kb.th')))
                mylogger.info("GPT MODEL {} LOADED".format(args['pretrain_GPT']))
                self.GPT2_model.load_state_dict(torch.load(os.path.join(args['pretrain_GPT'], 'GPT2.th')))
            else:
                if args['pretrain_GPT'] is not None and args['pretrain_KB'] is not None:
                    mylogger.info("KB MODEL {} LOADED".format(args['pretrain_KB']))
                    self.extKnow.load_state_dict(torch.load(os.path.join(args['pretrain_KB'], 'enc_kb.th'),
                                                            lambda storage, loc: storage))
                mylogger.info("GPT MODEL {} LOADED".format(args['pretrain_GPT']))
                self.GPT2_model.load_state_dict(torch.load(os.path.join(args['pretrain_GPT'], 'GPT2.th'),
                                                           lambda storage, loc: storage))

        # Initialize optimizers and criterion
        if lang is not None:
            self.extKnow_optimizer = optim.AdamW(self.extKnow.parameters(), lr=self.lr)
            self.extKnow_scheduler = get_linear_schedule_with_warmup(
                self.extKnow_optimizer, num_warmup_steps=args['warmup_steps'], num_training_steps=args['total_steps']
            )

        if args['freeze_GPT']:  # Freeze the GPT2_model
            for param in self.GPT2_model.base_model.parameters():
                param.requires_grad = False
        else:
            param_optimizer = list(self.GPT2_model.named_parameters())
            no_decay = ['bias', 'ln']  # no decay for bias and LayerNorm (ln)
            optimizer_grouped_parameters = [
                {'params': [p for n, p in param_optimizer
                            if not any(nd in n for nd in no_decay)],
                 'weight_decay': 0.01},
                {'params': [p for n, p in param_optimizer
                            if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
            ]
            self.GPT2_model_optimizer = optim.AdamW(optimizer_grouped_parameters, args['GPT2_learn'])
            self.GPT2_model_scheduler = get_linear_schedule_with_warmup(
                self.GPT2_model_optimizer, num_warmup_steps=args['warmup_steps'], num_training_steps=args['total_steps']
            )

        self.criterion_bce = nn.BCELoss()
        self.reset()

        if args['gpu']:
            if lang is not None:
                self.extKnow.cuda()
            self.GPT2_model.cuda()

        self.step_time = time.time()
        mylogger.info('Model init success')

    def print_loss(self, step=None):
        print_loss_avg = self.loss / self.print_every
        print_loss_g = self.loss_g / self.print_every
        print_loss_v = self.loss_v / self.print_every
        print_loss_l = self.loss_l / self.print_every
        print_loss_b = self.loss_b / self.print_every
        if self.print_every % args['logging_steps'] == 0:
            fitlog.add_loss(print_loss_avg, name='train loss', step=step)
            fitlog.add_loss(print_loss_g, name='train loss_g', step=step)
            fitlog.add_loss(print_loss_v, name='train loss_v', step=step)
            fitlog.add_loss(print_loss_l, name='train loss_l', step=step)
            fitlog.add_loss(print_loss_b, name='train loss_b', step=step)
        self.step_time = time.time()
        self.print_every += 1
        return 'L:{:.2f},LG:{:.2f},LV:{:.2f},LL:{:.2f},LB:{:.2f}'.format(print_loss_avg, print_loss_g, print_loss_v,
                                                                         print_loss_l, print_loss_b)

    def print_pretrain_GPT_loss(self, step=None):
        print_loss_avg = self.loss / self.print_every
        print_loss_v = self.loss_v / self.print_every
        print_loss_b = self.loss_b / self.print_every
        if self.print_every % args['logging_steps'] == 0:
            fitlog.add_loss(print_loss_avg, name='train loss', step=step)
            fitlog.add_loss(print_loss_v, name='train loss_v', step=step)
            fitlog.add_loss(print_loss_b, name='train loss_b', step=step)
        self.print_every += 1
        return 'L:{:.2f},LV:{:.2f},LB:{:.2f}'.format(print_loss_avg, print_loss_v, print_loss_b)

    def print_pretrain_KB_loss(self, step=None):
        print_loss_avg = self.loss / self.print_every
        if args['freeze_GPT']:
            if self.print_every % args['logging_steps'] == 0:
                fitlog.add_loss(print_loss_avg, name='train loss', step=step)
            self.print_every += 1
            return 'L:{:.2f}'.format(print_loss_avg)
        else:
            print_loss_g = self.loss_g / self.print_every
            print_loss_v = self.loss_v / self.print_every
            print_loss_b = self.loss_b / self.print_every
            if self.print_every % args['logging_steps'] == 0:
                fitlog.add_loss(print_loss_avg, name='train loss', step=step)
                fitlog.add_loss(print_loss_g, name='train loss_g', step=step)
                fitlog.add_loss(print_loss_v, name='train loss_v', step=step)
                fitlog.add_loss(print_loss_b, name='train loss_b', step=step)
            self.print_every += 1
            return 'L:{:.2f},LG:{:.2f},LV:{:.2f},LB:{:.2f}'.format(print_loss_avg, print_loss_g, print_loss_v,
                                                                   print_loss_b)

    def save_model(self, score):
        args['path'] = os.path.join(args['output_dir'], str(score))
        os.makedirs(args['path'], exist_ok=True)
        torch.save(self.extKnow.state_dict(), os.path.join(args['path'], 'enc_kb.th'))
        torch.save(self.GPT2_model.state_dict(), os.path.join(args['path'], 'GPT2.th'))

    def reset(self):
        self.loss, self.print_every, self.loss_g, self.loss_v, self.loss_l, self.loss_b = 0, 1, 0, 0, 0, 0

    def _cuda(self, x):
        if args['gpu']:
            return x.cuda()
        else:
            return x

    def pretarin_GPT_step(self, data, train=True):
        with torch.cuda.amp.autocast(enabled=args['fp16']):
            # Encode and Decode
            response_lengths = [l1 - l2 for (l1, l2) in
                                zip(data['input_ids_lengths'], data['conv_GPT2_lengths'])]
            labels = self._cuda(torch.ones_like(data['input_ids']) * -1)
            for i, start in enumerate(data['conv_GPT2_lengths']):
                labels[i, start - 1:start - 1 + response_lengths[i]] = data['input_ids'][
                                                                       i, start:start + response_lengths[i]]
            hidden_states, classifier_output, loss_v = self.GPT2_model(data['input_ids'], labels=labels,
                                                                       type_loss=True)
            classifier_output = classifier_output.squeeze(-1)
            predict_output = torch.zeros_like(data['binary_labels'])
            for i, start in enumerate(data['conv_GPT2_lengths']):
                predict_output[i, :response_lengths[i]] = classifier_output[i,
                                                          start - 1:start - 1 + response_lengths[i]]
            # Loss calculation and backpropagation
            loss_b = masked_binary_cross_entropy(predict_output, data['binary_labels'], response_lengths)
            loss = loss_v + loss_b
            if train:
                self.loss += loss.item() / args['accumulation_steps']
                self.loss_v += loss_v.item() / args['accumulation_steps']
                self.loss_b += loss_b.item() / args['accumulation_steps']
                loss /= args['accumulation_steps']
                return loss
            else:
                return loss.item(), loss_v.item(), loss_b.item()

    def pretrain_GPT(self, train, dev):
        self.GPT2_model.train()
        dev_score_best, patience, dev_score, step, global_step, epoch = 1e7, 0, 0.0, 0, 0, 0
        # Train
        while True:
            # Run the train function
            mylogger.info("Epoch:{}".format(epoch))
            pbar = tqdm(enumerate(train), total=len(train))
            for i, data in pbar:
                loss = self.pretarin_GPT_step(data)
                if args['fp16']:
                    scaler.scale(loss).backward()
                else:
                    loss.backward()
                step += 1
                if step % args['accumulation_steps'] == 0:
                    global_step += 1
                    # Clip gradient norms
                    if args['fp16']:
                        scaler.unscale_(self.GPT2_model_optimizer)
                    torch.nn.utils.clip_grad_norm_(self.GPT2_model.parameters(), args['clip'])

                    # Update parameters with optimizers
                    if args['fp16']:
                        scaler.step(self.GPT2_model_optimizer)
                        scaler.update()
                    else:
                        self.GPT2_model_optimizer.step()
                    self.GPT2_model_scheduler.step()
                    # Zero gradients of both optimizers
                    self.GPT2_model_optimizer.zero_grad()

                    pbar.set_description(self.print_pretrain_GPT_loss(step=global_step))

                    # Dev
                    if global_step % args['valid_steps'] == 0:
                        mylogger.info('Dev Start')
                        self.GPT2_model.eval()
                        dev_loss, dev_loss_v, dev_loss_b = 0.0, 0.0, 0.0
                        with torch.no_grad():
                            dev_pbar = tqdm(enumerate(dev), total=len(dev))
                            for dev_i, dev_data in dev_pbar:
                                cur_dev_loss, cur_dev_loss_v, cur_dev_loss_b = self.pretarin_GPT_step(dev_data,
                                                                                                      train=False)
                                dev_loss += cur_dev_loss
                                dev_loss_v += cur_dev_loss_v
                                dev_loss_b += cur_dev_loss_b
                        dev_score = dev_loss / len(dev)
                        fitlog.add_metric({"dev": {
                            "dev loss": dev_score,
                            "dev loss_v": dev_loss_v / len(dev),
                            "dev loss_b": dev_loss_b / len(dev),
                        }}, step=global_step)
                        if dev_score < dev_score_best:
                            dev_score_best = dev_score
                            patience = 0
                            fitlog.add_best_metric({"dev": {"dev loss": dev_score}})
                            torch.save(self.GPT2_model.state_dict(), os.path.join(args['output_dir'], 'GPT2.th'))
                        else:
                            patience += 1

                        self.GPT2_model.train()

                        if patience >= args['patience']:
                            mylogger.info("Ran out of patient, early stop...")
                            break

                    if global_step >= args['total_steps']:
                        break

            if global_step >= args['total_steps']:
                break

            if patience >= args['patience']:
                mylogger.info("Ran out of patient, early stop...")
                break

            epoch += 1

    def pretarin_KB_step(self, data, train=True):
        with torch.cuda.amp.autocast(enabled=args['fp16']):
            # Encode and Decode
            response_lengths = [l1 - l2 for (l1, l2) in
                                zip(data['input_ids_lengths'], data['conv_GPT2_lengths'])]
            labels = self._cuda(torch.ones_like(data['input_ids']) * -1)
            for i, start in enumerate(data['conv_GPT2_lengths']):
                labels[i, start - 1:start - 1 + response_lengths[i]] = data['input_ids'][
                                                                       i, start:start + response_lengths[i]]
            if args['freeze_GPT']:
                hidden_states, _, _ = self.GPT2_model(data['input_ids'], labels=labels)
            else:
                hidden_states, classifier_output, loss_v = self.GPT2_model(data['input_ids'], labels=labels,
                                                                           type_loss=True)

            hidden = torch.stack([hidden_states[i, length - 1] for i, length in enumerate(data['conv_GPT2_lengths'])])
            global_pointer, kb_readout = self.extKnow.load_memory(data['context_arr'], data['kb_arr_lengths'],
                                                                  data['conv_arr_lengths'], hidden, hidden_states)
            # Loss calculation and backpropagation
            if args['freeze_GPT']:
                loss = self.criterion_bce(global_pointer, data['selector_index'])
            else:
                loss_g = self.criterion_bce(global_pointer, data['selector_index'])
                classifier_output = classifier_output.squeeze(-1)
                predict_output = torch.zeros_like(data['binary_labels'])
                for i, start in enumerate(data['conv_GPT2_lengths']):
                    predict_output[i, :response_lengths[i]] = classifier_output[i,
                                                              start - 1:start - 1 + response_lengths[i]]
                # Loss calculation and backpropagation
                loss_b = masked_binary_cross_entropy(predict_output, data['binary_labels'], response_lengths)
                loss = loss_v + loss_b + loss_g

            if train:
                self.loss += loss.item() / args['accumulation_steps']
                if not args['freeze_GPT']:
                    self.loss_g += loss_g.item() / args['accumulation_steps']
                    self.loss_v += loss_v.item() / args['accumulation_steps']
                    self.loss_b += loss_b.item() / args['accumulation_steps']
                loss /= args['accumulation_steps']
                return loss
            else:
                if not args['freeze_GPT']:
                    return loss.item(), loss_g.item(), loss_v.item(), loss_b.item()
                else:
                    return loss.item()

    def pretrain_KB(self, train, dev):
        if not args['freeze_GPT']:
            self.GPT2_model.train()
        self.extKnow.train()
        dev_score_best, patience, dev_score, step, global_step, epoch = 1e7, 0, 0.0, 0, 0, 0
        # Train
        while True:
            # Run the train function
            mylogger.info("Epoch:{}".format(epoch))
            pbar = tqdm(enumerate(train), total=len(train))
            for _, data in pbar:
                loss = self.pretarin_KB_step(data)
                if args['fp16']:
                    scaler.scale(loss).backward()
                else:
                    loss.backward()
                step += 1
                if step % args['accumulation_steps'] == 0:
                    global_step += 1
                    # Clip gradient norms
                    if args['fp16']:
                        if not args['freeze_GPT']:
                            scaler.unscale_(self.GPT2_model_optimizer)
                        scaler.unscale_(self.extKnow_optimizer)
                    torch.nn.utils.clip_grad_norm_(self.extKnow.parameters(), args['clip'])
                    if not args['freeze_GPT']:
                        torch.nn.utils.clip_grad_norm_(self.GPT2_model.parameters(), args['clip'])

                    # Update parameters with optimizers
                    if args['fp16']:
                        scaler.step(self.extKnow_optimizer)
                        if not args['freeze_GPT']:
                            scaler.step(self.GPT2_model_optimizer)
                        scaler.update()
                    else:
                        self.extKnow_optimizer.step()
                        if not args['freeze_GPT']:
                            self.GPT2_model_optimizer.step()

                    # Zero gradients of both optimizers
                    self.extKnow_scheduler.step()
                    self.extKnow_optimizer.zero_grad()
                    if not args['freeze_GPT']:
                        self.GPT2_model_scheduler.step()
                        self.GPT2_model_optimizer.zero_grad()

                        pbar.set_description(self.print_pretrain_KB_loss(step=global_step))

                    # Dev
                    if global_step % args['valid_steps'] == 0:
                        mylogger.info('Dev Start')
                        if not args['freeze_GPT']:
                            self.GPT2_model.eval()
                        self.extKnow.eval()
                        dev_loss, dev_loss_g, dev_loss_v, dev_loss_b = 0.0, 0.0, 0.0, 0.0
                        with torch.no_grad():
                            dev_pbar = tqdm(enumerate(dev), total=len(dev))
                            for dev_i, dev_data in dev_pbar:
                                if not args['freeze_GPT']:
                                    cur_dev_loss, cur_dev_loss_g, cur_dev_loss_v, cur_dev_loss_b = self.pretarin_KB_step(
                                        dev_data, train=False)
                                    dev_loss += cur_dev_loss
                                    dev_loss_g += cur_dev_loss_g
                                    dev_loss_v += cur_dev_loss_v
                                    dev_loss_b += cur_dev_loss_b
                                else:
                                    cur_dev_loss = self.pretarin_KB_step(dev_data, train=False)
                                    dev_loss += cur_dev_loss
                        dev_score = dev_loss / len(dev)
                        if not args['freeze_GPT']:
                            fitlog.add_metric({"dev": {
                                "dev loss": dev_score,
                                "dev loss_g": dev_loss_g / len(dev),
                                "dev loss_v": dev_loss_v / len(dev),
                                "dev loss_b": dev_loss_b / len(dev),
                            }}, step=global_step)
                        else:
                            fitlog.add_metric({"dev": {"dev loss": dev_score}}, step=global_step)

                        if dev_score < dev_score_best:
                            dev_score_best = dev_score
                            patience = 0
                            fitlog.add_best_metric({"dev": {"dev loss": dev_score}})
                            if not args['freeze_GPT']:
                                torch.save(self.GPT2_model.state_dict(), os.path.join(args['output_dir'], 'GPT2.th'))
                            torch.save(self.extKnow.state_dict(), os.path.join(args['output_dir'], 'enc_kb.th'))
                        else:
                            patience += 1

                        if not args['freeze_GPT']:
                            self.GPT2_model.train()
                        self.extKnow.train()

                        if patience >= args['patience']:
                            mylogger.info("Ran out of patient, early stop...")
                            break

                    if global_step >= args['total_steps']:
                        break

            if global_step >= args['total_steps']:
                break

            if patience >= args['patience']:
                mylogger.info("Ran out of patient, early stop...")
                break

            epoch += 1

    def fine_tune_step(self, data, train=True):
        with torch.cuda.amp.autocast(enabled=args['fp16']):
            # Encode and Decode
            response_lengths = [l1 - l2 for (l1, l2) in zip(data['input_ids_lengths'], data['conv_GPT2_lengths'])]
            max_target_length = max(response_lengths)
            batch_size = data['context_arr'].size()[0]
            labels = self._cuda(torch.ones_like(data['input_ids']) * -1)

            for i, start in enumerate(data['conv_GPT2_lengths']):
                labels[i, start - 1:start - 1 + response_lengths[i]] = data['input_ids'][i,
                                                                       start:start + response_lengths[i]]
            hidden_states, classifier_output, loss_v = self.GPT2_model(data['input_ids'], labels=labels,
                                                                       type_loss=True)
            hidden = torch.stack([hidden_states[i, length - 1] for i, length in enumerate(data['conv_GPT2_lengths'])])
            global_pointer, kb_readout = self.extKnow.load_memory(data['context_arr'], data['kb_arr_lengths'],
                                                                  data['conv_arr_lengths'], hidden, hidden_states)
            all_decoder_outputs_ptr = self._cuda(
                torch.zeros(max_target_length, batch_size, data['context_arr'].size()[1]))
            yindex_list = []
            for t in range(0, max_target_length):
                yindex_list.append([data['conv_GPT2_lengths'][i] - 1 + t
                                    if data['conv_GPT2_lengths'][i] - 1 + t < data['input_ids_lengths'][i]
                                    else data['input_ids_lengths'][i] - 1 for i in list(range(batch_size))])
            for t in range(0, max_target_length):
                query_vector = hidden_states[list(range(batch_size)), yindex_list[t]]
                all_decoder_outputs_ptr[t] = self.extKnow(query_vector, global_pointer)[1]
            # Loss calculation and backpropagation
            loss_g = self.criterion_bce(global_pointer, data['selector_index'])
            loss_l = masked_cross_entropy(
                all_decoder_outputs_ptr.transpose(0, 1).contiguous(),
                data['ptr_index'].contiguous(),
                response_lengths)

            classifier_output = classifier_output.squeeze(-1)
            predict_output = torch.zeros_like(data['binary_labels'])
            for i, start in enumerate(data['conv_GPT2_lengths']):
                predict_output[i, :response_lengths[i]] = classifier_output[i,
                                                          start - 1:start - 1 + response_lengths[i]]
            loss_b = masked_binary_cross_entropy(predict_output, data['binary_labels'], response_lengths)
            loss = loss_g + loss_v + loss_l + loss_b

            if train:
                self.loss += loss.item() / args['accumulation_steps']
                self.loss_g += loss_g.item() / args['accumulation_steps']
                self.loss_v += loss_v.item() / args['accumulation_steps']
                self.loss_l += loss_l.item() / args['accumulation_steps']
                self.loss_b += loss_b.item() / args['accumulation_steps']
                loss /= args['accumulation_steps']
                return loss
            else:
                return loss.item(), loss_g.item(), loss_v.item(), loss_l.item(), loss_b.item()

    def fine_tune(self, train, dev):
        self.GPT2_model.train()
        self.extKnow.train()
        dev_score_best, patience, dev_score, step, global_step, epoch = 0, 0, 0.0, 0, 0, 0
        # Training
        while True:
            mylogger.info("Epoch:{}".format(epoch))
            # Run the train function
            pbar = tqdm(enumerate(train), total=len(train))
            for _, data in pbar:
                loss = self.fine_tune_step(data)
                if args['fp16']:
                    scaler.scale(loss).backward()
                else:
                    loss.backward()
                step += 1
                if step % args['accumulation_steps'] == 0:
                    global_step += 1
                    # Clip gradient norms
                    if args['fp16']:
                        scaler.unscale_(self.GPT2_model_optimizer)
                        scaler.unscale_(self.extKnow_optimizer)
                    torch.nn.utils.clip_grad_norm_(self.GPT2_model.parameters(), args['clip'])
                    torch.nn.utils.clip_grad_norm_(self.extKnow.parameters(), args['clip'])

                    # Update parameters with optimizers
                    if args['fp16']:
                        scaler.step(self.extKnow_optimizer)
                        scaler.step(self.GPT2_model_optimizer)
                        scaler.update()
                    else:
                        self.extKnow_optimizer.step()
                        self.GPT2_model_optimizer.step()

                    # Zero gradients of both optimizers
                    self.extKnow_scheduler.step()
                    self.GPT2_model_scheduler.step()
                    self.extKnow_optimizer.zero_grad()
                    self.GPT2_model_optimizer.zero_grad()

                    pbar.set_description(self.print_loss(step=global_step))

                    if global_step >= args['total_steps']:
                        break

            # Dev only when a epoch is end
            if (epoch + 1) % int(args['evalp']) == 0:
                mylogger.info('Dev Start')
                dev_score = self.evaluate(dev, dev_score_best, metric=metric, epoch=epoch)
                print(self.eval_log, file=eval_logger)
                if dev_score >= dev_score_best:
                    dev_score_best = dev_score
                    patience = 0
                else:
                    patience += 1

                if patience == args['patience']:
                    mylogger.info("Ran out of patient, early stop...")
                    break

            if global_step >= args['total_steps']:
                break

            epoch += 1

    def encode_and_decode(self, data, max_target_length):
        story, conv_story = data['context_arr'], data['conv_arr']
        # Encode dialog history and KB to vectors
        # TODO: hidden states 怎么作为 H 加到 MN 中去   是需要对应的
        # TODO: kb_readout 怎么融入到下阶段的生成中去
        # Get the words that can be copy from the memory
        batch_size = len(data['context_arr_lengths'])
        copy_list = []
        for elm in data['context_arr_plain']:
            elm_temp = [word_arr[0] for word_arr in elm]
            copy_list.append(elm_temp)

        # Initialize variables for vocab and pointer
        init_input = self._cuda(torch.LongTensor(self.tokenizer.encode(SYS) * batch_size))
        memory_mask_for_step = self._cuda(torch.ones(story.size()[0], story.size()[1]))
        decoded_fine, decoded_coarse = [], []
        global_pointers = []
        # Start to generate word-by-word
        for i in range(batch_size):
            decoded_fine.append([])
            decoded_coarse.append([])
            hidden_states, past = self.GPT2_model.transformer(data['conv_GPT2'][i])
            global_pointer, kb_readout = self.extKnow.load_memory(story[i].unsqueeze(0), data['kb_arr_lengths'],
                                                                  data['conv_arr_lengths'],
                                                                  hidden_states[-1].unsqueeze(0), hidden_states)
            global_pointers.append(global_pointer)
            prev_input = init_input[i]
            for t in range(max_target_length):
                hidden, past = self.GPT2_model.transformer(prev_input.unsqueeze(0), past=past)
                logits = self.GPT2_model.lm_head(hidden)
                _, topvi = logits.data.topk(1)
                prev_input = topvi.squeeze()
                # query the external konwledge using the hidden state of sketch RNN
                query_vector = hidden
                prob_soft, _ = self.extKnow(query_vector, global_pointer)
                search_len = min(5, data['context_arr_lengths'][i])
                prob_soft = prob_soft * memory_mask_for_step
                _, toppi = prob_soft.data.topk(search_len)
                token = topvi.item()
                token = self.tokenizer.decode(token)
                decoded_coarse[-1].append(token)
                if '@' in token:
                    cw = 'UNK'
                    for si in range(search_len):
                        if toppi[:, si][i] < data['context_arr_lengths'][i] - 1:
                            cw = copy_list[i][toppi[:, si][i].item()]
                            break
                    decoded_fine[-1].append(cw)

                    if args['record']:
                        memory_mask_for_step[i, toppi[:, si][i].item()] = 0
                else:
                    decoded_fine[-1].append(token)
                if token == EOS:
                    break

        return decoded_fine, decoded_coarse, torch.cat(global_pointers)

    def evaluate(self, dev, metric_best, metric=None, epoch=0, Test=False):
        mylogger.info("STARTING EVALUATION")
        # Set to not-training mode to disable dropout
        self.GPT2_model.eval()
        self.extKnow.eval()

        ref, hyp = [], []
        acc, total = 0, 0
        if args['dataset'] == 'smd':
            F1_pred, F1_cal_pred, F1_nav_pred, F1_wet_pred = 0, 0, 0, 0
            F1_count, F1_cal_count, F1_nav_count, F1_wet_count = 0, 0, 0, 0
            TP_all, FP_all, FN_all = 0, 0, 0

            TP_sche, FP_sche, FN_sche = 0, 0, 0
            TP_wea, FP_wea, FN_wea = 0, 0, 0
            TP_nav, FP_nav, FN_nav = 0, 0, 0
        elif args['dataset'] == 'woz':
            F1_pred, F1_police_pred, F1_restaurant_pred, F1_hospital_pred, F1_attraction_pred, F1_hotel_pred = 0, 0, 0, 0, 0, 0
            F1_count, F1_police_count, F1_restaurant_count, F1_hospital_count, F1_attraction_count, F1_hotel_count = 0, 0, 0, 0, 0, 0
            TP_all, FP_all, FN_all = 0, 0, 0

            TP_restaurant, FP_restaurant, FN_restaurant = 0, 0, 0
            TP_attraction, FP_attraction, FN_attraction = 0, 0, 0
            TP_hotel, FP_hotel, FN_hotel = 0, 0, 0
        elif args['dataset'] == 'cam':
            F1_pred, F1_count = 0, 0

            TP_all, FP_all, FN_all = 0, 0, 0

        pbar = tqdm(enumerate(dev), total=len(dev))

        if args['dataset'] == 'smd':
            entity_path = './data/fine-tune/SMD/kvret_entities.json'
        elif args['dataset'] == 'woz':
            entity_path = './data/fine-tune/MULTIWOZ2.1/global_entities.json'
        elif args['dataset'] == 'cam':
            entity_path = './data/fine-tune/CamRest676/CamRest676_entities.json'

        with open(entity_path) as f:
            global_entity = json.load(f)
            global_entity_list = []
            for key in global_entity.keys():
                if key != 'poi':
                    global_entity_list += [item.lower().replace(' ', '_') for item in global_entity[key]]
                else:
                    for item in global_entity['poi']:
                        global_entity_list += [item[k].lower().replace(' ', '_') for k in item.keys()]
            global_entity_list = list(set(global_entity_list))

        with torch.no_grad():
            for j, data_dev in pbar:
                # Encode and Decode
                decoded_fine, decoded_coarse, global_pointer = self.encode_and_decode(data_dev, self.max_resp_len)
                for bi, row in enumerate(decoded_fine):
                    st = ''
                    for e in row:
                        if e == EOS:
                            break
                        else:
                            st += e + ' '
                    st_c = ''
                    for e in decoded_coarse[bi]:
                        if e == EOS:
                            break
                        else:
                            st_c += e + ' '
                    pred_sent = st.lstrip().rstrip()
                    pred_sent_coarse = st_c.lstrip().rstrip()
                    gold_sent = data_dev['response_plain'][bi].lstrip().rstrip()
                    ref.append(gold_sent)
                    hyp.append(pred_sent)

                    if args['dataset'] == 'smd':
                        # compute F1 SCORE
                        single_tp, single_fp, single_fn, single_f1, count = self.compute_prf(data_dev['ent_index'][bi],
                                                                                             pred_sent.split(),
                                                                                             global_entity_list,
                                                                                             data_dev['kb_arr_plain'][
                                                                                                 bi])
                        F1_pred += single_f1
                        F1_count += count
                        TP_all += single_tp
                        FP_all += single_fp
                        FN_all += single_fn

                        single_tp, single_fp, single_fn, single_f1, count = self.compute_prf(
                            data_dev['ent_idx_cal'][bi],
                            pred_sent.split(),
                            global_entity_list,
                            data_dev['kb_arr_plain'][bi])
                        F1_cal_pred += single_f1
                        F1_cal_count += count
                        TP_sche += single_tp
                        FP_sche += single_fp
                        FN_sche += single_fn

                        single_tp, single_fp, single_fn, single_f1, count = self.compute_prf(
                            data_dev['ent_idx_nav'][bi],
                            pred_sent.split(),
                            global_entity_list,
                            data_dev['kb_arr_plain'][bi])
                        F1_nav_pred += single_f1
                        F1_nav_count += count
                        TP_nav += single_tp
                        FP_nav += single_fp
                        FN_nav += single_fn

                        single_tp, single_fp, single_fn, single_f1, count = self.compute_prf(
                            data_dev['ent_idx_wet'][bi],
                            pred_sent.split(),
                            global_entity_list,
                            data_dev['kb_arr_plain'][bi])
                        F1_wet_pred += single_f1
                        F1_wet_count += count
                        TP_wea += single_tp
                        FP_wea += single_fp
                        FN_wea += single_fn

                    elif args['dataset'] == 'woz':
                        # coimpute F1 SCORE
                        single_tp, single_fp, single_fn, single_f1, count = self.compute_prf(data_dev['ent_index'][bi],
                                                                                             pred_sent.split(),
                                                                                             global_entity_list,
                                                                                             data_dev['kb_arr_plain'][
                                                                                                 bi])
                        F1_pred += single_f1
                        F1_count += count
                        TP_all += single_tp
                        FP_all += single_fp
                        FN_all += single_fn

                        single_tp, single_fp, single_fn, single_f1, count = self.compute_prf(
                            data_dev['ent_idx_restaurant'][bi],
                            pred_sent.split(),
                            global_entity_list,
                            data_dev['kb_arr_plain'][bi])
                        F1_restaurant_pred += single_f1
                        F1_restaurant_count += count
                        TP_restaurant += single_tp
                        FP_restaurant += single_fp
                        FN_restaurant += single_fn

                        single_tp, single_fp, single_fn, single_f1, count = self.compute_prf(
                            data_dev['ent_idx_attraction'][bi],
                            pred_sent.split(),
                            global_entity_list,
                            data_dev['kb_arr_plain'][bi])
                        F1_attraction_pred += single_f1
                        F1_attraction_count += count
                        TP_attraction += single_tp
                        FP_attraction += single_fp
                        FN_attraction += single_fn

                        single_tp, single_fp, single_fn, single_f1, count = self.compute_prf(
                            data_dev['ent_idx_hotel'][bi],
                            pred_sent.split(),
                            global_entity_list,
                            data_dev['kb_arr_plain'][bi])
                        F1_hotel_pred += single_f1
                        F1_hotel_count += count
                        TP_hotel += single_tp
                        FP_hotel += single_fp
                        FN_hotel += single_fn

                    elif args['dataset'] == 'cam':
                        # compute F1 SCORE
                        single_tp, single_fp, single_fn, single_f1, count = self.compute_prf(data_dev['ent_index'][bi],
                                                                                             pred_sent.split(),
                                                                                             global_entity_list,
                                                                                             data_dev['kb_arr_plain'][
                                                                                                 bi])
                        F1_pred += single_f1
                        F1_count += count
                        TP_all += single_tp
                        FP_all += single_fp
                        FN_all += single_fn

                        # compute Per-response Accuracy Score
                    total += 1
                    if gold_sent == pred_sent:
                        acc += 1

        # Set back to training mode
        self.GPT2_model.train()
        self.extKnow.train()

        bleu_score = moses_multi_bleu(np.array(hyp), np.array(ref), lowercase=True, mylogger=mylogger)
        acc_score = acc / float(total)

        if args['dataset'] == 'smd':
            F1_macro_score = F1_pred / float(F1_count)
            F1_macro_sche_score = F1_cal_pred / float(F1_cal_count)
            F1_macro_wea_score = F1_wet_pred / float(F1_wet_count)
            F1_macro_nav_score = F1_nav_pred / float(F1_nav_count)

            P_score = TP_all / float(TP_all + FP_all) if (TP_all + FP_all) != 0 else 0
            R_score = TP_all / float(TP_all + FN_all) if (TP_all + FN_all) != 0 else 0
            P_nav_score = TP_nav / float(TP_nav + FP_nav) if (TP_nav + FP_nav) != 0 else 0
            P_sche_score = TP_sche / float(TP_sche + FP_sche) if (TP_sche + FP_sche) != 0 else 0
            P_wea_score = TP_wea / float(TP_wea + FP_wea) if (TP_wea + FP_wea) != 0 else 0
            R_nav_score = TP_nav / float(TP_nav + FN_nav) if (TP_nav + FN_nav) != 0 else 0
            R_sche_score = TP_sche / float(TP_sche + FN_sche) if (TP_sche + FN_sche) != 0 else 0
            R_wea_score = TP_wea / float(TP_wea + FN_wea) if (TP_wea + FN_wea) != 0 else 0

            F1_micro_score = self.compute_F1(P_score, R_score)
            F1_micro_sche_score = self.compute_F1(P_sche_score, R_sche_score)
            F1_micro_wea_score = self.compute_F1(P_wea_score, R_wea_score)
            F1_micro_nav_score = self.compute_F1(P_nav_score, R_nav_score)
            self.eval_log = '{},{},{},{},{},{},{},{},{},{},{}'.format(epoch + 1, acc_score, bleu_score, F1_macro_score,
                                                                      F1_micro_score, F1_macro_sche_score,
                                                                      F1_macro_wea_score,
                                                                      F1_macro_nav_score, F1_micro_sche_score,
                                                                      F1_micro_wea_score, F1_micro_nav_score)
            metric_result = {"acc": acc_score,
                             "bleu": bleu_score,
                             "F1 micro": F1_micro_score,
                             "F1 micro sche": F1_micro_sche_score,
                             "F1 micro wea": F1_micro_wea_score,
                             "F1 micro nav": F1_micro_nav_score
                             }
            fitlog.add_metric({"test" if Test else "dev": metric_result}, step=epoch)
        elif args['dataset'] == 'woz':
            F1_macro_score = F1_pred / float(F1_count)
            F1_macro_restaurant_score = F1_restaurant_pred / float(F1_restaurant_count)
            F1_macro_attraction_score = F1_attraction_pred / float(F1_attraction_count)
            F1_macro_hotel_score = F1_hotel_pred / float(F1_hotel_count)

            P_score = TP_all / float(TP_all + FP_all) if (TP_all + FP_all) != 0 else 0
            R_score = TP_all / float(TP_all + FN_all) if (TP_all + FN_all) != 0 else 0
            P_restaurant_score = TP_restaurant / float(TP_restaurant + FP_restaurant) if (
                                                                                                 TP_restaurant + FP_restaurant) != 0 else 0
            P_attraction_score = TP_attraction / float(TP_attraction + FP_attraction) if (
                                                                                                 TP_attraction + FP_attraction) != 0 else 0
            P_hotel_score = TP_hotel / float(TP_hotel + FP_hotel) if (TP_hotel + FP_hotel) != 0 else 0

            R_restaurant_score = TP_restaurant / float(TP_restaurant + FN_restaurant) if (
                                                                                                 TP_restaurant + FN_restaurant) != 0 else 0
            R_attraction_score = TP_attraction / float(TP_attraction + FN_attraction) if (
                                                                                                 TP_attraction + FN_attraction) != 0 else 0
            R_hotel_score = TP_hotel / float(TP_hotel + FN_hotel) if (TP_hotel + FN_hotel) != 0 else 0

            F1_micro_score = self.compute_F1(P_score, R_score)
            F1_micro_restaurant_score = self.compute_F1(P_restaurant_score, R_restaurant_score)
            F1_micro_attraction_score = self.compute_F1(P_attraction_score, R_attraction_score)
            F1_micro_hotel_score = self.compute_F1(P_hotel_score, R_hotel_score)
            self.eval_log = '{},{},{},{},{},{},{},{},{},{},{}'.format(epoch + 1, acc_score, bleu_score, F1_macro_score,
                                                                      F1_micro_score, F1_macro_restaurant_score,
                                                                      F1_macro_attraction_score,
                                                                      F1_macro_hotel_score, F1_micro_restaurant_score,
                                                                      F1_micro_attraction_score, F1_micro_hotel_score)
            metric_result = {"acc": acc_score,
                             "bleu": bleu_score,
                             "F1 micro": F1_micro_score,
                             "F1 micro restaurant": F1_micro_restaurant_score,
                             "F1 micro attraction": F1_micro_attraction_score,
                             "F1 micro hotel": F1_micro_hotel_score
                             }
            fitlog.add_metric({"test" if Test else "dev": metric_result}, step=epoch)
        elif args['dataset'] == 'cam':
            F1_macro_score = F1_pred / float(F1_count)

            P_score = TP_all / float(TP_all + FP_all) if (TP_all + FP_all) != 0 else 0
            R_score = TP_all / float(TP_all + FN_all) if (TP_all + FN_all) != 0 else 0

            F1_micro_score = self.compute_F1(P_score, R_score)
            self.eval_log = '{},{},{},{},{}'.format(epoch + 1, acc_score, bleu_score, F1_macro_score,
                                                    F1_micro_score)
            metric_result = {"acc": acc_score,
                             "bleu": bleu_score,
                             "F1 micro": F1_micro_score
                             }
            fitlog.add_metric({"test" if Test else "dev": metric_result}, step=epoch)

        if Test:
            mylogger.info('Test Finish!')
            fitlog.add_best_metric({"test": metric_result})
        else:
            mylogger.info('Dev Finish!')

        if metric == 'BLEU':
            if bleu_score >= metric_best:
                self.save_model(str(epoch + 1) + '.BLEU.' + str(bleu_score))
                mylogger.info("MODEL SAVED")
                fitlog.add_best_metric({"dev": metric_result})
            return bleu_score
        elif metric == 'ENTF1':
            if F1_micro_score >= metric_best:
                self.save_model(str(epoch + 1) + '.ENTF1.{:.4f}'.format(F1_micro_score))
                mylogger.info("MODEL SAVED")
                fitlog.add_best_metric({"dev": metric_result})
            return F1_micro_score
        else:
            if acc_score >= metric_best:
                self.save_model(str(epoch + 1) + '.ACC.{:.4f}'.format(acc_score))
                mylogger.info("MODEL SAVED")
                fitlog.add_best_metric({"dev": metric_result})
            return acc_score

    def compute_prf(self, gold, pred, global_entity_list, kb_plain):
        local_kb_word = [k[0] for k in kb_plain]
        TP, FP, FN = 0, 0, 0
        if len(gold) != 0:
            count = 1
            for g in gold:
                if g in pred:
                    TP += 1
                else:
                    FN += 1
            for p in set(pred):
                if p in global_entity_list or p in local_kb_word:
                    if p not in gold:
                        FP += 1
            precision = TP / float(TP + FP) if (TP + FP) != 0 else 0
            recall = TP / float(TP + FN) if (TP + FN) != 0 else 0
            F1 = 2 * precision * recall / float(precision + recall) if (precision + recall) != 0 else 0
        else:
            precision, recall, F1, count = 0, 0, 0, 0
        return TP, FP, FN, F1, count

    def compute_prf_sketch(self, gold, pred, global_entity_type):
        TP, FP, FN = 0, 0, 0
        pos = {value: 0 for key, value in enumerate(gold)}
        for g in gold:
            pos[g] += 1

        if len(gold) != 0:
            count = 1
            for g in gold:
                if g in pred:
                    TP += 1
                    pos[g] -= 1
                else:
                    FN += 1
            for p in pred:
                if p in global_entity_type:
                    if p not in gold:
                        FP += 1
                    # elif pos[p] == 0:
                    #     FP += 1
            precision = TP / float(TP + FP) if (TP + FP) != 0 else 0
            recall = TP / float(TP + FN) if (TP + FN) != 0 else 0
            F1 = 2 * precision * recall / float(precision + recall) if (precision + recall) != 0 else 0
        else:
            precision, recall, F1, count = 0, 0, 0, 0
        return TP, FP, FN, F1, count

    def compute_F1(self, precision, recall):
        F1 = 2 * precision * recall / float(precision + recall) if (precision + recall) != 0 else 0
        return F1
