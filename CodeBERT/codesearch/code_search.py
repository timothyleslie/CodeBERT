from transformers import BertModel
import datasets
from torch.utils.data import DataLoader, Dataset, SequentialSampler, RandomSampler, TensorDataset
import torch.nn as nn
import torch
import torch.nn.functional
import os
import torch.utils.data
import gc
from transformers import AutoTokenizer, AutoConfig, get_linear_schedule_with_warmup
import numpy as np
import sys
from tqdm import tqdm
import logging
import json

from my_dataset import TBAClassifyDataset, QAMemClassifyDataset, QAClassifyDataset, CrossClassifyDataset
import torch.nn.functional as F

logger = logging.getLogger(__name__)


class InputFeatures(object):
    """A single training/test features for a example."""

    def __init__(self,
                 code_tokens,
                 code_ids,
                 nl_tokens,
                 nl_ids,

                 ):
        self.code_tokens = code_tokens
        self.code_ids = code_ids
        self.nl_tokens = nl_tokens
        self.nl_ids = nl_ids


def convert_examples_to_features(js, tokenizer, text_max_len):
    # code
    if 'code_tokens' in js:
        code = ' '.join(js['code_tokens'])
    else:
        code = ' '.join(js['function_tokens'])

    code_tokens = tokenizer.tokenize(code)[:text_max_len-2]
    code_tokens = [tokenizer.cls_token]+code_tokens+[tokenizer.sep_token]
    code_ids = tokenizer.convert_tokens_to_ids(code_tokens)
    padding_length = text_max_len - len(code_ids)
    code_ids += [tokenizer.pad_token_id]*padding_length

    nl = ' '.join(js['docstring_tokens'])
    nl_tokens = tokenizer.tokenize(nl)[:text_max_len-2]
    nl_tokens = [tokenizer.cls_token]+nl_tokens+[tokenizer.sep_token]
    nl_ids = tokenizer.convert_tokens_to_ids(nl_tokens)
    padding_length = text_max_len - len(nl_ids)
    nl_ids += [tokenizer.pad_token_id]*padding_length

    return InputFeatures(code_tokens, code_ids, nl_tokens, nl_ids)


class TextDataset(Dataset):
    def __init__(self, tokenizer, text_max_len, file_path=None):
        self.examples = []
        data = []
        with open(file_path) as f:
            for line in f:
                line = line.strip()
                js = json.loads(line)
                data.append(js)
        for js in data:
            self.examples.append(convert_examples_to_features(
                js, tokenizer, text_max_len))
        if 'train' in file_path:
            for idx, example in enumerate(self.examples[:3]):
                logger.info("*** Example ***")
                logger.info("idx: {}".format(idx))
                logger.info("code_tokens: {}".format(
                    [x.replace('\u0120', '_') for x in example.code_tokens]))
                logger.info("code_ids: {}".format(
                    ' '.join(map(str, example.code_ids))))
                logger.info("nl_tokens: {}".format(
                    [x.replace('\u0120', '_') for x in example.nl_tokens]))
                logger.info("nl_ids: {}".format(
                    ' '.join(map(str, example.nl_ids))))

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, i):
        return (torch.tensor(self.examples[i].code_ids), torch.tensor(self.examples[i].nl_ids))


def clean_input_ids(input_ids, attention_mask, token_type_ids):
    max_seq_len = torch.max(attention_mask.sum(-1))

    # ensure only pad be filtered
    dropped_attention_mask = attention_mask[:, max_seq_len:]
    assert torch.max(dropped_attention_mask.sum(-1)) == 0

    input_ids = input_ids[:, :max_seq_len]
    token_type_ids = token_type_ids[:, :max_seq_len]
    attention_mask = attention_mask[:, :max_seq_len]

    return input_ids, attention_mask, token_type_ids


class BiConfig:
    def __init__(self, tokenizer_len, pretrained_bert_path='prajjwal1/bert-small', num_labels=2,
                 word_embedding_len=512, sentence_embedding_len=512, composition='pooler'):
        self.tokenizer_len = tokenizer_len
        self.pretrained_bert_path = pretrained_bert_path
        self.num_labels = num_labels
        self.word_embedding_len = word_embedding_len
        self.sentence_embedding_len = sentence_embedding_len
        self.composition = composition

    def __str__(self):
        print("*" * 20 + "config" + "*" * 20)
        print("tokenizer_len:", self.tokenizer_len)
        print("pretrained_bert_path:", self.pretrained_bert_path)
        print("num_labels:", self.num_labels)
        print("word_embedding_len:", self.word_embedding_len)
        print("sentence_embedding_len:", self.sentence_embedding_len)
        print("composition:", self.composition)
        

class BiEncoder(nn.Module):
    def __init__(self, config):
        super(BiEncoder, self).__init__()

        self.config = config

        # 毕竟num_label也算是memory的一部分
        self.num_labels = config.num_labels
        self.sentence_embedding_len = config.sentence_embedding_len

        # 这个学习率不一样
        self.bert_model = BertModel.from_pretrained(
            config.pretrained_bert_path)
        self.bert_model.resize_token_embeddings(config.tokenizer_len)
        # 这个embedding的grad会被计入bert model里，很好
        self.embeddings = self.bert_model.get_input_embeddings()

    def get_rep_by_pooler(self, input_ids, attention_mask):
        token_type_ids = torch.zeros(*input_ids.shape, device=input_ids.device, dtype=input_ids.dtype)
        # print(token_type_ids)
        # print(input_ids)
        # print(attention_mask)
        input_ids, attention_mask, token_type_ids = clean_input_ids(
            input_ids, attention_mask, token_type_ids)

        out = self.bert_model(input_ids=input_ids, attention_mask=attention_mask,
                              token_type_ids=token_type_ids)

        out = out['pooler_output']

        return out

    def forward(self, q_input_ids, q_attention_mask,
                a_input_ids, a_attention_mask, return_vector=False):
        if self.config.composition == 'pooler':
            q_embeddings = self.get_rep_by_pooler(input_ids=q_input_ids,
                                                  attention_mask=q_attention_mask)

            a_embeddings = self.get_rep_by_pooler(input_ids=a_input_ids,
                                                  attention_mask=a_attention_mask)
        else:
            raise Exception(
                f"Composition {self.config.composition} is not supported!!")

        if return_vector:
            return q_embeddings, a_embeddings

        logits = torch.matmul(q_embeddings, a_embeddings.t())

        return logits
    
    
class InputMemoryConfig:
    def __init__(self, tokenizer_len, pretrained_bert_path='prajjwal1/bert-small', num_labels=2,
                 word_embedding_len=512, sentence_embedding_len=512, memory_num=50, composition='pooler'):
        self.tokenizer_len = tokenizer_len
        self.pretrained_bert_path = pretrained_bert_path
        self.num_labels = num_labels
        self.word_embedding_len = word_embedding_len
        self.sentence_embedding_len = sentence_embedding_len
        self.memory_num = memory_num
        self.composition = composition

    def __str__(self):
        print("*" * 20 + "config" + "*" * 20)
        print("tokenizer_len:", self.tokenizer_len)
        print("pretrained_bert_path:", self.pretrained_bert_path)
        print("num_labels:", self.num_labels)
        print("word_embedding_len:", self.word_embedding_len)
        print("sentence_embedding_len:", self.sentence_embedding_len)
        print("memory_num:", self.memory_num)
        print("composition:", self.composition)


class InputMemory(nn.Module):
    def __init__(self, config):
        super(InputMemory, self).__init__()

        self.config = config

        # 毕竟num_label也算是memory的一部分
        self.num_labels = config.num_labels
        self.sentence_embedding_len = config.sentence_embedding_len
        self.memory_num = config.memory_num

        # 这个学习率不一样
        self.bert_model = BertModel.from_pretrained(
            config.pretrained_bert_path)
        self.bert_model.resize_token_embeddings(config.tokenizer_len)
        # 这个embedding的grad会被计入bert model里，很好
        self.embeddings = self.bert_model.get_input_embeddings()

        # 记忆力模块
        self.memory_for_answer = nn.Parameter(torch.randn(
            config.memory_num, config.word_embedding_len))
        self.memory_for_question = nn.Parameter(
            torch.randn(config.memory_num, config.word_embedding_len))

    def get_rep_by_pooler(self, input_ids, attention_mask, is_question):
        token_type_ids = torch.zeros(*input_ids.shape, device=input_ids.device, dtype=torch.long)

        input_ids, attention_mask, token_type_ids = clean_input_ids(
            input_ids, attention_mask, token_type_ids)

        # ----------------------通过memory来丰富信息---------------------
        # 要确认训练时它有没有被修改
        memory_len_one_tensor = torch.tensor([1] * self.config.memory_num, requires_grad=False,
                                             device=input_ids.device)
        zero_tensor = torch.tensor(
            [0], requires_grad=False, device=input_ids.device)

        # 获得隐藏层输出, (batch, sequence, embedding)
        temp_embeddings = self.embeddings(input_ids)

        final_embeddings = None
        final_attention_mask = None
        final_token_type_ids = None

        # input_ids (batch, sequence)
        for index, batch_attention_mask in enumerate(attention_mask):
            input_embeddings = temp_embeddings[index][batch_attention_mask == 1]
            pad_embeddings = temp_embeddings[index][batch_attention_mask == 0]

            if is_question:
                whole_embeddings = torch.cat(
                    (input_embeddings[0:1], self.memory_for_question, input_embeddings[1:], pad_embeddings), dim=0)
            else:
                whole_embeddings = torch.cat(
                    (input_embeddings[0:1], self.memory_for_answer, input_embeddings[1:], pad_embeddings), dim=0)

            # 处理attention_mask
            whole_attention_mask = torch.cat((batch_attention_mask[batch_attention_mask == 1], memory_len_one_tensor,
                                              batch_attention_mask[batch_attention_mask == 0]), dim=-1)

            # 处理token_type_id
            remain_token_type_ids_len = batch_attention_mask.shape[0] + \
                self.memory_num - input_embeddings.shape[0]
            whole_token_type_ids = torch.cat((token_type_ids[index][batch_attention_mask == 1],
                                              zero_tensor.repeat(remain_token_type_ids_len)), dim=-1)

            whole_embeddings = whole_embeddings.unsqueeze(0)
            whole_attention_mask = whole_attention_mask.unsqueeze(0)
            whole_token_type_ids = whole_token_type_ids.unsqueeze(0)

            if final_embeddings is None:
                final_embeddings = whole_embeddings
                final_attention_mask = whole_attention_mask
                final_token_type_ids = whole_token_type_ids
            else:
                final_embeddings = torch.cat(
                    (final_embeddings, whole_embeddings), dim=0)
                final_attention_mask = torch.cat(
                    (final_attention_mask, whole_attention_mask), dim=0)
                final_token_type_ids = torch.cat(
                    (final_token_type_ids, whole_token_type_ids), dim=0)

        out = self.bert_model(inputs_embeds=final_embeddings, attention_mask=final_attention_mask,
                              token_type_ids=final_token_type_ids)

        out = out['pooler_output']

        return out

    def forward(self, q_input_ids, q_attention_mask,
                a_input_ids, a_attention_mask, return_vector=False):
        if self.config.composition == 'pooler':
            q_embeddings = self.get_rep_by_pooler(input_ids=q_input_ids,
                                                  attention_mask=q_attention_mask, is_question=True)

            a_embeddings = self.get_rep_by_pooler(input_ids=a_input_ids,
                                                  attention_mask=a_attention_mask, is_question=False)
        else:
            raise Exception(
                f"Composition {self.config.composition} is not supported!!")

        if return_vector:
            return q_embeddings, a_embeddings

        logits = torch.matmul(q_embeddings, a_embeddings.t())

        return logits


class TrainWholeModel:
    def __init__(self, args, config=None):
        # 读取一些参数并存起来-------------------------------------------------------------------
        self.__read_args_for_train(args)
        self.args = args

        # 设置gpu-------------------------------------------------------------------
        # os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
        os.environ["TOKENIZERS_PARALLELISM"] = "false"
        os.environ["CUDA_VISIBLE_DEVICES"] = args.nvidia_number

        # for data_parallel-------------------------------------------------------------------
        nvidia_number = len(args.nvidia_number.split(","))
        self.device_ids = [i for i in range(nvidia_number)]

        if not torch.cuda.is_available():
            raise Exception("No cuda available!")

        local_rank = 0
        if self.data_distribute:
            local_rank = self.local_rank
            torch.cuda.set_device(local_rank)
            torch.distributed.init_process_group(
                'nccl',
                    init_method='env://'
            )
        else:
            torch.cuda.set_device(local_rank)

        print(f"local rank: {local_rank}")
        self.device = torch.device(f'cuda:{local_rank}')

        # 读取tokenizer-------------------------------------------------------------------
        tokenizer_path = args.pretrained_bert_path.replace("/", "_")
        tokenizer_path = tokenizer_path.replace("\\", "_")
        if tokenizer_path[0] != "_":
            tokenizer_path = "_" + tokenizer_path

        # read from disk or save to disk
        if os.path.exists("./tokenizer/" + tokenizer_path):
            self.tokenizer = AutoTokenizer.from_pretrained(
                "./tokenizer/" + tokenizer_path)
        else:
            print("first time use this tokenizer, downloading...")
            self.tokenizer = AutoTokenizer.from_pretrained(
                args.pretrained_bert_path)
            tokenizer_config = AutoConfig.from_pretrained(
                args.pretrained_bert_path)

            self.tokenizer.save_pretrained("./tokenizer/" + tokenizer_path)
            tokenizer_config.save_pretrained("./tokenizer/" + tokenizer_path)

        # 获得模型配置-------------------------------------------------------------------
        if config is None:
            self.config = self.__read_args_for_config(args)
        else:
            self.config = config
        
        train_dataset = TextDataset(
            self.tokenizer, self.text_max_len - self.memory_num, "./dataset/" + self.dataset_name + "/train.jsonl")

        print(len(train_dataset))
        exit()
        
        # instance attribute
        self.model = None

    def train(self):
        model_save_name = self.model_save_prefix + \
            self.model_class + "_" + self.dataset_name
        # best model save path
        model_save_path = self.save_model_dict + "/" + model_save_name
        # last model save path
        last_model_save_path = self.last_model_dict + "/" + model_save_name

        # 创建模型，根据 model_class 的选择
        if self.model_class == 'BiEncoder':
            self.model = BiEncoder(config=self.config)
            
            self.model.to(self.device)
            self.model.train()

            # 优化器，调参用的
            parameters_dict_list = [
                # 这几个一样
                {'params': self.model.bert_model.parameters(), 'lr': 5e-5},
            ]
        elif self.model_class == 'InputMemory':
            self.model = InputMemory(config=self.config)
            
            self.model.to(self.device)
            self.model.train()
            
            parameters_dict_list = [
                # 这几个一样
                {'params': self.model.bert_model.parameters(), 'lr': 5e-5},
                # 这几个一样
                {'params': self.model.memory_for_question, 'lr': 5e-5},
                {'params': self.model.memory_for_answer, 'lr': 5e-5},
            ]
        else:
            raise Exception("No this model class")

        self.model.to(self.device)
        self.model.train()

        optimizer = torch.optim.Adam(parameters_dict_list, lr=5e-5)

        # 准备训练，最优模型先设置为无
        previous_best_mrr = -1

        early_stop_threshold = 5
        early_stop_count = 0

        # 读取数据
        train_dataset = TextDataset(
            self.tokenizer, self.text_max_len - self.memory_num, "./dataset/" + self.dataset_name + "/train.jsonl")

        print(len(train_dataset))
        exit()
        
        for epoch in range(0, self.num_train_epochs):

            train_sampler = RandomSampler(train_dataset)

            train_dataloader = DataLoader(train_dataset, sampler=train_sampler,
                                        batch_size=self.train_batch_size, num_workers=2, pin_memory=False)

            torch.cuda.empty_cache()

            # whether this epoch get the best model
            this_epoch_best = False

            # 打印一下
            print("*" * 20 + f" {epoch} " + "*" * 20)

            # 开始训练
            train_loss = 0.0
            # 计算训练集的R@1
            now_batch_num = 0

            self.model.train()
            # 获取scheduler
            if epoch == 0:
                t_total = (len(train_dataloader) //
                        self.gradient_accumulation_steps) * self.num_train_epochs

                scheduler = get_linear_schedule_with_warmup(optimizer,
                                                            num_warmup_steps=int(
                                                                t_total * 0.02),
                                                            num_training_steps=t_total)

                initial_mrr = self.do_val()
                print(f"initial_mrr = {initial_mrr}!")

            # 开始训练----------------------------------------------------------------------------------------
            # 进度条
            bar = tqdm(train_dataloader, total=len(train_dataloader))

            # 开始训练
            for batch in bar:
                step_loss = self.__train_step_for_bi(batch=batch, optimizer=optimizer,
                                                    now_batch_num=now_batch_num,
                                                    scheduler=scheduler)

                # 更新一下信息
                train_loss += step_loss.item()
                now_batch_num += 1

                bar.set_description(
                    "epoch {:>3d} loss {:.4f}".format(epoch + 1, train_loss / now_batch_num))

            gc.collect()

            this_best_mrr = self.do_val()

            print(f"This mrr {this_best_mrr}, previous_best_mrr {previous_best_mrr}")
            
            # 存储最优模型
            if this_best_mrr > previous_best_mrr:
                previous_best_mrr = this_best_mrr
                this_epoch_best = True

                self.save_model(model_save_path=model_save_path, epoch=epoch, optimizer=optimizer,
                                scheduler=scheduler,
                                previous_best_performance=this_best_mrr,
                                early_stop_count=early_stop_count)

            # 存储最新的模型
            self.save_model(model_save_path=last_model_save_path, epoch=epoch, optimizer=optimizer,
                            scheduler=scheduler, previous_best_performance=previous_best_mrr,
                            early_stop_count=early_stop_count)

            torch.cuda.empty_cache()
            gc.collect()

            # 是否早停
            if this_epoch_best:
                early_stop_count = 0
            else:
                early_stop_count += 1

                if early_stop_count == early_stop_threshold:
                    print("early stop!")
                    break

            sys.stdout.flush()

        # 用之前保存的最好的模型
        print("Training stop!!!!")
        self.model = self.load_models(self.model, model_save_path)

        final_test_result = self.do_test()
        print(
            "#" * 15 + f" This stage result is {final_test_result}. " + "#" * 15)

    def do_val(self):
        val_dataset = TextDataset(
            self.tokenizer, self.text_max_len - self.memory_num, "./dataset/" + self.dataset_name + "/valid.jsonl")

        val_sampler = SequentialSampler(val_dataset)

        val_dataloader = DataLoader(val_dataset, sampler=val_sampler,
                                    batch_size=self.train_batch_size, num_workers=2, pin_memory=False)

        print("--------------------- begin validation -----------------------")
        self.model.eval()

        code_vecs=[] 
        nl_vecs=[]

        for batch in val_dataloader:
            code_inputs = batch[0].to(self.device)    
            nl_inputs = batch[1].to(self.device)

            code_attention_mask = torch.ne(code_inputs, self.tokenizer.pad_token_id)
            
            nl_attention_mask = torch.ne(nl_inputs, self.tokenizer.pad_token_id)

            with torch.no_grad():
                q_embeddings, a_embeddings = self.model(q_input_ids=nl_inputs, q_attention_mask=nl_attention_mask,
                a_input_ids=code_inputs, a_attention_mask=code_attention_mask, return_vector=True)

                code_vecs.append(a_embeddings.cpu().numpy())
                nl_vecs.append(q_embeddings.cpu().numpy())
        
        code_vecs=np.concatenate(code_vecs,0)
        nl_vecs=np.concatenate(nl_vecs,0)

        scores=np.matmul(nl_vecs,code_vecs.T)
        ranks=[]
        for i in range(len(scores)):
            score=scores[i,i]
            rank=1
            for j in range(len(scores)):
                if i!=j and scores[i,j]>=score:
                    rank+=1
            ranks.append(1/rank)    
        
        self.model.train()
        return float(np.mean(ranks))

    def do_test(self):
        val_dataset = TextDataset(
            self.tokenizer, self.text_max_len - self.memory_num, "./dataset/" + self.dataset_name + "/test.jsonl")

        val_sampler = SequentialSampler(val_dataset)

        val_dataloader = DataLoader(val_dataset, sampler=val_sampler,
                                    batch_size=self.train_batch_size, num_workers=2, pin_memory=False)

        print("--------------------- begin test -----------------------")
        self.model.eval()

        code_vecs=[] 
        nl_vecs=[]

        for batch in val_dataloader:
            code_inputs = batch[0].to(self.device)    
            nl_inputs = batch[1].to(self.device)

            code_attention_mask = torch.ne(code_inputs, self.tokenizer.pad_token_id)
            
            nl_attention_mask = torch.ne(nl_inputs, self.tokenizer.pad_token_id)

            with torch.no_grad():
                q_embeddings, a_embeddings = self.model(q_input_ids=nl_inputs, q_attention_mask=nl_attention_mask,
                a_input_ids=code_inputs, a_attention_mask=code_attention_mask, return_vector=True)

                code_vecs.append(a_embeddings.cpu().numpy())
                nl_vecs.append(q_embeddings.cpu().numpy())
        
        code_vecs=np.concatenate(code_vecs,0)
        nl_vecs=np.concatenate(nl_vecs,0)

        scores=np.matmul(nl_vecs,code_vecs.T)
        ranks=[]
        for i in range(len(scores)):
            score=scores[i,i]
            rank=1
            for j in range(len(scores)):
                if i!=j and scores[i,j]>=score:
                    rank+=1
            
            if rank > 100:
                ranks.append(0)
            else:
                ranks.append(1/rank)    
        
        self.model.train()
        return float(np.mean(ranks))

    def save_model(self, model_save_path, optimizer, scheduler, epoch, previous_best_performance, early_stop_count,
                   postfix=""):
        self.model.eval()

        save_path = model_save_path + postfix

        # Only save the model it-self, maybe parallel
        model_to_save = self.model.module if hasattr(
            self.model, 'module') else self.model

        # 保存模型
        save_state = {'pretrained_bert_path': self.config.pretrained_bert_path,
                      'memory_num': self.memory_num,
                                  'model': model_to_save.state_dict(),
                                  'optimizer': optimizer.state_dict(),
                                  'scheduler': scheduler.state_dict(),
                                  'best performance': previous_best_performance,
                                  'early_stop_count': early_stop_count,
                                  'epoch': epoch + 1}

        torch.save(save_state, save_path)

        print("!" * 60)
        print(f"model is saved at {save_path}")
        print("!" * 60)

        self.model.train()

    @staticmethod
    def load_models(model, load_model_path):
        if load_model_path is None:
            print("you should offer model paths!")

        load_path = load_model_path
        model.load_state_dict(torch.load(load_path)['model'])

        print("model is loaded from", load_path)

        return model

    # 读取命令行传入的参数
    def __read_args_for_train(self, args):
        self.save_model_dict = args.save_model_dict
        self.last_model_dict = args.last_model_dict
        self.val_candidate_num = args.val_candidate_num
        self.val_batch_size = args.val_batch_size
        self.text_max_len = args.text_max_len
        self.dataset_name = args.dataset_name
        self.model_class = args.model_class
        self.memory_num = args.memory_num
        self.load_model_dict = args.load_model_dict
        self.train_batch_size = args.train_batch_size
        self.ranking_candidate_num = args.ranking_candidate_num
        self.model_save_prefix = args.model_save_prefix

        self.local_rank = args.local_rank
        self.data_parallel = args.data_parallel
        self.data_distribute = args.data_distribute

        self.num_train_epochs = args.num_train_epochs
        self.gradient_accumulation_steps = args.gradient_accumulation_steps
        self.no_initial_test = args.no_initial_test
        self.composition = args.composition
        self.restore_flag = args.restore

    # 读取命令行传入的有关config的参数
    def __read_args_for_config(self, args):
        if self.model_class == 'InputMemory':
            config = InputMemoryConfig(len(self.tokenizer),
                                    pretrained_bert_path=args.pretrained_bert_path,
                                    num_labels=args.label_num,
                                    word_embedding_len=768,
                                    sentence_embedding_len=768,
                                    memory_num=args.memory_num,
                                    composition=self.composition)
        elif self.model_class == 'BiEncoder':
            config = BiConfig(len(self.tokenizer),
                            pretrained_bert_path=args.pretrained_bert_path,
                            num_labels=args.label_num,
                            word_embedding_len=768,
                            sentence_embedding_len=768,
                            composition=self.composition)
        else:
            raise Exception("No this model class")

        return config

    # 双塔模型的训练步
    def __train_step_for_bi(self, batch, optimizer, now_batch_num, scheduler, **kwargs):
        code_inputs = batch[0].to(self.device)    
        code_attention_mask = torch.ne(code_inputs, self.tokenizer.pad_token_id)
        
        nl_inputs = batch[1].to(self.device)
        nl_attention_mask = torch.ne(nl_inputs, self.tokenizer.pad_token_id)

        logits = self.model(q_input_ids=nl_inputs, q_attention_mask=nl_attention_mask,
                a_input_ids=code_inputs, a_attention_mask=code_attention_mask)
        
        # 计算损失
        mask = torch.eye(logits.size(0)).to(logits.device)
        loss = F.log_softmax(logits, dim=-1) * mask
        step_loss = (-loss.sum(dim=1)).mean()
        
        # 误差反向传播
        step_loss.backward()

        # 更新模型参数
        if (now_batch_num + 1) % self.gradient_accumulation_steps == 0:
            optimizer.step()
            optimizer.zero_grad()
            scheduler.step()

        return step_loss

    @staticmethod
    def hits_count(candidate_ranks, k):
        count = 0
        for rank in candidate_ranks:
            if rank <= k:
                count += 1
        return count / (len(candidate_ranks) + 1e-8)

    @staticmethod
    def dcg_score(candidate_ranks, k):
        score = 0
        for rank in candidate_ranks:
            if rank <= k:
                score += 1 / np.log2(1 + rank)
        return score / (len(candidate_ranks) + 1e-8)
