import argparse
import glob
import logging
import os
import sys
import random


import numpy as np
import torch
from torch.utils.data import (DataLoader, RandomSampler, SequentialSampler,
                              TensorDataset)
from torch.utils.data.distributed import DistributedSampler
from tensorboardX import SummaryWriter
from tqdm import tqdm, trange
from st_utils import (compute_metrics, convert_examples_to_features,
                        output_modes, processors)
from transformers import (WEIGHTS_NAME, get_linear_schedule_with_warmup, AdamW,
                          RobertaConfig,
                          RobertaForSequenceClassification,
                          RobertaForMaskedLM,
                          RobertaTokenizer,
                          T5Config,
                          T5ForConditionalGeneration
                          )
from openprompt.plms import T5TokenizerWrapper

logger = logging.getLogger(__name__)
os.environ['CUDA_VISIBLE_DEVICES'] = '1'
MODEL_CLASSES = {'roberta': (RobertaConfig, RobertaForMaskedLM, RobertaTokenizer)}


parser = argparse.ArgumentParser()

## Required parameters
parser.add_argument("--prompt_type", default=None, type=str, required=True,
                        help="Type of prompt type")
parser.add_argument("--data_dir", default=None, type=str, required=True,
                    help="The input data dir. Should contain the .tsv files (or other data files) for the task.")
parser.add_argument("--model_type", default=None, type=str, required=True,
                    help="Model type selected in the list: " + ", ".join(MODEL_CLASSES.keys()))
parser.add_argument("--model_name_or_path", default=None, type=str, required=True,
                    help="Path to pre-trained model or shortcut name")
parser.add_argument("--task_name", default='codesearch', type=str, required=True,
                    help="The name of the task to train selected in the list: " + ", ".join(processors.keys()))
parser.add_argument("--output_dir", default=None, type=str, required=True,
                    help="The output directory where the model predictions and checkpoints will be written.")

## Other parameters
parser.add_argument("--config_name", default="", type=str,
                    help="Pretrained config name or path if not the same as model_name")
parser.add_argument("--tokenizer_name", default="", type=str,
                    help="Pretrained tokenizer name or path if not the same as model_name")
parser.add_argument("--cache_dir", default="", type=str,
                    help="Where do you want to store the pre-trained models downloaded from s3")
parser.add_argument("--max_seq_length", default=128, type=int,
                    help="The maximum total input sequence length after tokenization. Sequences longer "
                            "than this will be truncated, sequences shorter will be padded.")
parser.add_argument("--do_train", action='store_true',
                    help="Whether to run training.")
parser.add_argument("--do_eval", action='store_true',
                    help="Whether to run eval on the dev set.")
parser.add_argument("--do_predict", action='store_true',
                    help="Whether to run predict on the test set.")
parser.add_argument("--evaluate_during_training", action='store_true',
                    help="Rul evaluation during training at each logging step.")
parser.add_argument("--do_lower_case", action='store_true',
                    help="Set this flag if you are using an uncased model.")

parser.add_argument("--per_gpu_train_batch_size", default=8, type=int,
                    help="Batch size per GPU/CPU for training.")
parser.add_argument("--per_gpu_eval_batch_size", default=8, type=int,
                    help="Batch size per GPU/CPU for evaluation.")
parser.add_argument('--gradient_accumulation_steps', type=int, default=1,
                    help="Number of updates steps to accumulate before performing a backward/update pass.")
parser.add_argument("--learning_rate", default=5e-5, type=float,
                    help="The initial learning rate for Adam.")
parser.add_argument("--weight_decay", default=0.0, type=float,
                    help="Weight deay if we apply some.")
parser.add_argument("--adam_epsilon", default=1e-8, type=float,
                    help="Epsilon for Adam optimizer.")
parser.add_argument("--max_grad_norm", default=1.0, type=float,
                    help="Max gradient norm.")
parser.add_argument("--num_train_epochs", default=3.0, type=float,
                    help="Total number of training epochs to perform.")
parser.add_argument("--max_steps", default=-1, type=int,
                    help="If > 0: set total number of training steps to perform. Override num_train_epochs.")
parser.add_argument("--warmup_steps", default=0, type=int,
                    help="Linear warmup over warmup_steps.")

parser.add_argument('--logging_steps', type=int, default=50,
                    help="Log every X updates steps.")
parser.add_argument('--save_steps', type=int, default=50,
                    help="Save checkpoint every X updates steps.")
parser.add_argument("--eval_all_checkpoints", action='store_true',
                    help="Evaluate all checkpoints starting with the same prefix as model_name ending and ending with step number")
parser.add_argument("--no_cuda", action='store_true',
                    help="Avoid using CUDA when available")
parser.add_argument('--overwrite_output_dir', action='store_true',
                    help="Overwrite the content of the output directory")
parser.add_argument('--overwrite_cache', action='store_true',
                    help="Overwrite the cached training and evaluation sets")
parser.add_argument('--seed', type=int, default=42,
                    help="random seed for initialization")

parser.add_argument('--fp16', action='store_true',
                    help="Whether to use 16-bit (mixed) precision (through NVIDIA apex) instead of 32-bit")
parser.add_argument('--fp16_opt_level', type=str, default='O1',
                    help="For fp16: Apex AMP optimization level selected in ['O0', 'O1', 'O2', and 'O3']."
                            "See details at https://nvidia.github.io/apex/amp.html")
parser.add_argument("--local_rank", type=int, default=-1,
                    help="For distributed training: local_rank")
parser.add_argument('--server_ip', type=str, default='', help="For distant debugging.")
parser.add_argument('--server_port', type=str, default='', help="For distant debugging.")
parser.add_argument("--train_file", default="train_top10_concat.tsv", type=str,
                    help="train file")
parser.add_argument("--dev_file", default="shared_task_dev_top10_concat.tsv", type=str,
                    help="dev file")
parser.add_argument("--test_file", default="shared_task_dev_top10_concat.tsv", type=str,
                    help="test file")
parser.add_argument("--pred_model_dir", default=None, type=str,
                    help='model for prediction')
parser.add_argument("--test_result_dir", default='test_results.tsv', type=str,
                    help='path to store test result')
args = parser.parse_args()


def load_and_cache_examples(args, task, tokenizer, template, ttype='train'):
    processor = processors[task]()
    output_mode = output_modes[task]
    # Load data features from cache or dataset file
    if ttype == 'train':
        file_name = args.train_file.split('.')[0]
    elif ttype == 'dev':
        file_name = args.dev_file.split('.')[0]
    elif ttype == 'test':
        file_name = args.test_file.split('.')[0]
    cached_features_file = os.path.join(args.data_dir, args.prompt_type, 'cached_{}_{}_{}_{}_{}'.format(
        ttype,
        file_name,
        args.prompt_type,
        str(args.max_seq_length),
        str(task)))

    

    # if os.path.exists(cached_features_file):
    try:
        logger.info("Loading features from cached file %s", cached_features_file)
        features = torch.load(cached_features_file)
        if ttype == 'test':
            examples, instances = processor.get_test_examples(args.data_dir, args.test_file)
    except:
        cached_features_dir = os.path.join(args.data_dir, args.prompt_type)
        if os.path.exists(cached_features_dir):
            print("cache_features_dir exists")
        else:
            print("make cached_features_dir")
            os.makedirs(cached_features_dir)
        logger.info("Creating features from dataset file at %s", cached_features_dir)
        label_list = processor.get_labels()
        if ttype == 'train':
            examples = processor.get_train_examples(args.data_dir, args.train_file)
        elif ttype == 'dev':
            examples = processor.get_dev_examples(args.data_dir, args.dev_file)
        elif ttype == 'test':
            examples, instances = processor.get_test_examples(args.data_dir, args.test_file)

        features = convert_examples_to_features(examples, label_list, args.max_seq_length, tokenizer, output_mode, template,
                                                cls_token_at_end=bool(args.model_type in ['xlnet']),
                                                # xlnet has a cls token at the end
                                                cls_token=tokenizer.cls_token,
                                                sep_token=tokenizer.sep_token,
                                                cls_token_segment_id=2 if args.model_type in ['xlnet'] else 1,
                                                pad_on_left=bool(args.model_type in ['xlnet']),
                                                # pad on the left for xlnet
                                                pad_token_segment_id=4 if args.model_type in ['xlnet'] else 0)
        # if args.local_rank in [-1, 0]:
        #     logger.info("Saving features into cached file %s", cached_features_file)
        #     torch.save(features, cached_features_file)
    # Convert to Tensors and build dataset
    
    # for f in features:
    #     print(len(f.input_ids)) #200
    #     print(len(f.input_mask))#200
    #     print(len(f.segment_ids))#200
    #     print(len(f.label_id))#200
    all_input_ids = torch.tensor([f.input_ids for f in features], dtype=torch.long)
    all_input_mask = torch.tensor([f.input_mask for f in features], dtype=torch.long)
    all_loss_ids = torch.tensor([f.loss_ids for f in features], dtype=torch.long)
    if output_mode == "classification":
        all_label_ids = torch.tensor([f.label_id for f in features], dtype=torch.long)

    dataset = TensorDataset(all_input_ids, all_input_mask, all_loss_ids, all_label_ids)
    if (ttype == 'test'):
        return dataset, instances
    else:
        return dataset

import csv
import logging
import os
import sys
from io import open
from sklearn.metrics import f1_score

from openprompt.data_utils import InputExample
from openprompt.prompts import ManualTemplate, MixedTemplate, SoftTemplate
from openprompt import PromptDataLoader
from openprompt.prompts import ManualVerbalizer
from openprompt.plms import MLMTokenizerWrapper
from openprompt import PromptForClassification

classes = ["0", "1"]
csv.field_size_limit(sys.maxsize)
logger = logging.getLogger(__name__)
MODEL_CLASSES = {'roberta': (RobertaConfig, RobertaForSequenceClassification, RobertaTokenizer)}
loss_func = torch.nn.CrossEntropyLoss()

def main():
    args.device = "cuda"

    print("loading tokenizer")
    tokenizer = RobertaTokenizer.from_pretrained("./models/MLM_base/")
    processor = processors['codesearch']()

    
    myverbalizer = ManualVerbalizer(
        classes = classes,
        label_words = {
            "0": ["No"],
            "1": ["Yes"],
        },
        tokenizer = tokenizer,
    )

    # config = RobertaConfig.from_pretrained('models/T5-small/config.json')
    # model = RobertaForMaskedLM.from_pretrained('models/T5-small/pytorch_model.bin', config=config)
    # mytemplate = MixedTemplate(model=model, tokenizer=tokenizer, text=template_text)

    print("loading model")
    config = RobertaConfig.from_pretrained('models/MLM_base/config.json')
    model = RobertaForMaskedLM.from_pretrained('models/MLM_base/pytorch_model.bin', config=config)
    
    mytemplate = SoftTemplate(model=model, tokenizer=tokenizer,
                                text=' The code is: {"placeholder":"text_a"},{"mask"}',
                                num_tokens=50)

    p_model = PromptForClassification(plm=model, template=mytemplate, verbalizer=myverbalizer, freeze_plm=False)
    p_model.to(args.device)

    ############# try prompt model ###########################################################################
    train_dataset = load_and_cache_examples(args, args.task_name, tokenizer, mytemplate, ttype='train')
    
    train_dataloader = DataLoader(train_dataset,batch_size=64)
    
    no_decay = ['bias', 'LayerNorm.weight']
    # it's always good practice to set no decay to biase and LayerNorm parameters
    optimizer_grouped_parameters = [
        {'params': [p for n, p in p_model.named_parameters() if not any(nd in n for nd in no_decay)], 'weight_decay': 0.01},
        {'params': [p for n, p in p_model.named_parameters() if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
    ]
    optimizer = AdamW(optimizer_grouped_parameters, lr=1e-4)
    for epoch in range(10):
        tot_loss = 0
        for step, batch in enumerate(train_dataloader):
            batch = tuple(t.to(args.device) for t in batch)
            inputs = {'input_ids': batch[0],
                        'input_mask': batch[1],
                        'loss_ids': batch[2]}
            
            labels = batch[3]
            # print(inputs)

            # inputs_batch = {'input_ids': batch[0],
            #         'attention_mask': batch[1],
            #         'token_type_ids': batch[2] if args.model_type in ['bert', 'xlnet'] else None
            #         }
            logits = p_model(inputs)
            # print(labels)
            # print(logits)
            loss = loss_func(logits, labels)
            loss.backward()
            tot_loss += loss.item()
            optimizer.step()
            optimizer.zero_grad()
        print("Epoch {}, average loss: {}".format(epoch, tot_loss/(step+1)), flush=True)
        # print(outputs)
        # outputs = outputs.logits
        # print(inputs['loss_ids'].size()) # [64,200]
        # print(outputs.size()) # [64, 200, 50265]
        # # print(torch.where(inputs['loss_ids']>0))
        # # print(outputs)
        # outputs_at_mask = outputs[torch.where(inputs['loss_ids']>0)]
        # print(outputs_at_mask)
        # loss = loss_func(logits, labels)
        # print(loss)
    sys.exit()


    

main()