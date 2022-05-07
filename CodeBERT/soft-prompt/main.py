import argparse
import os

import torch
from torch.backends import cudnn
import random
import numpy as np

from code_search import TrainWholeModel


def set_seed(seed):
	random.seed(seed)
	np.random.seed(seed)
	torch.manual_seed(seed)
	cudnn.deterministic = True
	cudnn.benchmark = False


def create_dir(this_args):
	if not os.path.exists(this_args.save_model_dict):
		os.makedirs(this_args.save_model_dict)
	if not os.path.exists(this_args.last_model_dict):
		os.makedirs(this_args.last_model_dict)
	if not os.path.exists("./tokenizer"):
		os.makedirs("./tokenizer")


def read_arguments():
	parser = argparse.ArgumentParser()

	# must set
	# add model
	parser.add_argument("--model_class", required=True,
						type=str, choices=['InputMemory', 'BiEncoder'])

	parser.add_argument("--dataset_name", "-d", required=True, type=str)
	parser.add_argument("--memory_num", "-m", default=50, type=int)
	parser.add_argument("--pretrained_bert_path",
						default='prajjwal1/bert-small', type=str)

	parser.add_argument("--nvidia_number", "-n", required=True, type=str)
	parser.add_argument("--one_stage", action="store_true", default=False)
	parser.add_argument("--model_save_prefix", default="", type=str)
	parser.add_argument("--val_batch_size", default=64, type=int)
	parser.add_argument("--train_batch_size", default=32, type=int)
	parser.add_argument("--gradient_accumulation_steps", type=int, default=1)
	parser.add_argument("--save_model_dict", default="./model/", type=str)
	parser.add_argument("--last_model_dict", default="./last_model/", type=str)

	# related to model
	parser.add_argument("--composition", type=str, default='pooler',
						help='control the way to get sentence representation')

	# related to train
	parser.add_argument("--restore", action="store_true", default=False,
						help="use restore and only_final together to control which model to read!")
	parser.add_argument("--no_initial_test", action="store_true", default=False)
	parser.add_argument("--load_model_dict", type=str)

	# 设置并行需要改的
	parser.add_argument('--local_rank', type=int, default=0,
						help='node rank for distributed training')
	parser.add_argument("--data_parallel", action="store_true", default=False)
	parser.add_argument("--data_distribute", action="store_true", default=False)

	# default arguments
	parser.add_argument("--seed", "-s", default=42, type=int)
	parser.add_argument("--text_max_len", default=512, type=int)
	parser.add_argument("--ranking_candidate_num", default=5, type=int)
	parser.add_argument("--label_num", default=2, type=int)  # !!!
	parser.add_argument("--num_train_epochs", "-e", type=int, default=50)

	parser.add_argument("--val_candidate_num", default=100,
						type=int, help="candidates num for ranking")
	parser.add_argument("--train_candidate_num", default=16,
						type=int, help="only need by cross")

	args = parser.parse_args()
	print("args:", args)
	return args


if __name__ == '__main__':
	my_args = read_arguments()

	# 创建路径
	create_dir(my_args)

	# 设置随机种子
	set_seed(my_args.seed)

	# 创建训练类
	my_train_model = TrainWholeModel(my_args)
	my_train_model.train()


# python main.py \
# --pretrained_bert_path  huggingface/CodeBERTa-small-v1 \
# --model_class InputMemory \
# --dataset_name ruby \
# --memory_num 50 \
# --nvidia_number 1 \
# --train_batch_size 32 
