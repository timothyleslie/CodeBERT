# Code Search

## Data Preprocess

Both training and validation datasets are created in a way that positive and negative samples are balanced. Negative samples consist of balanced number of instances with randomly replaced NL and PL.

We follow the official evaluation metric to calculate the Mean Reciprocal Rank (MRR) for each pair of test data (c, w) over a fixed set of 999 distractor codes.

You can use the following command to download the preprocessed training and validation dataset and preprocess the test dataset by yourself. The preprocessed testing dataset is very large, so only the preprocessing script is provided.

```shell
mkdir data data/codesearch
cd data/codesearch
gdown https://drive.google.com/uc?id=1xgSR34XO8xXZg4cZScDYj2eGerBE9iGo  
unzip codesearch_data.zip
rm  codesearch_data.zip
cd ../../codesearch
python process_data.py
cd ..
```

## Fine-Tune
We fine-tuned the model on 2*P100 GPUs. 
```shell
cd codesearch

lang=php #fine-tuning a language-specific model for each programming language 
pretrained_model=microsoft/codebert-base  #Roberta: roberta-base
prompt_type=fine-tune
python run_classifier.py \
--prompt_type $prompt_type \
--model_type roberta \
--task_name codesearch \
--do_train \
--do_eval \
--eval_all_checkpoints \
--train_file train.txt \
--dev_file valid.txt \
--max_seq_length 200 \
--per_gpu_train_batch_size 64 \
--per_gpu_eval_batch_size 64 \
--learning_rate 1e-5 \
--num_train_epochs 16 \
--gradient_accumulation_steps 1 \
--overwrite_output_dir \
--data_dir ../data/codesearch/train_valid/$lang \
--output_dir ./models/$prompt_type/$lang  \
--model_name_or_path $pretrained_model
```
## Inference and Evaluation

Inference
```shell
lang=php #programming language
idx=0 #test batch idx
prompt_type=fine-tune
python run_classifier.py \
--prompt_type $prompt_type \
--model_type roberta \
--model_name_or_path microsoft/codebert-base \
--task_name codesearch \
--do_predict \
--output_dir ./models/$prompt_type/$lang \
--data_dir ../data/codesearch/test/$lang \
--max_seq_length 200 \
--per_gpu_train_batch_size 32 \
--per_gpu_eval_batch_size 32 \
--learning_rate 1e-5 \
--num_train_epochs 16 \
--pred_model_dir ./models/$prompt_type/$lang/checkpoint-best/
# --test_file batch_${idx}.txt \
# --test_result_dir ./$prompt_type/results/$lang/${idx}_batch_result.txt
```

Evaluation
```shell
prompt_type=fine-tune
python mrr.py \
--prompt_type $prompt_type
```


<!-- ################################Soft Prompt#################################### -->

lang=python
pretrained_model=microsoft/codebert-base
prompt_type=soft-prompt
python soft_prompt.py \
--lang $lang \
--prompt_type $prompt_type \
--model_type roberta \
--task_name codesearch \
--do_train \
--do_eval \
--eval_all_checkpoints \
--train_file train.txt \
--dev_file valid.txt \
--max_seq_length 200 \
--per_gpu_train_batch_size 64 \
--per_gpu_eval_batch_size 32 \
--learning_rate 1e-5 \
--num_train_epochs 16 \
--gradient_accumulation_steps 1 \
--overwrite_output_dir \
--data_dir ../data/codesearch/train_valid/$lang \
--output_dir ./models/$prompt_type/$lang  \
--model_name_or_path $pretrained_model


Inference
```shell
lang=python #programming language
prompt_type=soft-prompt
python soft_prompt.py \
--lang $lang \
--prompt_type $prompt_type \
--model_type roberta \
--model_name_or_path microsoft/codebert-base \
--task_name codesearch \
--do_predict \
--output_dir ./models/$prompt_type/$lang \
--data_dir ../data/codesearch/test/$lang \
--max_seq_length 200 \
--per_gpu_train_batch_size 64 \
--per_gpu_eval_batch_size 32 \
--learning_rate 1e-5 \
--num_train_epochs 16 \
--pred_model_dir ./models/$prompt_type/$lang/checkpoint-best/
# --test_file batch_${idx}.txt \
# --test_result_dir ./results/$prompt_type/$lang/${idx}_batch_result.txt
```

<!-- ################################CodeT5 Soft Prompt#################################### -->

lang=python
pretrained_model=Salesforce/codet5-small
prompt_type=t5small-soft-prompt
python t5_soft_prompt.py \
--lang $lang \
--prompt_type $prompt_type \
--model_type roberta \
--task_name codesearch \
--do_train \
--do_eval \
--eval_all_checkpoints \
--train_file train.txt \
--dev_file valid.txt \
--max_seq_length 200 \
--per_gpu_train_batch_size 64 \
--per_gpu_eval_batch_size 64 \
--learning_rate 1e-5 \
--num_train_epochs 16 \
--gradient_accumulation_steps 1 \
--overwrite_output_dir \
--data_dir ../data/codesearch/train_valid/$lang \
--output_dir ./models/$prompt_type/$lang  \
--model_name_or_path $pretrained_model


Inference
```shell
lang=python #programming language
prompt_type=t5small-soft-prompt
python t5_soft_prompt.py \
--lang $lang \
--prompt_type $prompt_type \
--model_type roberta \
--model_name_or_path Salesforce/codet5-small \
--task_name codesearch \
--do_predict \
--output_dir ./models/$prompt_type/$lang \
--data_dir ../data/codesearch/test/$lang \
--max_seq_length 200 \
--per_gpu_train_batch_size 64 \
--per_gpu_eval_batch_size 32 \
--learning_rate 1e-5 \
--num_train_epochs 16 \
--pred_model_dir ./models/$prompt_type/$lang/checkpoint-best/
# --test_file batch_${idx}.txt \
# --test_result_dir ./results/$prompt_type/$lang/${idx}_batch_result.txt
```


<!-- ################################ TEST #################################### -->
lang=python
pretrained_model=Salesforce/codet5-small
prompt_type=soft-prompt
python test.py \
--prompt_type $prompt_type \
--model_type roberta \
--task_name codesearch \
--do_train \
--do_eval \
--eval_all_checkpoints \
--train_file train.txt \
--dev_file valid.txt \
--max_seq_length 200 \
--per_gpu_train_batch_size 64 \
--per_gpu_eval_batch_size 64 \
--learning_rate 1e-5 \
--num_train_epochs 16 \
--gradient_accumulation_steps 1 \
--overwrite_output_dir \
--data_dir ../data/codesearch0/train_valid/$lang \
--output_dir ./models/$prompt_type/$lang  \
--model_name_or_path $pretrained_model
