# coding=utf-8
# Copyright 2018 The Google AI Language Team Authors and The HuggingFace Inc. team.
# Copyright (c) 2018, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
""" BERT classification fine-tuning: utilities to work with GLUE tasks """

from __future__ import absolute_import, division, print_function

import csv
import logging
import os
import sys
from io import open
from sklearn.metrics import f1_score

csv.field_size_limit(sys.maxsize)
logger = logging.getLogger(__name__)




# template

class InputExample(object):
    """A single training/test example for simple sequence classification."""

    def __init__(self, guid, text, label=None):
        """Constructs a InputExample.

        Args:
            guid: Unique id for the example.
            text_a: string. The untokenized text of the first sequence. For single
            sequence tasks, only this sequence must be specified.
            text_b: (Optional) string. The untokenized text of the second sequence.
            Only must be specified for sequence pair tasks.
            label: (Optional) string. The label of the example. This should be
            specified for train and dev examples, but not for test examples.
        """
        self.guid = guid
        self.text = text
        self.label = label


class InputFeatures(object):
    """A single set of features of data."""

    def __init__(self, input_ids, input_mask, segment_ids, loss_ids, label_id):
        self.input_ids = input_ids
        self.input_mask = input_mask
        self.segment_ids = segment_ids
        self.loss_ids = loss_ids
        self.label_id = label_id


class DataProcessor(object):
    """Base class for data converters for sequence classification data sets."""

    def get_train_examples(self, data_dir):
        """Gets a collection of `InputExample`s for the train set."""
        raise NotImplementedError()

    def get_dev_examples(self, data_dir):
        """Gets a collection of `InputExample`s for the dev set."""
        raise NotImplementedError()

    def get_labels(self):
        """Gets the list of labels for this data set."""
        raise NotImplementedError()

    @classmethod
    def _read_tsv(cls, input_file, quotechar=None):
        """Reads a tab separated value file."""
        with open(input_file, "r", encoding='utf-8') as f:
            lines = []
            for line in f.readlines():
                line = line.strip().split('<CODESPLIT>')
                if len(line) != 5:
                    continue
                lines.append(line)
            return lines


class CodesearchProcessor(DataProcessor):
    """Processor for the MRPC data set (GLUE version)."""

    def get_train_examples(self, data_dir, train_file):
        """See base class."""
        logger.info("LOOKING AT {}".format(os.path.join(data_dir, train_file)))
        return self._create_examples(
            self._read_tsv(os.path.join(data_dir, train_file)), "train")

    def get_dev_examples(self, data_dir, dev_file):
        """See base class."""
        logger.info("LOOKING AT {}".format(os.path.join(data_dir, dev_file)))
        return self._create_examples(
            self._read_tsv(os.path.join(data_dir, dev_file)), "dev")

    def get_test_examples(self, data_dir, test_file):
        """See base class."""
        logger.info("LOOKING AT {}".format(os.path.join(data_dir, test_file)))
        return self._create_examples(
            self._read_tsv(os.path.join(data_dir, test_file)), "test")

    def get_labels(self):
        """See base class."""
        return ["0", "1"]

    def _create_examples(self, lines, set_type):
        """Creates examples for the training and dev sets."""
        examples = []
        for (i, line) in enumerate(lines):
            guid = "%s-%s" % (set_type, i)
            # text_a = line[3]
            # text_b = line[4]
            text = ""
            text.append("code: ")
            text.append(line[3])
            text.append(" query: ")
            text.append(line[4])
            print(text)
            sys.exit()
            if (set_type == 'test'):
                # label = self.get_labels()[0]
                label = "irrelevant"
            else:
                label = line[0]
                if label == "0":
                    label = "irrelevant"
                else:
                    label = "relevant"
            examples.append(
                InputExample(guid=guid, text=text, label=label))
        if (set_type == 'test'):
            return examples, lines
        else:
            return examples



def convert_examples_to_features(examples, label_list, max_seq_length,
                                 tokenizer, output_mode,
                                 cls_token_at_end=False, pad_on_left=False,
                                 cls_token='[CLS]', sep_token='[SEP]', pad_token=0,
                                 sequence_a_segment_id=0, sequence_b_segment_id=1,
                                 cls_token_segment_id=1, pad_token_segment_id=0,
                                 mask_padding_with_zero=True):
    """ Loads a data file into a list of `InputBatch`s
        `cls_token_at_end` define the location of the CLS token:
            - False (Default, BERT/XLM pattern): [CLS] + A + [SEP] + B + [SEP]
            - True (XLNet/GPT pattern): A + [SEP] + B + [SEP] + [CLS]
        `cls_token_segment_id` define the segment id associated to the CLS token (0 for BERT, 2 for XLNet)
    """
    
    label_map = {label: i for i, label in enumerate(label_list)}

    features = []
    for (ex_index, example) in enumerate(examples):
        if ex_index % 10000 == 0:
            logger.info("Writing example %d of %d" % (ex_index, len(examples)))

        codes.append(example.text)
        target.append(example.label)

    
    encoded_codes = tokenizer(
        codes, padding=True, verbose=False, add_special_tokens=True,
        truncation=True, max_length=args.max_source_length, return_tensors='pt')

    encoded_targets = tokenizer(
        target_nl, padding=True, verbose=False, add_special_tokens=True,
        truncation=True, max_length=args.max_source_length, return_tensors='pt')


        # assert len(tokenized_example['input_ids']) == max_seq_length
        # assert len(input_mask) == max_seq_length
        # assert len(segment_ids) == max_seq_length
        # assert len(tokenized_example['loss_ids']) == max_seq_length

        # if output_mode == "classification":
        #     label_id = label_map[example.label]
        # elif output_mode == "regression":
        #     label_id = float(example.label)
        # else:
        #     raise KeyError(output_mode)

        # if ex_index < 5:
        #     logger.info("*** Example ***")
        #     logger.info("guid: %s" % (example.guid))
        #     # logger.info("tokens: %s" % " ".join(
        #     #     [str(x) for x in tokens]))
        #     logger.info("input_ids: %s" % " ".join([str(x) for x in tokenized_example['input_ids']]))
        #     logger.info("input_mask: %s" % " ".join([str(x) for x in input_mask]))
        #     logger.info("segment_ids: %s" % " ".join([str(x) for x in segment_ids]))
        #     logger.info("loss_ids: %s" % " ".join([str(x) for x in tokenized_example['loss_ids']]))
        #     logger.info("label: %s (id = %d)" % (example.label, label_id))

        # features.append(
        #     InputFeatures(input_ids=tokenized_example['input_ids'],
        #                   input_mask=input_mask,
        #                   segment_ids=segment_ids,
        #                   loss_ids=tokenized_example['loss_ids'],
        #                   label_id=label_id))
    return {'source_ids':encoded_codes['input_ids'], 'target_ids':encoded_targets['input_ids'],
            'source_mask':encoded_codes['attention_mask'], 'target_mask':encoded_targets['attention_mask']}



def _truncate_seq_pair(tokens_a, tokens_b, max_length):
    """Truncates a sequence pair in place to the maximum length."""

    # This is a simple heuristic which will always truncate the longer sequence
    # one token at a time. This makes more sense than truncating an equal percent
    # of tokens from each, since if one sequence is very short then each token
    # that's truncated likely contains more information than a longer sequence.
    while True:
        total_length = len(tokens_a) + len(tokens_b)
        if total_length <= max_length:
            break
        if len(tokens_a) > len(tokens_b):
            tokens_a.pop()
        else:
            tokens_b.pop()


def simple_accuracy(preds, labels):
    return (preds == labels).mean()


def acc_and_f1(preds, labels):
    acc = simple_accuracy(preds, labels)
    f1 = f1_score(y_true=labels, y_pred=preds)
    return {
        "acc": acc,
        "f1": f1,
        "acc_and_f1": (acc + f1) / 2,
    }


def compute_metrics(task_name, preds, labels):
    assert len(preds) == len(labels)
    if task_name == "codesearch":
        return acc_and_f1(preds, labels)
    else:
        raise KeyError(task_name)


processors = {
    "codesearch": CodesearchProcessor,
}

output_modes = {
    "codesearch": "classification",
}

GLUE_TASKS_NUM_LABELS = {
    "codesearch": 2,
}



def glue(x, benchmark_name, label_names, feature_names=None, id_key='idx'):
  """Convert a dataset from glue to text2text examples.
  This function uses the feature names from the dataset to unpack examples into
  a format amenable for a text2text problem. For example, consider the Quora
  Question Pairs (QQP) benchmark, which would suggest
  benchmark_name="qqp"
  label_names=['not_duplicate', 'duplicate']
  For QQP, a typical example might look like
  {
      "question1": "Why do I easily get bored of my friends?",
      "question2": "Why do I get bored of friends so quickly?",
      "label": 1,
      "idx": 10,
  }
  This example would be transformed to
  {
       "inputs": (
           "qqp question1: Why do I easily get bored of my friends? question2: "
           "Why do I get bored of my friends so quickly?"
       ),
       "targets": "duplicate",
      "idx": 10,
  }
  Args:
    x: an example to process.
    benchmark_name: the name of the GLUE benchmark for this dataset.
    label_names: a list of label names corresponding to class index.
    feature_names: an optional ordered list of feature names. If provided,
      features will be ordered in this way in the output. If not provided, all
      features (except 'idx' and 'label') will be used, sorted by name.
    id_key: str, key for id in the dataset. If not provided, 'idx' will be used.
      if None, no id will be added to the dataset.
  Returns:
    A preprocessed example.
  """
  # If an ordering is not provided, sort feature keys to ensure a consistent
  # order.
  feature_keys = (
      feature_names or sorted(set(x.keys()).difference(['label', 'idx'])))
  # Pack keys (formatted as " key: ") and corresponding text feature
  strs_to_join = []
  for key in feature_keys:
    strs_to_join.append('{}:'.format(key))
    strs_to_join.append(x[key])
  # Add benchmark name at the start
  strs_to_join.insert(0, benchmark_name)
  label_name = tf.cond(
      # When no label is provided (label == -1), use "<unk>"
      tf.equal(x['label'], -1),
      lambda: tf.constant('<unk>'),
      # Otherwise grab the label text from label_names
      lambda: tf.gather(label_names, x['label']),
  )
  joined = tf.strings.join(strs_to_join, separator=' ')

  ex = {}

  if benchmark_name == 'multirc':
    # Remove HTML markup.
    joined = tf.strings.regex_replace(joined, '<br>', ' ')
    joined = tf.strings.regex_replace(joined, '<(/)?b>', '')

    # Store the data index in the returned example (used by eval)
    ex['idx/paragraph'] = x['idx']['paragraph']
    ex['idx/question'] = x['idx']['question']
    ex['idx/answer'] = x['idx']['answer']
  else:
    # Store the data index in the returned example (used by eval)
    if id_key:
      ex['idx'] = x[id_key]

  ex['inputs'] = joined
  ex['targets'] = label_name

  return ex