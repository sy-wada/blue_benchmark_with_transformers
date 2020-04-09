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
""" Named entity recognition fine-tuning: utilities to work with CoNLL-2003 task. """


import logging
import os

from .tokenization import convert_to_unicode


logger = logging.getLogger(__name__)


class InputExample(object):
    """A single training/test example for token classification."""

    def __init__(self, guid, words, labels):
        """Constructs a InputExample.

        Args:
            guid: Unique id for the example.
            words: list. The words of the sequence.
            labels: (Optional) list. The labels for each word of the sequence. This should be
            specified for train and dev examples, but not for test examples.
        """
        self.guid = guid
        self.words = words
        self.labels = labels


class InputFeatures(object):
    """A single set of features of data."""

    def __init__(self, input_ids, input_mask, segment_ids, label_ids):
        self.input_ids = input_ids
        self.input_mask = input_mask
        self.segment_ids = segment_ids
        self.label_ids = label_ids
        
class DataProcessor(object):
    """Base class for data converters for sequence classification data sets."""

    def get_train_examples(self, data_dir):
        """Gets a collection of `InputExample`s for the train set."""
        raise NotImplementedError()

    def get_dev_examples(self, data_dir):
        """Gets a collection of `InputExample`s for the dev set."""
        raise NotImplementedError()
        
    def get_test_examples(self, data_dir):
        """Gets a collection of `InputExample`s for the test set."""
        raise NotImplementedError()
        
    def get_labels(self):
        """Gets the list of labels for this data set."""
        raise NotImplementedError()


    @classmethod
    def _read_data(cls, input_file):
        """Reads a BIO data.
        Same as the original: https://github.com/ncbi-nlp/bluebert/blob/master/bluebert/run_bluebert_ner.py
        """
        with open(input_file, "r") as f:
            lines = []
            words = []
            labels = []
            for line in f:
                contends = line.strip()
                if len(contends) == 0:
                    assert len(words) == len(labels)
                    if len(words) > 30:
                        # split if the sentence is longer than 30
                        while len(words) > 30:
                            tmplabel = labels[:30]
                            for iidx in range(len(tmplabel)):
                                if tmplabel.pop() == 'O':
                                    break
                            l = ' '.join(
                                [label for label in labels[:len(tmplabel) + 1] if len(label) > 0])
                            w = ' '.join(
                                [word for word in words[:len(tmplabel) + 1] if len(word) > 0])
                            lines.append([l, w])
                            words = words[len(tmplabel) + 1:]
                            labels = labels[len(tmplabel) + 1:]

                    if len(words) == 0:
                        continue
                    l = ' '.join([label for label in labels if len(label) > 0])
                    w = ' '.join([word for word in words if len(word) > 0])
                    lines.append([l, w])
                    words = []
                    labels = []
                    continue

                word = line.strip().split()[0]
                label = line.strip().split()[-1]
                words.append(word)
                labels.append(label)
            return lines


class BC5CDRProcessor(DataProcessor):
    def get_train_examples(self, data_dir):
        l1 = self._read_data(os.path.join(data_dir, "train.tsv"))
        l2 = self._read_data(os.path.join(data_dir, "devel.tsv"))
        return self._create_example(l1 + l2, "train")

    def get_dev_examples(self, data_dir):
        return self._create_example(
            self._read_data(os.path.join(data_dir, "devel.tsv")), "dev"
        )

    def get_test_examples(self, data_dir):
        return self._create_example(
            self._read_data(os.path.join(data_dir, "test.tsv")), "test")

    def get_labels(self):
        # remove "X", "[CLS]", "[SEP]".
        return ["B", "I", "O"]

    def _create_example(self, lines, set_type):
        examples = []
        for (i, line) in enumerate(lines):
            guid = "%s-%s" % (set_type, i)
            text = convert_to_unicode(line[1])
            label = convert_to_unicode(line[0])
            # ADD:
            # To match the huggingface implimentation, convert str to list.
            text = text.split(' ')
            label = label.split(' ')
            examples.append(InputExample(guid=guid, words=text, labels=label))
        return examples

class CLEFEProcessor(DataProcessor):
    def get_train_examples(self, data_dir):
        lines1 = self._read_data2(os.path.join(data_dir, "Training.tsv"))
        lines2 = self._read_data2(os.path.join(data_dir, "Development.tsv"))
        return self._create_example(
            lines1 + lines2, "train"
        )

    def get_dev_examples(self, data_dir):
        return self._create_example(
            self._read_data2(os.path.join(data_dir, "Development.tsv")), "dev"
        )

    def get_test_examples(self, data_dir):
        return self._create_example(
            self._read_data2(os.path.join(data_dir, "Test.tsv")), "test")

    def get_labels(self):
        # remove "X", "[CLS]", "[SEP]".
        return ["B", "I", "O"]
    
    def _create_example(self, lines, set_type):
        examples = []
        for (i, line) in enumerate(lines):
            guid = "%s-%s" % (set_type, i)
            text = convert_to_unicode(line[1])
            label = convert_to_unicode(line[0])
            # ADD:
            # To fit the huggingface implimentation, convert str to list.
            text = text.split(' ')
            label = label.split(' ')
            examples.append(InputExample(guid=guid, words=text, labels=label))
        return examples

    @classmethod
    def _read_data2(cls, input_file):
        """Reads a BIO data.
        Almost same as the original: https://github.com/ncbi-nlp/bluebert/blob/master/bluebert/run_bluebert_ner.py
        """
        with open(input_file, "r") as f:
            lines = []
            words = []
            labels = []
            for line in f:
                contends = line.strip()
                if len(contends) == 0:
                    assert len(words) == len(labels)
                    if len(words) > 30:
                        # split if the sentence is longer than 30
                        # quote from _read_?data1() in DataProcessor
                        while len(words) > 30:
                            tmplabel = labels[:30]
                            for iidx in range(len(tmplabel)):
                                if tmplabel.pop() == 'O':
                                    break
                            l = ' '.join(
                                [label for label in labels[:len(tmplabel) + 1] if len(label) > 0])
                            w = ' '.join(
                                [word for word in words[:len(tmplabel) + 1] if len(word) > 0])
                            lines.append([l, w])
                            words = words[len(tmplabel) + 1:]
                            labels = labels[len(tmplabel) + 1:]
                            
                    if len(words) == 0:
                        continue
                    l = ' '.join([label for label in labels if len(label) > 0])
                    w = ' '.join([word for word in words if len(word) > 0])
                    lines.append([l, w])
                    words = []
                    labels = []
                    continue
                elif contends.startswith('###'):
                    continue

                word = line.strip().split()[0]
                label = line.strip().split()[-1]
                words.append(word)
                labels.append(label)
            return lines

def convert_examples_to_features(
    examples,
    label_list,
    max_seq_length,
    tokenizer,
    cls_token_at_end=False,
    cls_token="[CLS]",
    cls_token_segment_id=1,
    sep_token="[SEP]",
    sep_token_extra=False,
    pad_on_left=False,
    pad_token=0,
    pad_token_segment_id=0,
    pad_token_label_id=-100,
    sequence_a_segment_id=0,
    mask_padding_with_zero=True,
):
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
            logger.info("Writing example %d of %d", ex_index, len(examples))

        tokens = []
        label_ids = []
        for word, label in zip(example.words, example.labels):
            word_tokens = tokenizer.tokenize(word)
            tokens.extend(word_tokens)
            # Use the real label id for the first token of the word, and padding ids for the remaining tokens
            label_ids.extend([label_map[label]] + [pad_token_label_id] * (len(word_tokens) - 1))

        # Account for [CLS] and [SEP] with "- 2" and with "- 3" for RoBERTa.
        special_tokens_count = 3 if sep_token_extra else 2
        if len(tokens) > max_seq_length - special_tokens_count:
            tokens = tokens[: (max_seq_length - special_tokens_count)]
            label_ids = label_ids[: (max_seq_length - special_tokens_count)]

        # The convention in BERT is:
        # (a) For sequence pairs:
        #  tokens:   [CLS] is this jack ##son ##ville ? [SEP] no it is not . [SEP]
        #  type_ids:   0   0  0    0    0     0       0   0   1  1  1  1   1   1
        # (b) For single sequences:
        #  tokens:   [CLS] the dog is hairy . [SEP]
        #  type_ids:   0   0   0   0  0     0   0
        #
        # Where "type_ids" are used to indicate whether this is the first
        # sequence or the second sequence. The embedding vectors for `type=0` and
        # `type=1` were learned during pre-training and are added to the wordpiece
        # embedding vector (and position vector). This is not *strictly* necessary
        # since the [SEP] token unambiguously separates the sequences, but it makes
        # it easier for the model to learn the concept of sequences.
        #
        # For classification tasks, the first vector (corresponding to [CLS]) is
        # used as as the "sentence vector". Note that this only makes sense because
        # the entire model is fine-tuned.
        tokens += [sep_token]
        label_ids += [pad_token_label_id]
        if sep_token_extra:
            # roberta uses an extra separator b/w pairs of sentences
            tokens += [sep_token]
            label_ids += [pad_token_label_id]
        segment_ids = [sequence_a_segment_id] * len(tokens)

        if cls_token_at_end:
            tokens += [cls_token]
            label_ids += [pad_token_label_id]
            segment_ids += [cls_token_segment_id]
        else:
            tokens = [cls_token] + tokens
            label_ids = [pad_token_label_id] + label_ids
            segment_ids = [cls_token_segment_id] + segment_ids

        input_ids = tokenizer.convert_tokens_to_ids(tokens)

        # The mask has 1 for real tokens and 0 for padding tokens. Only real
        # tokens are attended to.
        input_mask = [1 if mask_padding_with_zero else 0] * len(input_ids)

        # Zero-pad up to the sequence length.
        padding_length = max_seq_length - len(input_ids)
        if pad_on_left:
            input_ids = ([pad_token] * padding_length) + input_ids
            input_mask = ([0 if mask_padding_with_zero else 1] * padding_length) + input_mask
            segment_ids = ([pad_token_segment_id] * padding_length) + segment_ids
            label_ids = ([pad_token_label_id] * padding_length) + label_ids
        else:
            input_ids += [pad_token] * padding_length
            input_mask += [0 if mask_padding_with_zero else 1] * padding_length
            segment_ids += [pad_token_segment_id] * padding_length
            label_ids += [pad_token_label_id] * padding_length

        assert len(input_ids) == max_seq_length
        assert len(input_mask) == max_seq_length
        assert len(segment_ids) == max_seq_length
        assert len(label_ids) == max_seq_length

        if ex_index < 5:
            logger.info("*** Example ***")
            logger.info("guid: %s", example.guid)
            logger.info("tokens: %s", " ".join([str(x) for x in tokens]))
            logger.info("input_ids: %s", " ".join([str(x) for x in input_ids]))
            logger.info("input_mask: %s", " ".join([str(x) for x in input_mask]))
            logger.info("segment_ids: %s", " ".join([str(x) for x in segment_ids]))
            logger.info("label_ids: %s", " ".join([str(x) for x in label_ids]))

        features.append(
            InputFeatures(input_ids=input_ids, input_mask=input_mask, segment_ids=segment_ids, label_ids=label_ids)
        )
    return features