# coding=utf-8
# Copyright (c) 2019 NVIDIA CORPORATION and 2024 Shoya Wada. All rights reserved.
# Copyright 2018 The Google AI Language Team Authors and The HugginFace Inc. team.
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

import argparse
import random

import pandas as pd
from transformers import AutoTokenizer, PreTrainedTokenizer

import random

def create_training_instances(
    input_files: list[str],
    tokenizer: PreTrainedTokenizer,
    rng: random.Random
    ):
    """
    Create `TrainingInstance`s from raw text.
    # Input file format:
    (1) One sentence per line. These should ideally be actual sentences, not
        entire paragraphs or arbitrary spans of text. (Because we use the
        sentence boundaries for the "next sentence prediction" task).
    (2) Blank lines between documents. Document boundaries are needed so
        that the "next sentence prediction" task doesn't span between documents.
    """
    input_documents = input_files
    all_documents = [[]]
    for input_file in input_documents:
        with open(input_file, "r") as reader:
            while True:
                line = reader.readline()
                if not line:
                    break
                line = line.strip()

                # Empty lines are used as document delimiters
                if not line:
                    all_documents.append([])
                tokens = tokenizer.tokenize(line)
                if tokens:
                    all_documents[-1].append(line)

    # Remove empty documents
    all_documents = [x for x in all_documents if x]
    rng.shuffle(all_documents)

    instances = []
    for document_index in range(len(all_documents)):
        instances.extend(
            create_instances_from_document(
                all_documents=all_documents, document_index=document_index,
                tokenizer=tokenizer, rng=rng
            )
        )

    rng.shuffle(instances)
    return instances

def create_instances_from_document(
    all_documents: list[list[str]],
    document_index: int,
    tokenizer: PreTrainedTokenizer,
    rng: random.Random
    ):
    """Creates `TrainingInstance`s for a single document."""
    document = all_documents[document_index]

    instances = []
    current_chunk = []
    current_length = 0
    i = 0
    while i < len(document):
        segment = document[i]
        current_chunk.append(segment)
        current_length += len(segment)
        if i == len(document) - 1:
            if current_chunk:
                # `a_end` is how many segments from `current_chunk` go into the `A`
                # (first) sentence.
                a_end = 1
                if len(current_chunk) >= 2:
                    a_end = rng.randint(1, len(current_chunk) - 1)

                tokens_a = []
                for j in range(a_end):
                    tokens_a.append(current_chunk[j])

                tokens_b = []
                # Random next
                is_random_next = False
                if len(current_chunk) == 1 or rng.random() < 0.5:
                    is_random_next = True

                    # This should rarely go for more than one iteration for large
                    # corpora. However, just to be careful, we try to make sure that
                    # the random document is not the same as the document
                    # we're processing.
                    for _ in range(10):
                        random_document_index = rng.randint(0, len(all_documents) - 1)
                        if random_document_index != document_index:
                            break

                    random_document = all_documents[random_document_index]
                    random_start = rng.randint(0, len(random_document) - 1)
                    for j in range(random_start, len(random_document)):
                        tokens_b.append(random_document[j])
                    # We didn't actually use these segments so we "put them back" so
                    # they don't go to waste.
                    num_unused_segments = len(current_chunk) - a_end
                    i -= num_unused_segments
                # Actual next
                else:
                    is_random_next = False
                    for j in range(a_end, len(current_chunk)):
                        tokens_b.append(current_chunk[j])

                tokens = ['[CLS]'] + tokens_a + ['[SEP]'] + tokens_b + ['[SEP]']
                instance = {
                   'is_random_next': is_random_next,
                   'input_tokens': ' '.join(tokens)
                }
                instances.append(instance)
            current_chunk = []
            current_length = 0
        i += 1

    return instances

def main():

    parser = argparse.ArgumentParser()
    ## Required parameters
    parser.add_argument("--tokenizer_path",
                        default=None,
                        type=str,
                        required=True)
    parser.add_argument("--input_files",
                        default=None,
                        type=str,
                        required=True,
                        help="Specify the input files in a comma-separated list (no spaces)")
    parser.add_argument("--output_file",
                        default=None,
                        type=str,
                        required=True,
                        help="The output file where the model checkpoints will be written.")
    parser.add_argument('--random_seed',
                        type=int,
                        default=12345,
                        help="random seed for initialization")

    args = parser.parse_args()
    
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_path)

    input_files = args.input_files.split(',')
    if not len(input_files):
        raise ValueError("{} is not a valid path".format(args.input_files))

    rng = random.Random(args.random_seed)
    instances = create_training_instances(
                input_files=input_files,
                tokenizer=tokenizer,
                rng=rng)

    output_file = args.output_file
    pd.DataFrame(instances).to_csv(output_file, sep='\t', index=False)
    print(f'Number of instances: {len(instances)}. Save to {output_file}')

if __name__ == "__main__":
    main()
