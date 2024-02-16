# Copyright (c) 2024 Shoya Wada. All rights reserved.
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

from tokenizers import BertWordPieceTokenizer
from transformers import BertTokenizer, AutoConfig
from pathlib import Path
import json

def create_balanced_vocabulary(
        small_path: str,
        large_path: str,
        output_dir: str,
        vocab_size: int
    ):
    p_small = Path(small_path)
    p_large = Path(large_path)
    amplified_rate = round(p_large.stat().st_size / p_small.stat().st_size)
    convoy = [str(p_small)]
    escorts = [str(p_large)]
    
    filelist = escorts + convoy * amplified_rate
    tokenizer = BertWordPieceTokenizer(lowercase=True)

    output = Path(output_dir)

    if not output.exists():
        output.mkdir(parents=True, exist_ok=False)

    tokenizer.train(filelist, vocab_size=vocab_size)
    vocab = tokenizer.get_vocab()
    id2key = {v: k for k, v in vocab.items()}
    with open(output / 'vocab.txt', 'w') as writer:
        writer.write('\n'.join([id2key[i] for i in range(len(vocab))]))