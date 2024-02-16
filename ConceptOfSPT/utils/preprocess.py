# Copyright (c) 2019 NVIDIA CORPORATION and 2024 Shoya Wada. All rights reserved.
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

from .TextSharding import (
    Sharding,
    NLTKSegmenter,

)

import json
import random
import math
import multiprocessing
import subprocess
from pathlib import Path

import nltk

utilDir = Path(__file__).parent

def sharding(
        input_files: list[str] | str,
        output_file_prefix: str,
        output_dir: str,
        n_shards: int,
    ):
    if isinstance(input_files, str | Path):
        list_input_files =  [input_files]
    elif isinstance(input_files, list):
        list_input_files = input_files
    else:
        raise ValueError("input_file must be a string or a list")
    
    output_path = Path(output_dir) / Path(list_input_files[0]).stem
    if not output_path.exists():
        output_path.mkdir(parents=True)
    
    segmenter = NLTKSegmenter(nltk)
        
    # Split Sentences into Single sentence.
    sharding = Sharding(input_files=list_input_files,
                                     output_name_prefix=str(output_path / output_file_prefix),
                                     n_shards=n_shards)
    sharding.load_articles()
    sharding.segment_articles_into_sentences(segmenter)
    sharding.distribute_articles_over_shards()
    sharding.write_shards_to_disk()
    return

def create_training_instances(
        small_corpus_dir: str | Path,
        large_corpus_dir: str | Path,
        output_dir: str | Path,
        tokenizer_path: str | Path,
        filename_prefix: str,
        n_convoys: int,
        n_escorts: int,
        n_training_files: int,
        n_processes: int = multiprocessing.cpu_count(),
        random_seed: int = 12
    ):
    def create_record_worker(
            input_files: str | Path,
            filename_prefix: str,
            tokenizer_path: str,
            shard_id: int
        ):
        processor = str(utilDir / 'create_pretraining_data.py')
        bert_preprocessing_command = 'python ' + processor
        bert_preprocessing_command += ' --input_files=' + ','.join(input_files)
        bert_preprocessing_command += ' --output_file=' + str(output_dir / filename_prefix) + '_' + str(shard_id) + '.txt'
        bert_preprocessing_command += ' --tokenizer_path=' + str(tokenizer_path)
        bert_preprocessing_command += ' --random_seed=' + str(random_seed)

        bert_preprocessing_process = subprocess.Popen(bert_preprocessing_command, shell=True)

        last_process = bert_preprocessing_process

        # This could be better optimized (fine if all take equal time)
        if shard_id % n_processes == 0 and shard_id > 0:
            bert_preprocessing_process.wait()
        return last_process
    
    last_process = None

    convoys_dir = Path(small_corpus_dir)
    escorts_dir = Path(large_corpus_dir)
    output_dir = Path(output_dir)

    if not output_dir.exists():
        output_dir.mkdir(parents=True)
        
    # Convoy files.
    input_files_c = list(map(str, convoys_dir.glob('*.txt')))

    if not len(input_files_c):
        raise ValueError("{} is not a valid path".format(small_corpus_dir))

    # Escort files.
    input_files_e = list(map(str, escorts_dir.glob('*.txt')))

    if not len(input_files_e):
        raise ValueError("{} is not a valid path".format(large_corpus_dir))

    rng = random.Random(random_seed)
    data_shards = {
        'training_c': len(input_files_c),
        'training_e': len(input_files_e)
    }
    n_output_files = n_training_files

    composition = {}
    rng = random.Random(random_seed)
    input_files_c_order = []
    for _ in range(n_output_files // math.ceil(data_shards['training_c'] / n_convoys) + 1):
        input_files_c_order.extend(rng.sample(input_files_c, data_shards['training_c']))
    input_files_c_order = input_files_c_order[:n_output_files * n_convoys]

    input_files_e_order = []
    for _ in range(n_output_files // math.ceil(data_shards['training_e'] / n_escorts) + 1):
        input_files_e_order.extend(rng.sample(input_files_e, data_shards['training_e']))
    input_files_e_order = input_files_e_order[:n_output_files * n_escorts]

    for i in range(n_output_files):
        file_list = input_files_c_order[i * n_convoys:  (i + 1) * n_convoys] + \
                    input_files_e_order[i * n_escorts:  (i + 1) * n_escorts]
        composition[f'{filename_prefix}_{i}.txt'] = file_list
        last_process = create_record_worker(
                            input_files=file_list,
                            filename_prefix=filename_prefix,
                            tokenizer_path=tokenizer_path,
                            shard_id=i
                        )
    last_process.wait()

    with open(output_dir / 'composition.json', 'w', encoding='utf-8') as f:
        json.dump(composition, f, ensure_ascii=False, indent=2)
    return composition