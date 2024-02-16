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
