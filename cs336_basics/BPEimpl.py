from collections import defaultdict
from .pretokenization_example import find_chunk_boundaries
import re
from pathlib import Path

import os
from typing import BinaryIO
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed

def _run_bpe_trainning(
    input_path: Path,
    vocab_size: int,
    special_tokens: list[str] | None = None,
) -> tuple[dict[int, bytes], list[tuple[bytes, bytes]]]:
    """
    Train a BPE tokenizer on the given input text file.

    Args:
        input_path: Path to the input text file.
        vocab_size: Desired vocabulary size.
        special_tokens: List of special tokens to include in the vocabulary.

    Returns:
        A tuple containing:
            - A dictionary mapping vocabulary indices to byte sequences.
            - A list of merges, each represented as a tuple of byte sequences
              representing that <token1> was merged with <token2>.
              Merges are ordered by order of creation.
    """

    vocab=[]
    word_freqs=defaultdict(int)
    pair_freqs=defaultdict(int)
    process_num=4
    chunk_list=[]
    encode_txt=[]
    with open(input_path, "rb") as f:
        num_processes = 4
        boundaries = find_chunk_boundaries(f, num_processes, b"<|endoftext|>")

        # The following is a serial implementation, but you can parallelize this
        # by sending each start/end pair to a set of processes.
        for start, end in zip(boundaries[:-1], boundaries[1:]):
            f.seek(start)
            chunk = f.read(end - start).decode("utf-8", errors="ignore")
            chunk_list.append(chunk)
    vocab=dict[int, bytes]
    vocab={x: bytes([x]) for x in range(256)}
    with ThreadPoolExecutor(max_workers=8) as pool:
        futures = [pool.submit(encode, chunk_list[i]) for i in range(len(chunk_list))]
        for f in as_completed(futures):
            encode_txt.extend(f.result())

    merges=[]
    while len(vocab) < vocab_size:
        pair_freqs = get_stats(encode_txt)
       # print(pair_freqs)
        best_pair = None
        max_freq = None
        for pair, freq in pair_freqs.items():
            if max_freq is None or max_freq < freq:
                best_pair = pair
                max_freq = freq
        new_index=len(vocab)+256
        vocab[new_index]=vocab[best_pair[0]]+vocab[best_pair[1]]
        encode_txt = merge_vocab(*best_pair, encode_txt,new_index)
       # print(best_pair[0],best_pair[1])
        merges.append(tuple((vocab[best_pair[0]],vocab[best_pair[1]])))
    return vocab, merges
    
        

def encode(input_strings:str)->list[int]:
    indices = list(map(int,input_strings.encode("utf-8")))
    return indices

def get_stats(indice:list) -> dict[tuple[int, int], int]:
    pairs= defaultdict(int)
   
    for i in range(len(indice) - 2):
        pair = (indice[i], indice[i + 1])
        pairs[pair] += 1
    return pairs
    
def merge_vocab(a: int, b: int, splits: list[int], new_index:int ) -> list[str]:
    """
    Merge vocabulary tokens according to the BPE merges.

    Args:
        vocab: The initial vocabulary mapping indices to byte sequences.
        merges: The list of merges to apply.

    Returns:
        The updated vocabulary after applying the merges.
    """
    i = 0
    while i < len(splits) - 1:
        token1 = splits[i]
        token2 = splits[i + 1]

        if token1 == a and token2 == b:
            splits[i] = new_index
            del splits[i + 1]
            # ⚠️ 不 i += 1（因为新 token 可能继续 merge）
        else:
            i += 1
    return splits

