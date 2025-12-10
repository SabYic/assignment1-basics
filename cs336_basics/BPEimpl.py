from collections import defaultdict
from .pretokenization_example import find_chunk_boundaries
import re
from pathlib import Path

import os
from typing import BinaryIO
from collections import Counter
from concurrent.futures import ProcessPoolExecutor

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

    split_special = b"<|endoftext|>"
    word_freqs=defaultdict(int)
    counts = count_pretokens_parallel(
        input_path,
        2,
        split_special,
        special_tokens,
    )
    for token, freq in counts.items():
        word_freqs[repr(token)] = freq
    vocab = []

    for word in word_freqs.keys():
        for letter in word:
            if letter not in vocab:
                vocab.append(letter)
    vocab.sort()
   # print("Vocabulary:", vocab)
    splits = {word:[c for c in word] for word in word_freqs.keys()}
    merges=[]
    while len(vocab) < vocab_size:
        pair_freqs = get_stats(word_freqs)
        best_pair = ""
        max_freq = None
        for pair, freq in pair_freqs.items():
            if max_freq is None or max_freq < freq:
                best_pair = pair
                max_freq = freq
        splits = merge_vocab(*best_pair, splits,word_freqs)
        vocab.append(best_pair[0] + best_pair[1])
        merges.append(tuple([best_pair[0] , best_pair[1]]))
    return vocab, merges
    
        

def get_stats(vocab_items: defaultdict) -> dict[tuple[str, str], int]:
   pairs= defaultdict(int)
   for word,freq in vocab_items.items():
       symbols = list(word)
       for i in range(len(symbols) - 1):
           pair = (symbols[i], symbols[i + 1])
           pairs[pair] += freq
   return pairs
    
def merge_vocab(a: str, b: str, splits: list[str] ,word_freq: defaultdict[int]) -> list[str]:
    """
    Merge vocabulary tokens according to the BPE merges.

    Args:
        vocab: The initial vocabulary mapping indices to byte sequences.
        merges: The list of merges to apply.

    Returns:
        The updated vocabulary after applying the merges.
    """
    for word in word_freq.keys():
        split_word = splits[word]
        i=0
        while i < len(split_word) - 1:
            token1 = split_word[i]
            token2 = split_word[i + 1]
            if token1 == a and token2 == b:
                # Merge the tokens
                split_word[i] = token1 + token2
                del split_word[i + 1]
            else:
                i += 1
    return splits


def pre_tokenize(text: str, special_tokens: list[str]) -> list[str]:
    tokens = []
    i = 0
    n = len(text)

    # Sort special tokens by length to prefer longest match
    special_tokens = sorted(special_tokens, key=len, reverse=True)

    def is_punct(ch):
        return ("!" <= ch <= "/" or
                ":" <= ch <= "@" or
                "[" <= ch <= "`" or
                "{" <= ch <= "~")

    while i < n:
        # 1. Try to match a special token at position i
        matched = False
        for sp in special_tokens:
            if text.startswith(sp, i):
                tokens.append(sp)
                i += len(sp)
                matched = True
                break
        if matched:
            continue

        ch = text[i]

        # 2. Whitespace
        if ch.isspace():
            tokens.append(ch)
            i += 1
            continue

        # 3. Punctuation
        if is_punct(ch):
            tokens.append(ch)
            i += 1
            continue

        # 4. Normal token (accumulate)
        start = i
        i += 1
        while i < n:
            ch2 = text[i]
            if ch2.isspace() or is_punct(ch2):
                break

            # Also must break if a special token begins here
            if any(text.startswith(sp, i) for sp in special_tokens):
                break

            i += 1

        tokens.append(text[start:i])

    return tokens

def process_chunk(
    path: str,
    start: int,
    end: int,
    special_tokens: list[str],
) -> Counter:
    """
    Worker：在一个子进程里处理 [start, end) 这一段文件，
    做 pre-tokenization 并返回 Counter。
    """
    cnt: Counter = Counter()
    with open(path, "rb") as f:
        f.seek(start)
        data = f.read(end - start)
    text = data.decode("utf-8", errors="ignore")
    tokens = pre_tokenize(text, special_tokens)
    cnt.update(tokens)
    return cnt


def count_pretokens_parallel(
    path: str,
    num_processes: int,
    split_special_token: bytes,
    special_tokens: list[str],
) -> Counter:
    """
    整体入口：
    - 用 find_chunk_boundaries 切块
    - 多进程并行跑 pre-tokenizer + 计数
    - 合并所有 Counter
    """
    with open(path, "rb") as f:
        boundaries = find_chunk_boundaries(f, num_processes, split_special_token)

    # 生成 (start, end) 列表
    ranges = list(zip(boundaries[:-1], boundaries[1:]))

    total_counter: Counter = Counter()

    # 多进程并行
    with ProcessPoolExecutor(max_workers=num_processes) as ex:
        futures = [
            ex.submit(process_chunk, path, start, end, special_tokens)
            for start, end in ranges
        ]
        for fut in futures:
            total_counter.update(fut.result())

    return total_counter