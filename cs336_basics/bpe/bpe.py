import regex as re
import time
import itertools
import os
from typing import BinaryIO
from collections import Counter
import multiprocessing


def timeit(func):
    def call_func(*args, **kwargs):
        s = time.perf_counter_ns()
        ret = func(*args, **kwargs)
        e = time.perf_counter_ns()
        print(f"{func.__name__} took {int((e - s) / 10**6)} milli seconds")
        return ret

    return call_func


def _pretokenize_chunk(input_path: str, special_tokens: list[str], chunk_boundary: tuple[int, int] | None = None):
    with open(input_path, "rb") as f:
        if chunk_boundary is not None:
            start, end = chunk_boundary
            f.seek(start)
            chunk = f.read(end - start).decode("utf-8", errors="ignore")
        else:
            chunk = f.read().decode("utf-8", errors="ignore")

    PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
    frequency_table: dict[tuple[bytes, ...], int] = {}
    split_pattern = "|".join(re.escape(k) for k in sorted(special_tokens, key=len, reverse=True))
    split_texts = re.split(split_pattern, chunk)

    for text in split_texts:
        for token in re.finditer(PAT, text):
            bytes_tuple = tuple(bytes([b]) for b in bytes(token.group(), encoding="utf-8"))
            frequency_table[bytes_tuple] = frequency_table.get(bytes_tuple, 0) + 1
    return frequency_table


def _pretokenize_example(input_path: str, special_tokens: list[str]) -> dict[tuple[bytes, ...], int]:
    frequency_table = _pretokenize_chunk(input_path, special_tokens)
    return frequency_table


def _find_chunk_boundaries(
    file: BinaryIO,
    desired_num_chunks: int,
    split_special_token: bytes,
) -> list[int]:
    """
    Chunk the file into parts that can be counted independently.
    May return fewer chunks if the boundaries end up overlapping.
    """
    assert isinstance(split_special_token, bytes), "Must represent special token as a bytestring"

    # Get total file size in bytes
    file.seek(0, os.SEEK_END)
    file_size = file.tell()
    file.seek(0)

    chunk_size = file_size // desired_num_chunks

    # Initial guesses for chunk boundary locations, uniformly spaced
    # Chunks start on previous index, don't include last index
    chunk_boundaries = [i * chunk_size for i in range(desired_num_chunks + 1)]
    chunk_boundaries[-1] = file_size

    mini_chunk_size = 4096  # Read ahead by 4k bytes at a time

    for bi in range(1, len(chunk_boundaries) - 1):
        initial_position = chunk_boundaries[bi]
        file.seek(initial_position)  # Start at boundary guess
        while True:
            mini_chunk = file.read(mini_chunk_size)  # Read a mini chunk

            # If EOF, this boundary should be at the end of the file
            if mini_chunk == b"":
                chunk_boundaries[bi] = file_size
                break

            # Find the special token in the mini chunk
            found_at = mini_chunk.find(split_special_token)
            if found_at != -1:
                chunk_boundaries[bi] = initial_position + found_at
                break
            initial_position += mini_chunk_size

    # Make sure all boundaries are unique, but might be fewer than desired_num_chunks
    return sorted(set(chunk_boundaries))


def _pretokenize_mp(input_path: str, special_tokens: list[str]) -> dict[tuple[bytes, ...], int]:
    frequency_table: dict[tuple[bytes, ...], int] = {}
    with open(input_path, "rb") as f:
        num_processes = 6
        boundaries = _find_chunk_boundaries(f, num_processes, b"<|endoftext|>")

        # The following is a serial implementation, but you can parallelize this
        # by sending each start/end pair to a set of processes.
        with multiprocessing.Pool(processes=num_processes) as pool:
            frequency_tables = pool.starmap(
                _pretokenize_chunk,
                [(input_path, special_tokens, (start, end)) for start, end in itertools.pairwise(boundaries)],
            )

    d = Counter()
    for ft in frequency_tables:
        d.update(ft)
    frequency_table = dict(d)
    return frequency_table


# @timeit
def pretokenize(input_path: str, special_tokens: list[str]) -> dict[tuple[bytes, ...], int]:
    # return _pretokenize_example(input_path, special_tokens)
    return _pretokenize_mp(input_path, special_tokens)


def compute_pair_freq(
    frequency_table: dict[tuple[bytes, ...], int],
) -> dict[tuple[bytes, bytes], tuple[int, set[tuple[bytes, ...]]]]:
    pair_freq: dict[tuple[bytes, bytes], tuple[int, set[tuple[bytes, ...]]]] = {}

    for token_bytes, freq in frequency_table.items():
        for b1, b2 in itertools.pairwise(token_bytes):
            f, t = pair_freq.get((b1, b2), (0, set()))
            f = f + freq
            t.add(token_bytes)
            pair_freq[(b1, b2)] = (f, t)
    return pair_freq


# @timeit
def compute_bpe_merge(
    frequency_table: dict[tuple[bytes, ...], int],
    vocab: dict[int, bytes],
    pair_freq: dict[tuple[bytes, bytes], tuple[int, set[tuple[bytes, ...]]]],
) -> tuple[bytes, bytes]:

    pair = max(pair_freq.items(), key=lambda x: (x[1][0], x[0]))[0]

    to_merge_token_bytes = list(pair_freq[pair][1])
    for token_bytes in to_merge_token_bytes:
        # merge the pair
        new_token_bytes = []
        idx = 0
        while idx < len(token_bytes):
            if idx < len(token_bytes) - 1 and (token_bytes[idx], token_bytes[idx + 1]) == pair:
                new_token_bytes.append(pair[0] + pair[1])
                idx += 2
                continue
            new_token_bytes.append(token_bytes[idx])
            idx += 1

        new_token_bytes = tuple(new_token_bytes)

        # update the frequency table with the new merged token_bytes
        freq = frequency_table[token_bytes]
        del frequency_table[token_bytes]
        frequency_table[new_token_bytes] = freq

        # update the pair frequency
        for b1, b2 in itertools.pairwise(token_bytes):
            f, t = pair_freq[(b1, b2)]
            f -= freq
            if token_bytes in t:
                t.remove(token_bytes)
            pair_freq[(b1, b2)] = (f, t)

        for b1, b2 in itertools.pairwise(new_token_bytes):
            f, t = pair_freq.get((b1, b2), (0, set()))
            f += freq
            t.add(new_token_bytes)
            pair_freq[(b1, b2)] = (f, t)

    vocab[len(vocab)] = pair[0] + pair[1]

    return pair


@timeit
def train_bpe(
    input_path: str, vocab_size: int, special_tokens: list[str], **kwargs
) -> tuple[dict[int, bytes], list[tuple[bytes, bytes]]]:

    vocab: dict[int, bytes] = {x: bytes([x]) for x in range(256)}
    merges: list[tuple[bytes, bytes]] = []

    for token in special_tokens:
        vocab[len(vocab)] = bytes(token, encoding="utf-8")

    # pre-tokenization
    frequency_table = pretokenize(input_path, special_tokens)

    # compute bpe merges
    pair_freq = compute_pair_freq(frequency_table)
    for _ in range(len(vocab), vocab_size):
        merge = compute_bpe_merge(frequency_table, vocab, pair_freq)
        merges.append(merge)
        print(f"Completed {len(vocab)} / {vocab_size}")

    return vocab, merges


def main():
    print("# 2.1 Unicode characters")
    print(f"{chr(0)=} - {chr(0)}-")
    print("repr =", repr(chr(0)))
    print(f"print ={chr(0)}=")
    print(f"This is a test{chr(0)}string")
    print(repr(f"This is a test{chr(0)}string"))
    print()

    print("# 2.2 Unicode encodings")
    test_string = "hello! こんにちは!"
    utf8_encoded = test_string.encode("utf-8")
    print(f"UTF encoding of '{test_string}' = '{utf8_encoded}'")
    print(f"{type(utf8_encoded)=}")
    print(f"{list(utf8_encoded)}")
    print(f"{len(test_string)=}")
    print(f"{len(utf8_encoded)=}")
    print(f"{utf8_encoded.decode("utf-8")=}")
    print()

    print("# Problem (unicode)")

    def decode_utf8_bytes_to_str_wrong(bytesstring: bytes):
        return "".join([bytes([b]).decode("utf-8") for b in bytesstring])

    print(f"{decode_utf8_bytes_to_str_wrong('hello'.encode('utf-8'))=}")

    try:
        print(f"{decode_utf8_bytes_to_str_wrong('こ'.encode('utf-8'))=}")
    except UnicodeDecodeError as e:
        print(f"'こ' cannot work with decode_utf8_bytes_to_str_wrong: {e}")

    try:
        invalid_unicode = b"\xc7\x17"
        print(f"{invalid_unicode.decode("utf-8")=}")
    except UnicodeDecodeError as e:
        print(f"Cannot decode byte sequence: {invalid_unicode}: {e}")
    print()

    print("# 2.4 BPE Tokenizer Training")
    PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
    pt = re.findall(PAT, "sine text that i'll pre-tokenize")
    print(f"Pretokens = {pt}")


    print("# Problem train_bpe_tinystories")
    vocab, merges = train_bpe("data/TinyStoriesV2-GPT4-train.txt", vocab_size=10_000, special_tokens=["<|endoftext|>"])
    breakpoint()

    print("Done")

if __name__ == "__main__":
    main()
