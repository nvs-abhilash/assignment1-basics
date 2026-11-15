import itertools
import json
from collections.abc import Iterable
import regex as re


class BPETokenizer:
    def __init__(self, vocab, merges, special_tokens=None):
        self.vocab: dict[int, bytes] = vocab
        self.merges: list[tuple[bytes, bytes]] = merges
        self.special_tokens = list(set(special_tokens)) if special_tokens else ["<|endoftext|>"]
        self.token2idx: dict[bytes, int] = {token: idx for idx, token in self.vocab.items()}
        self.split_pattern = f"({'|'.join(re.escape(k) for k in sorted(self.special_tokens, key=len, reverse=True))})"
        self.pattern = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""

    @classmethod
    def from_files(cls, vocab_filepath, merges_filepath, special_tokens=None):
        with open(vocab_filepath) as f:
            vocab = json.load(f)
            vocab = {int(idx): bytes.fromhex(token) for idx, token in vocab.items()}
        with open(merges_filepath) as f:
            merges = json.load(f)
            merges = [(bytes.fromhex(a), bytes.fromhex(b)) for a, b in merges]

        return cls(vocab, merges, special_tokens)

    def encode(self, text: str) -> list[int]:
        if not text or text == "":
            return []

        # step-1 pretokenize
        split_texts = re.split(self.split_pattern, text)
        encoded_array = []
        for st in split_texts:
            stb = st.encode("utf-8", errors="replace")
            if stb in self.token2idx:
                encoded_array.append(self.token2idx[stb])
                continue

            # Pre-tokenize
            cache = {}
            for token in re.finditer(self.pattern, st):
                pretoken = tuple(bytes([b]) for b in bytes(token.group(), encoding="utf-8"))

                # merge them in order
                if pretoken in cache:
                    encoded_array.extend(cache[pretoken])
                    continue
                p = pretoken
                for pair in self.merges:
                    new_pretoken = []
                    idx = 0
                    while idx < len(pretoken):
                        if idx < len(pretoken) - 1 and (pretoken[idx], pretoken[idx + 1]) == pair:
                            new_pretoken.append(pretoken[idx] + pretoken[idx + 1])
                            idx += 2
                        else:
                            new_pretoken.append(pretoken[idx])
                            idx += 1
                    pretoken = new_pretoken
                cache[p] = [self.token2idx[b] for b in tuple(pretoken)]
                encoded_array.extend(cache[p])
        return encoded_array

    def encode_iterable(self, iterable: Iterable[str]) -> Iterable[int]:
        if not iterable:
            yield from ()

        for text in iterable:
            encoding = self.encode(text)
            yield from encoding

    def decode(self, ids: list[int]) -> str:
        return b"".join([self.vocab[idx] for idx in ids]).decode("utf-8", errors="replace")


def main():
    tokenizer = BPETokenizer.from_files(
        "/Users/abhilash/Library/CloudStorage/OneDrive-GeorgiaInstituteofTechnology/learn/cs336/assignments/assignment1-basics/data/bpe/tiny_stories_v2_gpt4_train_vocab.json",
        "/Users/abhilash/Library/CloudStorage/OneDrive-GeorgiaInstituteofTechnology/learn/cs336/assignments/assignment1-basics/data/bpe/tiny_stories_v2_gpt4_train_merges.json",
        special_tokens=["<|endoftext|>"],
    )

    encoding = tokenizer.encode("Héllò hôw <|endoftext|><|endoftext|> are ü? 🙃<|endoftext|>")
    print(encoding)

    decoding = [tokenizer.decode([x]) for x in encoding]
    print(decoding)


if __name__ == "__main__":
    main()
