import json
import os
from collections import Counter
from typing import List


class BPE():
    "BPE tokenizer"

    def __init__(self, vocab_size: int):
        self.vocab_size = vocab_size
        self.unique_tokens = None
        self.token2id = None
        self.id2token = None

    def _maxFreqTokens(self, tokens: List[str], n: int | None) -> tuple[str]:
        "If order does not matter"
        if n is None:
            n = len(tokens)
        cnt = {}
        max_frequency, max_fr_pair = 0, None
        for i in range(n - 1):
            pair = tokens[i] + tokens[i + 1]
            cnt[pair] = cnt.get(pair, 0) + 1
            if cnt[pair] > max_frequency:
                max_frequency = cnt[pair]
                max_fr_pair = tokens[i], tokens[i + 1]
        return max_fr_pair

    def _maxFreqTokens2(self, tokens: List[str], n: int | None) -> tuple[str]:
        if n is None:
            n = len(tokens)

        pair_counts = Counter(zip(tokens, tokens[1:]))
        max_freq = max(pair_counts.values())

        for i in range(n - 1):
            pair = (tokens[i], tokens[i + 1])
            if pair_counts[pair] == max_freq:
                return pair

    def fit(self, text: str) -> None:
        text_list = list(text)
        self.unique_tokens_set = set(text_list).copy()
        self.unique_tokens = sorted(self.unique_tokens_set).copy()

        while len(self.unique_tokens) < self.vocab_size:
            i, n = 0, len(text_list)
            max_fr_pair = self._maxFreqTokens2(text_list, n)
            new_text_list = []
            while i < n:
                if i != n - 1 and (text_list[i], text_list[i + 1]) == max_fr_pair:
                    pair = "".join(max_fr_pair)
                    if pair not in self.unique_tokens_set:
                        self.unique_tokens.append(pair)
                        self.unique_tokens_set.add(pair)
                    new_text_list.append(pair)
                    i += 1
                else:
                    new_text_list.append(text_list[i])
                i += 1
            text_list = new_text_list.copy()

        self.token2id = dict(zip(self.unique_tokens, range(self.vocab_size)))
        self.id2token = dict(zip(range(self.vocab_size), self.unique_tokens))

    def encode(self, text: str) -> List[int]:
        encode_tokens = []
        i, n = 0, len(text)
        while i < n:
            start_char = text[i]
            relevants = tuple(filter(lambda x: x.startswith(start_char), self.token2id.keys()))
            target = None
            while relevants and i < n - 1:
                i += 1
                start_char += text[i]
                target = relevants
                relevants = tuple(filter(lambda x: x.startswith(start_char), relevants))
            if target is None:
                target = start_char,
                i += 1
            large_token = target[0]
            encode_tokens.append(self.token2id[large_token])
        return encode_tokens

    def decode(self, tokens_id: List[int]) -> str:
        return "".join((self.id2token[token_id] for token_id in tokens_id))

    def save_pretrained(self, save_dir: str):
        os.makedirs(save_dir, exist_ok=True)

        with open(os.path.join(save_dir, "tokenizer_config.json"), "w", encoding="utf-8") as f:
            json.dump({"tokenizer_class": "BPE", "vocab_size": self.vocab_size}, f, indent=2)

        tokenizer_data = {
            "vocab": self.token2id,
            "id2token": {str(k): v for k, v in self.id2token.items()},
        }
        with open(os.path.join(save_dir, "tokenizer.json"), "w", encoding="utf-8") as f:
            json.dump(tokenizer_data, f, indent=2, ensure_ascii=False)

    @classmethod
    def from_pretrained(cls, save_dir: str):
        with open(os.path.join(save_dir, "tokenizer_config.json"), encoding="utf-8") as f:
            config = json.load(f)

        tokenizer = cls(vocab_size=config["vocab_size"])

        with open(os.path.join(save_dir, "tokenizer.json"), encoding="utf-8") as f:
            data = json.load(f)

        tokenizer.token2id = data["vocab"]
        tokenizer.id2token = {int(k): v for k, v in data["id2token"].items()}
        tokenizer.unique_tokens = list(data["vocab"].keys())

        return tokenizer
