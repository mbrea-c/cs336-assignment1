import itertools
import re
import logging
from cs336_basics.domain import TokenId
import regex
from typing import Self, Iterable, Iterator
import json
from pathlib import Path
from cattr.gen import override
from cattrs.gen import make_dict_structure_fn, make_dict_unstructure_fn
from cs336_basics.utils import unstructure_bytes, structure_bytes
import cattrs
from cs336_basics.tokenizer.bpe_domain import BpeHyperparameters, PRETOK_PATTERN
from dataclasses import dataclass


@dataclass
class BpeTokenizerData:
    vocab: dict[int, bytes]
    merges: list[tuple[bytes, bytes]]

    data_sha256_digest: str
    hyper_params: BpeHyperparameters

    @staticmethod
    def from_file(tokenizer_path: Path) -> "BpeTokenizerData":
        with open(tokenizer_path, "r") as tokenizer_file:
            tokenizer = BpeTokenizerData._converter().structure(
                json.load(tokenizer_file), BpeTokenizerData
            )

        return tokenizer

    def save_to_file(self, path: Path):
        with open(path, "w") as file:
            json.dump(BpeTokenizerData._converter().unstructure(self), file)

    @staticmethod
    def _converter() -> cattrs.Converter:
        converter = cattrs.Converter()

        converter.register_unstructure_hook(bytes, unstructure_bytes)
        converter.register_structure_hook(bytes, lambda val, _: structure_bytes(val))

        return converter


@dataclass
class _SpecialToken:
    token: TokenId


@dataclass
class _SubDocument:
    text: str


class BpeTokenizer:
    def __init__(self, data: BpeTokenizerData):
        self.data = data
        self._reverse_vocab = {val: token for token, val in self.data.vocab.items()}
        self._tok_merges = [
            (
                self._reverse_vocab[left_val],
                self._reverse_vocab[right_val],
                self._reverse_vocab[left_val + right_val],
            )
            for left_val, right_val in self.data.merges
        ]

    def encode(self, text: str) -> list[TokenId]:
        # print("Token 198: ", self.data.vocab[198].__repr__())
        # print("Token 50256: ", self.data.vocab[50256].__repr__())
        # print("Special tokens: ", self.data.hyper_params.special_tokens)

        sequence = []

        if len(self.data.hyper_params.special_tokens) > 0:
            special_tok_pattern = "|".join(
                [
                    regex.escape(stok, special_only=False)
                    for stok in self.data.hyper_params.special_tokens
                ]
            )

            subdoc_start = 0

            for special_tok_match in regex.finditer(
                special_tok_pattern, text, flags=regex.POSIX
            ):
                subdoc_end: int = special_tok_match.start()
                if subdoc_end - subdoc_start > 0:
                    sequence.append(_SubDocument(text[subdoc_start:subdoc_end]))
                # print("special_tok: ", speciak_tok, special_tok_utf8)
                sequence.append(
                    _SpecialToken(
                        self._reverse_vocab[special_tok_match.group().encode("utf-8")]
                    )
                )
                subdoc_start = special_tok_match.end()
            if len(text) - subdoc_start > 0:
                sequence.append(_SubDocument(text[subdoc_start:]))
        else:
            sequence.append(_SubDocument(text[:]))

        encoded_toks = []

        # print("last item in sequence is: ", type(sequence[-1]))

        for item in sequence:
            if isinstance(item, _SubDocument):
                # print("last char: ", item.text[-1].__repr__())
                pretok_vals: list[list[bytes]] = []
                for match in regex.finditer(PRETOK_PATTERN, item.text):
                    match_text: str = match.group(0)
                    match_text_utf8 = match_text.encode("utf-8")
                    pretok_vals.append([bytes([c]) for c in match_text_utf8])
                pretoks = [
                    [self._reverse_vocab[val] for val in pretok]
                    for pretok in pretok_vals
                ]
                # print("last pretok: ", pretoks[-1])
                for left, right, new in self._tok_merges:
                    for pretok in pretoks:
                        i = 0
                        while i < len(pretok) - 1:
                            if pretok[i] == left and pretok[i + 1] == right:
                                # apply merge
                                pretok.pop(i + 1)
                                pretok.pop(i)
                                pretok.insert(i, new)
                            i += 1
                for pretok in pretoks:
                    for tok in pretok:
                        encoded_toks.append(tok)
            else:
                encoded_toks.append(item.token)

        # print(encoded_toks[-1])
        return encoded_toks

    def encode_iterable(self, iterable: Iterable[str]) -> Iterator[TokenId]:
        for text in iterable:
            for tok in self.encode(text):
                yield tok

    def decode(self, ids: list[TokenId]) -> str:
        return b"".join([self.data.vocab[tok] for tok in ids]).decode(
            "utf-8", errors="replace"
        )


def test_entrypoint():
    # text = """
    # The cat ate some stuff. He didn't really like it too much I fear. He might have just figured out that there was some medicine mixed in.<|endoftext|>
    # A short one.<|endoftext|>
    # Yet another story. In this one Sarits is reading a book, and she's very focused. I wonder what's happening in the book?<|endoftext|>
    # """
    text = ""
    logging.basicConfig(level=logging.INFO)

    tokenizer_data = BpeTokenizerData.from_file(
        Path("tokenizers/final/owt_train.txt.tok.json")
    )
    tokenizer = BpeTokenizer(tokenizer_data)
    tokenizer.data.hyper_params.special_tokens = []

    encoded = tokenizer.encode(text)

    print(encoded)
    print(tokenizer.decode(tokenizer.encode(text)).__repr__())

    print(
        f"Text length {len(text.encode("utf-8"))}, encoded length {len(encoded)}, compression ratio {len(encoded) / len(text.encode("utf-8")) if len(text.encode("utf-8")) > 0 else 'n/a'}"
    )

    # training_run("data/TinyStoriesV2-GPT4-train.txt", 10000)
