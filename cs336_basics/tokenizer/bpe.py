import logging
from cs336_basics.domain import TokenId
import regex as re
from typing import Self
import json
from pathlib import Path
from cattr.gen import override
from cattrs.gen import make_dict_structure_fn, make_dict_unstructure_fn
from cs336_basics.utils import unstructure_bytes, structure_bytes
import cattrs
from cs336_basics.tokenizer.bpe_domain import BpeHyperparameters, PRETOK_PATTERN
from dataclasses import dataclass


@dataclass
class BpeTokenizer:
    vocab: dict[int, bytes]
    merges: list[tuple[bytes, bytes]]

    data_sha256_digest: str
    hyper_params: BpeHyperparameters

    @staticmethod
    def from_file(tokenizer_path: Path) -> "BpeTokenizer":
        with open(tokenizer_path, "r") as tokenizer_file:
            tokenizer = BpeTokenizer._converter().structure(
                json.load(tokenizer_file), BpeTokenizer
            )

        return tokenizer

    def save_to_file(self, path: Path):
        with open(path, "w") as file:
            json.dump(BpeTokenizer._converter().unstructure(self), file)

    @staticmethod
    def _converter() -> cattrs.Converter:
        converter = cattrs.Converter()

        converter.register_unstructure_hook(bytes, unstructure_bytes)
        converter.register_structure_hook(bytes, lambda val, _: structure_bytes(val))

        return converter

    def encode(self, text: str) -> list[TokenId]:
        pretoks = []

        for subdoc in re.split(
            "|".join(
                [
                    re.escape(stok, special_only=False)
                    for stok in self.hyper_params.special_tokens
                ]
            ),
            text,
        ):
            for match in re.finditer(PRETOK_PATTERN, subdoc):
                match_text: str = match.group(0)
                match_text_utf8 = match_text.encode("utf-8")
                pretoks.append([bytes([c]) for c in match_text_utf8])

        for left, right in self.merges:
            for pretok in pretoks:
                i = 0
                while i < len(pretok) - 1:
                    if pretok[i] == left and pretok[i + 1] == right:
                        # apply merge
                        pretok.pop(i + 1)
                        pretok.pop(i)
                        pretok.insert(i, left + right)
                    i += 1

        return pretoks


def test_entrypoint():
    logging.basicConfig(level=logging.INFO)

    tokenizer = BpeTokenizer.from_file(Path("tokenizers/final/owt_train.txt.tok.json"))
    print(tokenizer.encode("the cat ate"))

    # training_run("data/TinyStoriesV2-GPT4-train.txt", 10000)
