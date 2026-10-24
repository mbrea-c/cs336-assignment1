from typing import Self
import json
from pathlib import Path
from cattr.gen import override
from cattrs.gen import make_dict_structure_fn, make_dict_unstructure_fn
from cs336_basics.utils import unstructure_bytes, structure_bytes
import cattrs
from cs336_basics.tokenizer.bpe_domain import BpeHyperparameters
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

    def encode(self, text: str) -> list[int]:
        raise NotImplementedError()
