from dataclasses import dataclass

PRETOK_PATTERN = (
    r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
)


@dataclass
class BpeHyperparameters:
    max_vocab_size: int
    special_tokens: list[str]
    chunk_delimiter: bytes
