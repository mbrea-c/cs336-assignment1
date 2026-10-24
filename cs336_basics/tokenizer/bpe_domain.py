from dataclasses import dataclass


@dataclass
class BpeHyperparameters:
    max_vocab_size: int
    special_tokens: list[str]
    chunk_delimiter: bytes
