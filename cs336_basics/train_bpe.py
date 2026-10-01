import base64
import cattrs
import json
from dataclasses import dataclass
from typing import BinaryIO
import os
from multiprocessing.pool import Pool
from pathlib import Path
from cs336_basics.domain import TokenId
from cs336_basics.text import TokenizedText, TextCoords
from collections import deque
import multiprocessing
import logging
import pprint
import regex as re

logger = logging.getLogger(__name__)


@dataclass
class PreTokenizerChunkInput:
    path: Path
    start: int
    end: int
    special_tokens: list[str]


@dataclass
class BpeTokenizer:
    vocab: dict[int, bytes]
    merges: list[tuple[bytes, bytes]]


class BpeTokenizerTrainer:
    PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""

    def __init__(self):
        self.vocab: dict[int, bytes] = dict()
        self.merges: list[tuple[bytes, bytes]] = list()

        self._pre_token_counts: dict[tuple[TokenId, ...], int] = dict()
        self._pair_counts: dict[tuple[TokenId, TokenId], int] = dict()

    def train(
        self, path: Path, max_vocab_size: int, special_tokens: list[str]
    ) -> BpeTokenizer:
        # Initialize vocab
        self.vocab = {
            i: tok
            for i, tok in enumerate(
                [bytes([i]) for i in range(256)]
                + [tok.encode("utf-8") for tok in special_tokens]
            )
        }
        self.merges: list[tuple[bytes, bytes]] = []

        # Pre-tokenize
        logger.info("Pre-tokenizing")
        self._update_pre_token_counts(path, special_tokens)

        # Initialize structures
        logger.info("Initial pair counts")
        self._pair_counts = self._initial_pair_counts(self._pre_token_counts)

        logger.info("Entering main loop")
        while len(self._pair_counts) > 0 and len(self.vocab) < max_vocab_size:
            # Step: find most common token pair
            best_pair = self._step_most_frequent_pair()
            # Step: add pair to vocabulary
            self._step_add_pair_to_vocab(best_pair)

        return BpeTokenizer(self.vocab, self.merges)

    def _update_pre_token_counts(self, path: Path, special_tokens: list[str]):
        boundaries = BpeTokenizerTrainer._find_chunk_boundaries(
            path, 2**21, b"<|endoftext|>"
        )
        inputs = map(
            lambda bounds: PreTokenizerChunkInput(
                path, bounds[0], bounds[1], special_tokens
            ),
            zip(boundaries[:-1], boundaries[1:]),
        )

        with Pool() as pool:
            result = pool.map(BpeTokenizerTrainer._get_chunk_pre_token_counts, inputs)

        total = dict()

        for chunk_counts in result:
            for pretok, count in chunk_counts.items():
                total[pretok] = total.get(pretok, 0) + count
        self._pre_token_counts = total

    @staticmethod
    def _find_chunk_boundaries(
        path: Path, min_chunk_size: int, delimiter: bytes
    ) -> list[int]:
        boundaries = [0]
        curr_offset = 0
        mini_chunk_size = 4096

        with open(path, "rb") as f:
            f.seek(0, os.SEEK_END)
            file_size = f.tell()
            f.seek(0)

            while True:
                while True:
                    f.seek(max(0, curr_offset))
                    mini_chunk = f.read(mini_chunk_size)
                    curr_offset

                    # If EOF, this boundary should be at the end of the file
                    if mini_chunk == b"":
                        boundaries.append(file_size)
                        return boundaries

                    found_at = mini_chunk.find(delimiter)
                    if found_at != -1:
                        boundaries.append(curr_offset + found_at)
                        break
                    curr_offset += mini_chunk_size
                curr_offset += min_chunk_size

    @staticmethod
    def _get_chunk_pre_token_counts(
        input: PreTokenizerChunkInput,
    ) -> dict[tuple[TokenId, ...], int]:
        path = input.path
        boundaries = (input.start, input.end)
        special_tokens = input.special_tokens
        with open(path, "rb") as file:
            file.seek(boundaries[0])
            text = file.read(boundaries[1] - boundaries[0]).decode("utf-8")
        counts = dict()
        for subdoc in re.split(
            "|".join([re.escape(stok, special_only=False) for stok in special_tokens]),
            text,
        ):
            for match in re.finditer(BpeTokenizerTrainer.PAT, subdoc):
                match_text: str = match.group(0)
                match_text_utf8 = match_text.encode("utf-8")
                counts[match_text_utf8] = counts.get(match_text_utf8, 0) + 1

        return {tuple(c for c in pretok): count for pretok, count in counts.items()}

    def _initial_tokenize_text(self, text_utf8: bytes) -> list[TokenId]:
        return [c for c in text_utf8]

    def _initial_pair_counts(
        self, pretoken_counts: dict[tuple[TokenId, ...], int]
    ) -> dict[tuple[TokenId, TokenId], int]:
        pair_counts = dict()
        for pretok, count in pretoken_counts.items():
            pretok_pair_counts = dict()
            for i in range(len(pretok) - 1):
                pair = pretok[i], pretok[i + 1]
                pretok_pair_counts[pair] = pretok_pair_counts.get(pair, 0) + 1
            for pair, pair_count in pretok_pair_counts.items():
                pair_counts[pair] = pair_counts.get(pair, 0) + pair_count * count
        return pair_counts

    def _step_most_frequent_pair(self) -> tuple[TokenId, TokenId]:
        best_pair = None
        best_freq = 0

        for pair, freq in self._pair_counts.items():
            if freq > best_freq or (
                freq == best_freq
                and (
                    best_pair is None
                    or compare_tok_pair(pair, best_pair, self.vocab) > 0
                )
            ):
                best_pair = pair
                best_freq = freq

        assert best_pair is not None

        return best_pair

    def _step_add_pair_to_vocab(self, pair: tuple[TokenId, TokenId]):
        self.merges.append((self.vocab[pair[0]], self.vocab[pair[1]]))
        new_tok = self.vocab[pair[0]] + self.vocab[pair[1]]
        new_tok_id = len(self.vocab)
        self.vocab[new_tok_id] = new_tok

        # Update tokenized text and pair counts
        logger.debug(
            f"Replacing token {new_tok} in {len(self._pre_token_counts)} pre-tokens"
        )

        changes = []

        for old_pretok, pretok_count in self._pre_token_counts.items():
            pretok_joins = None
            i = 0
            while i < len(old_pretok) - 1:
                next_i = i + 1
                if old_pretok[i] == pair[0] and old_pretok[next_i] == pair[1]:
                    if i > 0:
                        prev_i = i - 1
                        old_pair = old_pretok[prev_i], pair[0]
                        new_pair = old_pretok[prev_i], new_tok_id

                        self._pair_counts[old_pair] = max(
                            self._pair_counts.get(old_pair, 0) - pretok_count, 0
                        )
                        self._pair_counts[new_pair] = (
                            self._pair_counts.get(new_pair, 0) + pretok_count
                        )

                    if i < len(old_pretok) - 2:
                        next_next_i = i + 2
                        old_pair = pair[1], old_pretok[next_next_i]
                        new_pair = new_tok_id, old_pretok[next_next_i]

                        self._pair_counts[old_pair] = max(
                            self._pair_counts.get(old_pair, 0) - pretok_count, 0
                        )
                        self._pair_counts[new_pair] = (
                            self._pair_counts.get(new_pair, 0) + pretok_count
                        )

                    if pretok_joins is None:
                        pretok_joins = deque()
                    pretok_joins.append(i)

                    i += 2
                else:
                    i += 1
            if pretok_joins is not None:
                pretok = list()
                i = 0
                while i < len(old_pretok):
                    if len(pretok_joins) > 0 and i == pretok_joins[0]:
                        pretok.append(new_tok_id)
                        pretok_joins.popleft()
                        i += 2
                    else:
                        pretok.append(old_pretok[i])
                        i += 1
                changes.append((old_pretok, tuple(pretok)))

        self._pair_counts.pop(pair)

        for old_pretok, new_pretok in changes:
            count = self._pre_token_counts.pop(old_pretok)
            self._pre_token_counts[new_pretok] = count


def compare_tok_pair(
    left: tuple[TokenId, TokenId],
    right: tuple[TokenId, TokenId],
    vocab: dict[TokenId, bytes],
) -> int:
    l = (vocab[left[0]], vocab[left[1]])
    r = (vocab[right[0]], vocab[right[1]])
    if l[0] > r[0]:
        return 1
    elif l[0] < r[0]:
        return -1
    else:
        if l[1] > r[1]:
            return 1
        elif l[1] < r[1]:
            return -1
        else:
            return 0


def byte_encoder(val: bytes) -> str:
    try:
        val.decode("utf-8")
        return (b"utf8:" + val).decode("utf-8")
    except:
        return (b"b64:" + base64.b64encode(val)).decode("ascii")


def byte_decoder(val: str) -> bytes:
    if val.startswith("utf8:"):
        trimmed = val.removeprefix("utf8:")
        return trimmed.encode("utf-8")
    elif val.startswith("b64:"):
        trimmed = val.removeprefix("b64:")
        return base64.b64decode(trimmed)
    else:
        raise ValueError("`bytes` encoding not recognized")


def test_entrypoint():
    logging.basicConfig(level=logging.INFO)
    tokenizer_trainer = BpeTokenizerTrainer()

    # file_path = Path("data/TinyStoriesV2-GPT4-train.txt")
    # tokenizer = tokenizer_trainer.train(file_path, 10000, ["<|endoftext|>"])

    file_path = Path("data/owt_train.txt")
    tokenizer = tokenizer_trainer.train(file_path, 32000, ["<|endoftext|>"])

    converter = cattrs.Converter()
    converter.register_unstructure_hook(bytes, byte_encoder)
    converter.register_structure_hook(bytes, lambda val, _: byte_decoder(val))

    save_path = f"tokenizers/{file_path.name}.json"

    with open(save_path, "w") as f:
        json.dump(converter.unstructure(tokenizer), f)

    logger.info(f"Saved tokenizer file at {save_path}")
