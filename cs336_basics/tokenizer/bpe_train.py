from cs336_basics.tokenizer.bpe import BpeTokenizer
from cs336_basics.utils import (
    unstructure_bytes,
    structure_bytes,
    structure_tuple_dict,
    unstructure_tuple_dict,
)
from cs336_basics.tokenizer.bpe_domain import BpeHyperparameters
import signal
from contextlib import contextmanager
from cattr.gen import make_dict_structure_fn, override, make_dict_unstructure_fn
import hashlib
import base64
import cattrs
import json
from dataclasses import dataclass
from typing import BinaryIO
import os
from multiprocessing.pool import Pool
from pathlib import Path
from cs336_basics.domain import TokenId
from collections import deque
import multiprocessing
import logging
import pprint
import regex as re

logger = logging.getLogger(__name__)


@dataclass
class BpeCheckpointParams:
    vocab: dict[int, bytes]
    merges: list[tuple[bytes, bytes]]
    pretoken_counts: dict[tuple[TokenId, ...], int]


@dataclass
class BpeCheckpoint:
    data_sha256_digest: str
    hyper_params: BpeHyperparameters
    params: BpeCheckpointParams | None


@dataclass
class _PreTokenizerChunkInput:
    path: Path
    start: int
    end: int
    special_tokens: list[str]


class BpeTokenizerTrainer:
    PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""

    def __init__(
        self,
        data_path: Path,
        checkpoint_path: Path | None,
        hyper_params: BpeHyperparameters,
    ):
        self._converter = cattrs.Converter()

        self._converter.register_unstructure_hook(bytes, unstructure_bytes)
        self._converter.register_structure_hook(
            bytes, lambda val, _: structure_bytes(val)
        )
        self._converter.register_structure_hook(
            BpeCheckpointParams,
            make_dict_structure_fn(
                BpeCheckpointParams,
                self._converter,
                pretoken_counts=override(struct_hook=structure_tuple_dict),
            ),
        )
        self._converter.register_unstructure_hook(
            BpeCheckpointParams,
            make_dict_unstructure_fn(
                BpeCheckpointParams,
                self._converter,
                pretoken_counts=override(unstruct_hook=unstructure_tuple_dict),
            ),
        )

        self._checkpoint_path = checkpoint_path
        self._data_path = data_path
        try:
            self._checkpoint = BpeTokenizerTrainer._load_checkpoint(
                data_path, checkpoint_path, hyper_params, self._converter
            )
        except:
            self._checkpoint = BpeTokenizerTrainer._init_checkpoint(
                data_path, hyper_params
            )

        self._pair_counts: dict[tuple[TokenId, TokenId], int] = dict()

    @staticmethod
    def _load_checkpoint(
        data_path: Path,
        checkpoint_path: Path | None,
        hyper_params: BpeHyperparameters,
        converter: cattrs.Converter,
    ) -> BpeCheckpoint:
        if checkpoint_path is None:
            raise RuntimeError("Checkpointing disabled")

        with open(checkpoint_path, "r") as checkpoint_file:
            ck = converter.structure(json.load(checkpoint_file), BpeCheckpoint)
        with open(data_path, "rb") as data_file:
            digest = hashlib.file_digest(data_file, "sha256").hexdigest()
        if digest != ck.data_sha256_digest:
            raise ValueError("Could not match digest")
        if hyper_params != ck.hyper_params:
            raise ValueError("Could not hyper parameters")
        return ck

    @staticmethod
    def _init_checkpoint(
        data_path: Path, hyper_params: BpeHyperparameters
    ) -> BpeCheckpoint:
        with open(data_path, "rb") as data_file:
            digest = hashlib.file_digest(data_file, "sha256").hexdigest()
        return BpeCheckpoint(
            data_sha256_digest=digest, hyper_params=hyper_params, params=None
        )

    def train(self):
        if self._checkpoint.params is None:
            logger.info("Initializing checkpoint parameters")
            self._init_ck_params()

        # Initialize structures
        logger.info("Initial pair counts")
        self._pair_counts = BpeTokenizerTrainer._initial_pair_counts(
            self._ck_params().pretoken_counts
        )

        logger.info("Entering main loop")
        while (
            len(self._pair_counts) > 0
            and len(self._ck_params().vocab)
            < self._checkpoint.hyper_params.max_vocab_size
        ):
            if len(self._ck_params().merges) % 25 == 0:
                logger.info(
                    f"Completed {len(self._ck_params().merges)} merges. "
                    + f"Current vocab size is {len(self._ck_params().vocab)}. "
                    + f"Longest token in vocab is {max(self._ck_params().vocab.values(), key=lambda bts: len(bts))}"
                )
            if (
                self._checkpoint_path is not None
                and len(self._ck_params().merges) % 100 == 0
            ):
                logger.info(f"Saving checkpoint at {self._checkpoint_path}...")
                with delay_keyboard_interrupt():
                    self._save_checkpoint_atomic()
                logger.info(f"Checkpoint saved")
            # Step: find most common token pair
            best_pair = self._step_most_frequent_pair()
            # Step: add pair to vocabulary
            self._step_add_pair_to_vocab(best_pair)

        logger.info("Finished training. Saving final checkpoint...")
        with delay_keyboard_interrupt():
            self._save_checkpoint_atomic()
        logger.info("Done")

    def tokenizer(self) -> BpeTokenizer:
        return BpeTokenizer(
            vocab=self._ck_params().vocab,
            merges=self._ck_params().merges,
            data_sha256_digest=self._checkpoint.data_sha256_digest,
            hyper_params=self._checkpoint.hyper_params,
        )

    def _save_checkpoint_atomic(self):
        if self._checkpoint_path is None:
            return

        tmp_path = self._checkpoint_path.with_name(self._checkpoint_path.name + ".tmp")
        with open(tmp_path, "w") as checkpoint_file:
            json.dump(self._converter.unstructure(self._checkpoint), checkpoint_file)
            checkpoint_file.flush()
            os.fsync(checkpoint_file.fileno())

        os.replace(tmp_path, self._checkpoint_path)

    def _init_ck_params(self):
        # Initialize vocab
        vocab = {
            i: tok
            for i, tok in enumerate(
                [bytes([i]) for i in range(256)]
                + [
                    tok.encode("utf-8")
                    for tok in self._checkpoint.hyper_params.special_tokens
                ]
            )
        }
        merges: list[tuple[bytes, bytes]] = []
        pretoken_counts = BpeTokenizerTrainer._get_initial_pretoken_counts(
            self._data_path, self._checkpoint.hyper_params
        )

        self._checkpoint.params = BpeCheckpointParams(
            vocab=vocab, merges=merges, pretoken_counts=pretoken_counts
        )

    @staticmethod
    def _get_initial_pretoken_counts(
        data_path: Path, hyper_params: BpeHyperparameters
    ) -> dict[tuple[TokenId, ...], int]:
        boundaries = BpeTokenizerTrainer._find_chunk_boundaries(
            data_path,
            2**21,
            hyper_params.chunk_delimiter,  # b"<|endoftext|>"
        )
        inputs = map(
            lambda bounds: _PreTokenizerChunkInput(
                data_path,
                bounds[0],
                bounds[1],
                hyper_params.special_tokens,
            ),
            zip(boundaries[:-1], boundaries[1:]),
        )

        total = dict()

        with Pool() as pool:
            partial_results = pool.imap_unordered(
                BpeTokenizerTrainer._get_chunk_pre_token_counts, inputs
            )

            for partial in partial_results:
                for pretok, count in partial.items():
                    total[pretok] = total.get(pretok, 0) + count

        return total

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
        input: _PreTokenizerChunkInput,
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

    @staticmethod
    def _initial_pair_counts(
        pretoken_counts: dict[tuple[TokenId, ...], int],
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
                    or compare_tok_pair(pair, best_pair, self._ck_params().vocab) > 0
                )
            ):
                best_pair = pair
                best_freq = freq

        assert best_pair is not None

        return best_pair

    def _step_add_pair_to_vocab(self, pair: tuple[TokenId, TokenId]):
        vocab = self._ck_params().vocab
        pretoken_counts = self._ck_params().pretoken_counts

        self._ck_params().merges.append((vocab[pair[0]], vocab[pair[1]]))
        new_tok = vocab[pair[0]] + vocab[pair[1]]
        new_tok_id = len(vocab)
        vocab[new_tok_id] = new_tok

        # Update tokenized text and pair counts
        logger.debug(f"Replacing token {new_tok} in {len(pretoken_counts)} pre-tokens")

        changes = []

        for old_pretok, pretok_count in pretoken_counts.items():
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
            count = pretoken_counts.pop(old_pretok)
            pretoken_counts[new_pretok] = count

    def _ck_params(self) -> BpeCheckpointParams:
        assert self._checkpoint.params is not None
        return self._checkpoint.params


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


@contextmanager
def delay_keyboard_interrupt():
    old_mask = signal.pthread_sigmask(signal.SIG_BLOCK, {signal.SIGINT})
    try:
        yield
    finally:
        signal.pthread_sigmask(signal.SIG_SETMASK, old_mask)


def training_run(path: str, max_vocab_size: int):
    file_path = Path(path)
    save_path = Path(f"tokenizers/final/{file_path.name}.tok.json")
    checkpoint_path = Path(f"tokenizers/checkpoints/{file_path.name}.ck.json")

    tokenizer_trainer = BpeTokenizerTrainer(
        data_path=file_path,
        checkpoint_path=checkpoint_path,
        hyper_params=BpeHyperparameters(
            max_vocab_size=max_vocab_size,
            special_tokens=["<|endoftext|>"],
            chunk_delimiter=b"<|endoftext|>",
        ),
    )
    tokenizer_trainer.train()
    tokenizer_trainer.tokenizer().save_to_file(save_path)


def test_entrypoint():
    logging.basicConfig(level=logging.INFO)

    training_run("data/owt_train.txt", 32000)
    # training_run("data/TinyStoriesV2-GPT4-train.txt", 10000)
