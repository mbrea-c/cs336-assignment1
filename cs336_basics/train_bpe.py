from cs336_basics.domain import TokenId
from cs336_basics.text import TokenizedText, TextCoords
from collections import deque
import multiprocessing
import logging
import pprint
import regex as re

logger = logging.getLogger(__name__)

class BpeTokenizer:
    PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""

    def __init__(self):
        self.vocab: dict[int, bytes] = {i: bytes(i) for i in range(256)}
        self.reverse_vocab = {v: k for k, v in self.vocab.items()}
        self.merges: list[tuple[bytes, bytes]] = []

        self._pre_token_counts: dict[bytes, int] = dict()
        self._tokenized_text: TokenizedText = TokenizedText([])
        self._pair_counts: dict[tuple[TokenId, TokenId], int] = dict()

    def train(self, input_path: str, max_vocab_size: int):
        logger.info("Reading training data")
        with open(input_path, "r", encoding="utf-8") as file:
            text = file.read()
            # Pre-tokenize
            logger.info("Updating pre-token counts")
            self._update_pre_token_counts(text)

            # Initialize structures
            text_utf8 = text.encode("utf-8")
            logger.info("Initial text tokenization")
            tokens = self._tokenize_text(text_utf8)
            self._tokenized_text = TokenizedText(tokens)
            logger.info("Initial pair counts")
            self._pair_counts = self._initial_pair_counts(tokens)

            while len(self._tokenized_text) > 1 and len(self.vocab) < max_vocab_size:
                # Step: find most common token pair
                best_pair = self._step_most_frequent_pair()
                # Step: add pair to vocabulary
                self._step_add_pair_to_vocab(best_pair)

    def _update_pre_token_counts(self, text: str):
        self._pre_token_counts = self._get_chunk_pre_token_counts(text)

    def _get_chunk_pre_token_counts(self, text: str) -> dict[bytes, int]:
        counts = dict()
        for match in re.finditer(self.PAT, text):
            match_text: str = match.group(0)
            match_text_utf8 = match_text.encode("utf-8")
            counts[match_text_utf8] = counts.get(match_text_utf8, 0) + 1
        return counts

    def _tokenize_text(self, text_utf8: bytes) -> list[TokenId]:
        return [self.reverse_vocab[bytes(c)] for c in text_utf8]

    def _initial_pair_counts(
        self, tokenized_text: list[TokenId]
    ) -> dict[tuple[TokenId, TokenId], int]:
        pair_counts = dict()
        for i in range(len(tokenized_text) - 1):
            pair = tokenized_text[i], tokenized_text[i + 1]
            pair_counts[pair] = pair_counts.get(pair, 0) + 1
        return pair_counts

    def _step_most_frequent_pair(self) -> tuple[TokenId, TokenId]:
        best_pair = None
        best_freq = 0

        for pair, freq in self._pair_counts.items():
            if freq > best_freq or (
                freq == best_freq
                and (best_pair is None or compare_tok_pair(pair, best_pair) > 0)
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
        self.reverse_vocab[new_tok] = new_tok_id

        # Update tokenized text and pair counts
        logger.info(
            f"Replacing token {new_tok} in text of length {len(self._tokenized_text)}"
        )
        i = TextCoords(0, 0)
        while self._tokenized_text._has_next_coord(i):
            next_i = self._tokenized_text._next_coord(i)
            if (
                self._tokenized_text._get(i) == pair[0]
                and self._tokenized_text._get(next_i) == pair[1]
            ):
                # logger.info(f"Replacing token {new_tok} at {i}")
                if self._tokenized_text._has_prev_coord(i):
                    prev_i = self._tokenized_text._prev_coord(i)
                    old_pair = self._tokenized_text._get(prev_i), pair[0]
                    new_pair = self._tokenized_text._get(prev_i), new_tok_id

                    self._pair_counts[old_pair] = max(
                        self._pair_counts.get(old_pair, 0) - 1, 0
                    )
                    self._pair_counts[new_pair] = self._pair_counts.get(new_pair, 0) + 1

                if self._tokenized_text._has_next_coord(next_i):
                    next_next_i = self._tokenized_text._next_coord(next_i)
                    old_pair = pair[1], self._tokenized_text._get(next_next_i)
                    new_pair = new_tok_id, self._tokenized_text._get(next_next_i)
                    self._pair_counts[old_pair] = max(
                        self._pair_counts.get(old_pair, 0) - 1, 0
                    )
                    self._pair_counts[new_pair] = self._pair_counts.get(new_pair, 0) + 1

                self._tokenized_text._pop(next_i)
                self._tokenized_text._pop(i)
                self._tokenized_text._insert(i, new_tok_id)

            i = self._tokenized_text._next_coord(i)

        # Update token pair counts


def compare_tok_pair(
    left: tuple[TokenId, TokenId], right: tuple[TokenId, TokenId]
) -> int:
    if left[0] > right[0]:
        return 1
    elif left[0] < right[0]:
        return -1
    else:
        if left[1] > right[1]:
            return 1
        elif left[1] < right[1]:
            return -1
        else:
            return 0


def test_entrypoint():
    logging.basicConfig(level=logging.INFO)
    tokenizer = BpeTokenizer()

    tokenizer.train("data/TinyStoriesV2-GPT4-valid.txt", 2000)

    pprint.pprint(tokenizer.vocab)
    pprint.pprint(tokenizer.merges)
