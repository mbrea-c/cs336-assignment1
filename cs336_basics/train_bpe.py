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
        self.vocab: dict[int, bytes] = dict()
        self.merges: list[tuple[bytes, bytes]] = list()

        self._pre_token_counts: dict[tuple[TokenId, ...], int] = dict()
        self._pair_counts: dict[tuple[TokenId, TokenId], int] = dict()

    def train(self, text: str, max_vocab_size: int, special_tokens: list[str]):
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
        self._update_pre_token_counts(text, special_tokens)

        # Initialize structures
        logger.info("Initial pair counts")
        self._pair_counts = self._initial_pair_counts(self._pre_token_counts)

        while len(self._pair_counts) > 0 and len(self.vocab) < max_vocab_size:
            # Step: find most common token pair
            best_pair = self._step_most_frequent_pair()
            # Step: add pair to vocabulary
            self._step_add_pair_to_vocab(best_pair)

    def _update_pre_token_counts(self, text: str, special_tokens: list[str]):
        self._pre_token_counts = self._get_chunk_pre_token_counts(text, special_tokens)

    def _get_chunk_pre_token_counts(
        self, text: str, special_tokens: list[str]
    ) -> dict[tuple[TokenId, ...], int]:
        counts = dict()
        for subdoc in re.split(
            "|".join([re.escape(stok, special_only=False) for stok in special_tokens]),
            text,
        ):
            for match in re.finditer(self.PAT, subdoc):
                match_text: str = match.group(0)
                match_text_utf8 = match_text.encode("utf-8")
                counts[match_text_utf8] = counts.get(match_text_utf8, 0) + 1

        return {
            tuple(self._initial_tokenize_text(pretok)): count
            for pretok, count in counts.items()
        }

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

        # Update tokenized text and pair counts
        logger.debug(
            f"Replacing token {new_tok} in {len(self._pre_token_counts)} pre-tokens"
        )

        changes = []

        for old_pretok, pretok_count in self._pre_token_counts.items():
            pretok = list(old_pretok)
            i = 0
            while i < len(pretok) - 1:
                next_i = i + 1
                if pretok[i] == pair[0] and pretok[next_i] == pair[1]:
                    if i > 0:
                        prev_i = i - 1
                        old_pair = pretok[prev_i], pair[0]
                        new_pair = pretok[prev_i], new_tok_id

                        self._pair_counts[old_pair] = max(
                            self._pair_counts.get(old_pair, 0) - pretok_count, 0
                        )
                        self._pair_counts[new_pair] = (
                            self._pair_counts.get(new_pair, 0) + pretok_count
                        )

                    if i < len(pretok) - 2:
                        next_next_i = i + 2
                        old_pair = pair[1], pretok[next_next_i]
                        new_pair = new_tok_id, pretok[next_next_i]

                        self._pair_counts[old_pair] = max(
                            self._pair_counts.get(old_pair, 0) - pretok_count, 0
                        )
                        self._pair_counts[new_pair] = (
                            self._pair_counts.get(new_pair, 0) + pretok_count
                        )

                    pretok.pop(next_i)
                    pretok.pop(i)
                    pretok.insert(i, new_tok_id)

                i += 1
            changes.append((old_pretok, tuple(pretok)))

        self._pair_counts.pop(pair)

        for old_pretok, new_pretok in changes:
            count = self._pre_token_counts.pop(old_pretok)
            self._pre_token_counts[new_pretok] = count


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

    with open("data/TinyStoriesV2-GPT4-valid.txt", "r", encoding="utf-8") as file:
        text = file.read()
        tokenizer.train(text, 2000, ["<|endoftext|>"])

    pprint.pprint(tokenizer.vocab)
    pprint.pprint(tokenizer.merges)
