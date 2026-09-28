from cs336_basics.domain import TokenId
from dataclasses import dataclass


@dataclass
class TextCoords:
    block: int
    offset: int


class TokenizedText:
    BLOCK_SIZE = 5000

    def __init__(self, tokenized_text: list[TokenId]):
        self._blocks: list[list[TokenId]] = TokenizedText._init_blocks(tokenized_text)

    @staticmethod
    def _init_blocks(tokenized_text: list[TokenId]) -> list[list[TokenId]]:
        blocks = list()

        curr_block = list()

        for tok in tokenized_text:
            if len(curr_block) < TokenizedText.BLOCK_SIZE:
                curr_block.append(tok)
            else:
                blocks.append(curr_block)
                curr_block = list()

        if len(curr_block) > 0:
            blocks.append(curr_block)

        return blocks

    def _find_coords(self, i: int) -> TextCoords:
        """
        Finds the block index, and the index of the token within the block

        Linear time
        """
        if i < 0:
            raise IndexError("Cannot negative index")

        offset = 0
        for block_i, block in enumerate(self._blocks):
            offset += len(block)
            if offset > i:
                return TextCoords(block_i, i - (offset - len(block)))

        raise IndexError("Index out of range")

    def _has_prev_coord(self, coords: TextCoords) -> bool:
        return (
            coords.offset > 0
            or coords.block > 0
            and coords.block - 1 < len(self._blocks)
        )

    def _prev_coord(self, coords: TextCoords) -> TextCoords:
        if coords.offset > 0:
            return TextCoords(coords.block, coords.offset - 1)
        elif coords.block > 0 and coords.block - 1 < len(self._blocks):
            return TextCoords(coords.block - 1, len(self._blocks[coords.block - 1]) - 1)
        else:
            raise IndexError("out of range")

    def _has_next_coord(self, coords: TextCoords) -> bool:
        return not (
            coords.block < 0 or coords.block >= len(self._blocks) or coords.offset < 0
        ) and (
            coords.offset < len(self._blocks[coords.block]) - 1
            or coords.block + 1 < len(self._blocks)
        )

    def _next_coord(self, coords: TextCoords) -> TextCoords:
        if coords.block < 0 or coords.block >= len(self._blocks) or coords.offset < 0:
            raise IndexError("out of range")

        if coords.offset < len(self._blocks[coords.block]) - 1:
            return TextCoords(coords.block, coords.offset + 1)
        elif coords.block + 1 < len(self._blocks):
            return TextCoords(coords.block + 1, 0)
        else:
            raise IndexError("out of range")

    def _is_in_range(self, coords: TextCoords, allow_new: bool = False) -> bool:
        return (
            coords.block >= 0
            and coords.block < len(self._blocks)
            and coords.offset >= 0
            and (
                coords.offset <= len(self._blocks[coords.block])
                if allow_new
                else coords.offset <= len(self._blocks[coords.block])
            )
        )

    def _verify_in_range(self, coords: TextCoords, allow_new: bool = False):
        if not self._is_in_range(coords, allow_new):
            raise IndexError(
                f"Text coordinates {coords} out of range for text with {len(self._blocks)} blocks"
            )

    def _pop(self, coords: TextCoords) -> TokenId:
        self._verify_in_range(coords)

        return self._blocks[coords.block].pop(coords.offset)

    def _insert(self, coords: TextCoords, tok: TokenId):
        self._verify_in_range(coords, True)

        self._blocks[coords.block].insert(coords.offset, tok)

    def _get(self, coords: TextCoords) -> TokenId:
        self._verify_in_range(coords)

        return self._blocks[coords.block][coords.offset]

    def __len__(self) -> int:
        return sum(map(lambda block: len(block), self._blocks))
