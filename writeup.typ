== Byte-Pair Encoding (BPE) Tokenizer

=== `unicode1`
#enum(
  numbering: "(a)",
  [
    It returns the null character, `\x00`.
  ],
  [
    The string representation is `\x00`, but the printed representation is an empty string.
  ],
  [
    Same as above, within a string representation `chr(0)` shows as `\x00` but
    when printed it shows as an empty string.
  ],
)

=== `unicode2`
#enum(
  numbering: "(a)",
  [
    The vocabulary needed to represent all possible codepoints is much smaller if
    code points are only 8 bits. With UTF-16 we would need a $2^16$, and up to $2^32$ for UTF-32
    (though only the last 21 bits in UTF-32 codepoints are actually used for unicode, so we
    could get away with a $2^21$ vocabulary).
  ],
  [
    UTF-8 is a variable-length encoding, which means that some unicode characters
    are represented by multiple codepoints (bytes). The given function assumes 1 byte =
    1 unicode character. Example: `"hello! こんにちは!"`.
  ],
  [
    `"\xff\xff"` is a 2-byte sequence that fails to decode to any unicode
    characters. The reason is that `\xff` is an invalid start byte for an
    unicode character.
  ],
)
