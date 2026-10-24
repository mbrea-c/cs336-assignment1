import base64


def unstructure_bytes(val: bytes) -> str:
    try:
        val.decode("utf-8")
        return (b"utf8:" + val).decode("utf-8")
    except:
        return (b"b64:" + base64.b64encode(val)).decode("ascii")


def structure_bytes(val: str) -> bytes:
    if val.startswith("utf8:"):
        trimmed = val.removeprefix("utf8:")
        return trimmed.encode("utf-8")
    elif val.startswith("b64:"):
        trimmed = val.removeprefix("b64:")
        return base64.b64decode(trimmed)
    else:
        raise ValueError("`bytes` encoding not recognized")


def unstructure_tuple_dict[K, V](
    val: dict[tuple[K, ...], V],
) -> list[tuple[tuple[K, ...], V]]:
    return [(key, value) for key, value in val.items()]


def structure_tuple_dict[K, V](
    val: list[tuple[tuple[K, ...], V]],
    _,
) -> dict[tuple[K, ...], V]:
    return {tuple(key): value for key, value in val}
