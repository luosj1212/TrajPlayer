from __future__ import annotations

import re


class ChainSelectionError(ValueError):
    pass


def parse_chain_selection(text: str, maximum: int) -> tuple[int, ...]:
    chain_count = int(maximum)
    if chain_count < 1:
        raise ChainSelectionError("No chains are available")

    normalized = str(text).strip().replace("\uff0c", ",").replace("\u2013", "-")
    if not normalized:
        raise ChainSelectionError("Enter one or more chain numbers")

    selected: set[int] = set()
    for token in re.split(r"[,;\s]+", normalized):
        match = re.fullmatch(r"(\d+)(?:-(\d+))?", token)
        if match is None:
            raise ChainSelectionError(
                "Use chain numbers, commas, and ranges such as 1,3-5"
            )
        start = int(match.group(1))
        stop = int(match.group(2) or start)
        if start < 1 or stop < 1:
            raise ChainSelectionError("Chain numbers start at 1")
        if stop < start:
            raise ChainSelectionError("A chain range must run from low to high")
        if stop > chain_count:
            raise ChainSelectionError(
                f"Chain {stop} is outside the available range 1-{chain_count}"
            )
        selected.update(range(start, stop + 1))

    return tuple(sorted(selected))


def format_chain_selection(chains: tuple[int, ...]) -> str:
    if not chains:
        return ""

    ranges: list[str] = []
    start = previous = int(chains[0])
    for value in chains[1:]:
        current = int(value)
        if current == previous + 1:
            previous = current
            continue
        ranges.append(str(start) if start == previous else f"{start}-{previous}")
        start = previous = current
    ranges.append(str(start) if start == previous else f"{start}-{previous}")
    return ",".join(ranges)
