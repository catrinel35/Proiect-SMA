"""
Input parser for the Tile World MAS.
Reads the system configuration file.
"""

import re
from dataclasses import dataclass, field
from typing import List, Tuple


@dataclass
class SystemConfig:
    N: int                          # number of agents
    t: int                          # ms per operation
    T: int                          # total simulation time ms
    W: int                          # grid width
    H: int                          # grid height
    colors: List[str]               # agent colors
    agent_positions: List[Tuple[int, int]]
    obstacles: List[Tuple[int, int]] = field(default_factory=list)
    tiles: List[Tuple[int, str, int, int]] = field(default_factory=list)  # (count, color, x, y)
    holes: List[Tuple[int, str, int, int]] = field(default_factory=list)  # (depth, color, x, y)


def parse_input(filepath: str) -> SystemConfig:
    with open(filepath, "r") as f:
        content = f.read()

    # Tokenize: split by any whitespace
    tokens = re.split(r'\s+', content.strip())
    idx = 0

    def next_token():
        nonlocal idx
        t = tokens[idx]
        idx += 1
        return t

    def next_int():
        return int(next_token())

    N = next_int()
    t = next_int()
    T = next_int()
    W = next_int()
    H = next_int()

    colors = [next_token() for _ in range(N)]

    agent_positions = []
    for _ in range(N):
        x = next_int()
        y = next_int()
        agent_positions.append((x, y))

    obstacles = []
    tiles = []
    holes = []

    while idx < len(tokens):
        keyword = next_token()

        if keyword == "OBSTACLES":
            while idx < len(tokens) and not tokens[idx].isalpha() or (
                idx < len(tokens) and tokens[idx] not in ("TILES", "HOLES") and tokens[idx].lstrip('-').isdigit()
            ):
                if idx >= len(tokens):
                    break
                tok = tokens[idx]
                if tok in ("TILES", "HOLES"):
                    break
                x = next_int()
                y = next_int()
                obstacles.append((x, y))

        elif keyword == "TILES":
            while idx < len(tokens) and tokens[idx] not in ("OBSTACLES", "HOLES"):
                count = next_int()
                color = next_token()
                x = next_int()
                y = next_int()
                tiles.append((count, color, x, y))

        elif keyword == "HOLES":
            while idx < len(tokens) and tokens[idx] not in ("OBSTACLES", "TILES"):
                depth = next_int()
                color = next_token()
                x = next_int()
                y = next_int()
                holes.append((depth, color, x, y))

    return SystemConfig(
        N=N, t=t, T=T, W=W, H=H,
        colors=colors,
        agent_positions=agent_positions,
        obstacles=obstacles,
        tiles=tiles,
        holes=holes,
    )
