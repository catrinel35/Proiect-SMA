"""
Display module for the Tile World MAS.
Provides text-based grid visualization to the console.
"""

import time
from typing import Dict, List, Tuple, Optional
from environment import Environment


# ANSI color codes for terminal output
ANSI_COLORS = {
    "blue":    "\033[34m",
    "green":   "\033[32m",
    "red":     "\033[31m",
    "yellow":  "\033[33m",
    "cyan":    "\033[36m",
    "magenta": "\033[35m",
    "white":   "\033[37m",
    "reset":   "\033[0m",
    "bold":    "\033[1m",
    "grey":    "\033[90m",
}


def colorize(text: str, color: str) -> str:
    c = ANSI_COLORS.get(color.lower(), "")
    reset = ANSI_COLORS["reset"]
    return f"{c}{text}{reset}" if c else text


def short_color(color: str, length: int = 3) -> str:
    """Return uppercase abbreviation of color."""
    return color[:length].upper()


def render_grid(env: Environment) -> str:
    """Render the grid as a text table."""
    W = env.width
    H = env.height

    # Build cell contents
    cells: Dict[Tuple[int, int], List[str]] = {(x, y): [] for x in range(W) for y in range(H)}

    # Obstacles
    for (ox, oy) in env.obstacles:
        cells[(ox, oy)].append(colorize("###", "grey"))

    # Holes
    for (hx, hy), hole in env.holes.items():
        marker = f"H{hole.depth}{short_color(hole.color, 2)}"
        cells[(hx, hy)].append(colorize(marker, hole.color))

    # Tiles
    for (tx, ty), tile_list in env.tiles.items():
        color_counts: Dict[str, int] = {}
        for tile in tile_list:
            color_counts[tile.color] = color_counts.get(tile.color, 0) + 1
        for color, count in color_counts.items():
            marker = f"T{count}{short_color(color, 2)}"
            cells[(tx, ty)].append(colorize(marker, color))

    # Agents
    for aid, agent in env.agents.items():
        ax, ay = agent.position
        carry = f"[{short_color(agent.carried_tile, 2)}]" if agent.carried_tile else ""
        marker = f"@{short_color(agent.color, 3)}{carry}"
        cells[(ax, ay)].append(colorize(marker, agent.color))

    # Build output
    col_width = 12
    separator = "+" + (("-" * col_width + "+") * W)

    lines = []
    # Header row
    header = " " * 3
    for x in range(W):
        header += f"{'  ' + str(x):<{col_width}}"
    lines.append(header)
    lines.append(separator)

    for y in range(H):
        # Each row may need multiple sub-lines if cells have multiple items
        max_items = max(len(cells[(x, y)]) for x in range(W))
        max_items = max(max_items, 1)

        for sub in range(max_items):
            row = f"{y:2} " if sub == 0 else "   "
            for x in range(W):
                items = cells[(x, y)]
                item = items[sub] if sub < len(items) else ""
                # strip ANSI for padding calculation
                visible_len = len(_strip_ansi(item))
                padding = col_width - visible_len - 1
                row += "|" + item + " " * max(0, padding)
            row += "|"
            lines.append(row)

        lines.append(separator)

    return "\n".join(lines)


def _strip_ansi(s: str) -> str:
    import re
    return re.sub(r'\033\[[0-9;]*m', '', s)


def render_agent_summary(env: Environment) -> str:
    lines = []
    for aid, agent in env.agents.items():
        carry = agent.carried_tile if agent.carried_tile else "nothing"
        color_str = colorize(agent.color.capitalize(), agent.color)
        lines.append(f"  {color_str} agent: {agent.points} points; carries {carry}")
    return "\n".join(lines)


def display_state(env: Environment, log_buffer: List[str]):
    elapsed = int(time.time() - env.start_time)
    print("\n" + "=" * 60)
    print(colorize(f"Time: {elapsed}s", "bold"))
    print("=" * 60)
    print(render_grid(env))
    print(render_agent_summary(env))
    if log_buffer:
        print("-" * 60)
        for line in log_buffer:
            print(line)
    print("=" * 60, flush=True)
