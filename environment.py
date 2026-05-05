import threading
import time
import queue
from dataclasses import dataclass, field
from typing import Optional, Dict, List, Tuple, Any
from enum import Enum


class Direction(Enum):
    NORTH = "North"
    SOUTH = "South"
    EAST = "East"
    WEST = "West"


class OperationType(Enum):
    PICK = "Pick"
    DROP = "Drop"
    MOVE = "Move"
    USE_TILE = "UseTile"
    TRANSFER_POINTS = "TransferPoints"
    GET_STATE = "GetState"


@dataclass
class Operation:
    op_type: OperationType
    agent_id: str
    args: dict
    response_queue: queue.Queue
    timestamp: float = field(default_factory=time.time)


@dataclass
class Tile:
    color: str
    position: Tuple[int, int]


@dataclass
class Hole:
    depth: int
    color: str
    position: Tuple[int, int]


@dataclass
class AgentState:
    agent_id: str
    color: str
    position: Tuple[int, int]
    points: int = 0
    carried_tile: Optional[str] = None  # color of carried tile, or None


class Environment:
    def __init__(self, width: int, height: int, operation_time_ms: int):
        self.width = width
        self.height = height
        self.operation_time_ms = operation_time_ms
        self.operation_time_s = operation_time_ms / 1000.0

        self.obstacles: set = set()
        self.tiles: Dict[Tuple[int, int], List[Tile]] = {}
        self.holes: Dict[Tuple[int, int], Hole] = {}
        self.agents: Dict[str, AgentState] = {}

        self.message_queue: queue.Queue = queue.Queue()
        self.log_buffer: List[str] = []
        self.log_lock = threading.Lock()

        self._running = False
        self._thread = None

        self.start_time = time.time()

    def add_obstacle(self, x: int, y: int):
        self.obstacles.add((x, y))

    def add_tile(self, x: int, y: int, color: str, count: int = 1):
        pos = (x, y)
        if pos not in self.tiles:
            self.tiles[pos] = []
        for _ in range(count):
            self.tiles[pos].append(Tile(color=color, position=pos))

    def add_hole(self, x: int, y: int, color: str, depth: int):
        self.holes[(x, y)] = Hole(depth=depth, color=color, position=(x, y))

    def add_agent(self, agent_id: str, color: str, x: int, y: int):
        self.agents[agent_id] = AgentState(
            agent_id=agent_id, color=color, position=(x, y)
        )

    def _log(self, msg: str):
        elapsed = time.time() - self.start_time
        line = f"[{elapsed:.3f}][ENV] {msg}"
        with self.log_lock:
            self.log_buffer.append(line)
        print(line, flush=True)

    def get_log_buffer(self) -> List[str]:
        with self.log_lock:
            buf = self.log_buffer[:]
            self.log_buffer.clear()
        return buf

    def _direction_to_delta(self, direction: Direction) -> Tuple[int, int]:
        return {
            Direction.NORTH: (0, -1),
            Direction.SOUTH: (0, 1),
            Direction.EAST: (1, 0),
            Direction.WEST: (-1, 0),
        }[direction]

    def _is_passable(self, x: int, y: int) -> bool:
        if x < 0 or x >= self.width or y < 0 or y >= self.height:
            return False
        if (x, y) in self.obstacles:
            return False
        if (x, y) in self.holes and self.holes[(x, y)].depth > 0:
            return False
        return True

    def _process_operation(self, op: Operation) -> dict:
        agent = self.agents.get(op.agent_id)
        if agent is None:
            return {"success": False, "error": "Unknown agent"}

        elapsed = time.time() - self.start_time

        if op.op_type == OperationType.GET_STATE:
            return {"success": True, "state": self._get_full_state()}

        if op.op_type == OperationType.MOVE:
            direction = op.args["direction"]
            dx, dy = self._direction_to_delta(direction)
            nx, ny = agent.position[0] + dx, agent.position[1] + dy

            self._log(f"[{agent.color}] Move {direction.value}")
            time.sleep(self.operation_time_s)

            if self._is_passable(nx, ny):
                agent.position = (nx, ny)
                self._log(f"[{agent.color}] Move complete -> ({nx},{ny})")
                return {"success": True}
            else:
                self._log(f"[{agent.color}] Move FAILED (blocked)")
                return {"success": False, "error": "Cell not passable"}

        elif op.op_type == OperationType.PICK:
            color = op.args["color"]
            pos = agent.position
            self._log(f"[{agent.color}] Pick {color}")
            time.sleep(self.operation_time_s)

            if agent.carried_tile is not None:
                self._log(f"[{agent.color}] Pick FAILED (already carrying)")
                return {"success": False, "error": "Already carrying a tile"}

            tiles_here = self.tiles.get(pos, [])
            for i, tile in enumerate(tiles_here):
                if tile.color == color:
                    tiles_here.pop(i)
                    if not tiles_here:
                        del self.tiles[pos]
                    agent.carried_tile = color
                    self._log(f"[{agent.color}] Pick {color} complete")
                    return {"success": True}

            self._log(f"[{agent.color}] Pick FAILED (no {color} tile here)")
            return {"success": False, "error": f"No {color} tile at position"}

        elif op.op_type == OperationType.DROP:
            self._log(f"[{agent.color}] Drop tile")
            time.sleep(self.operation_time_s)

            if agent.carried_tile is None:
                self._log(f"[{agent.color}] Drop FAILED (not carrying)")
                return {"success": False, "error": "Not carrying a tile"}

            pos = agent.position
            color = agent.carried_tile
            agent.carried_tile = None
            if pos not in self.tiles:
                self.tiles[pos] = []
            self.tiles[pos].append(Tile(color=color, position=pos))
            self._log(f"[{agent.color}] Drop complete at {pos}")
            return {"success": True}

        elif op.op_type == OperationType.USE_TILE:
            direction = op.args["direction"]
            dx, dy = self._direction_to_delta(direction)
            hx, hy = agent.position[0] + dx, agent.position[1] + dy
            hole_pos = (hx, hy)

            self._log(f"[{agent.color}] UseTile {direction.value}")
            time.sleep(self.operation_time_s)

            if agent.carried_tile is None:
                self._log(f"[{agent.color}] UseTile FAILED (not carrying)")
                return {"success": False, "error": "Not carrying a tile"}

            hole = self.holes.get(hole_pos)
            if hole is None or hole.depth == 0:
                self._log(f"[{agent.color}] UseTile FAILED (no valid hole)")
                return {"success": False, "error": "No hole in that direction"}

            tile_color = agent.carried_tile
            agent.carried_tile = None

            points_earned = 0
            points_owner = None

            if tile_color == hole.color:
                if hole.depth == 1:
                    points_earned = 50
                else:
                    points_earned = 10
                points_owner = hole.color

            hole.depth -= 1
            if hole.depth == 0:
                del self.holes[hole_pos]

            if points_earned > 0 and points_owner:
                for a in self.agents.values():
                    if a.color == points_owner:
                        a.points += points_earned
                        self._log(
                            f"[{agent.color}] UseTile complete -> {points_owner} agent gets {points_earned} pts"
                        )
                        break
            else:
                self._log(f"[{agent.color}] UseTile complete (no points, color mismatch)")

            return {"success": True, "points_earned": points_earned}

        elif op.op_type == OperationType.TRANSFER_POINTS:
            target_id = op.args["agent_id"]
            points = op.args["points"]
            target = self.agents.get(target_id)

            self._log(f"[{agent.color}] TransferPoints {points} -> {target_id}")
            time.sleep(self.operation_time_s)

            if target is None:
                return {"success": False, "error": "Target agent not found"}

            agent.points -= points
            target.points += points
            self._log(f"[{agent.color}] Transfer complete")
            return {"success": True}

        return {"success": False, "error": "Unknown operation"}

    def _get_full_state(self) -> dict:
        return {
            "width": self.width,
            "height": self.height,
            "obstacles": list(self.obstacles),
            "tiles": {
                f"{pos[0]},{pos[1]}": [t.color for t in tile_list]
                for pos, tile_list in self.tiles.items()
            },
            "holes": {
                f"{pos[0]},{pos[1]}": {"depth": h.depth, "color": h.color}
                for pos, h in self.holes.items()
            },
            "agents": {
                aid: {
                    "color": a.color,
                    "position": a.position,
                    "points": a.points,
                    "carried_tile": a.carried_tile,
                }
                for aid, a in self.agents.items()
            },
        }

    def run(self):
        self._running = True
        while self._running:
            try:
                op: Operation = self.message_queue.get(timeout=0.05)
                result = self._process_operation(op)
                op.response_queue.put(result)
            except queue.Empty:
                pass

    def start(self):
        self._thread = threading.Thread(target=self.run, daemon=True)
        self._thread.start()

    def stop(self):
        self._running = False
        if self._thread:
            self._thread.join(timeout=2)

    def send_operation(self, op: Operation) -> dict:
        resp_q = op.response_queue
        self.message_queue.put(op)
        return resp_q.get()

    def get_state(self, agent_id: str) -> dict:
        resp_q = queue.Queue()
        op = Operation(
            op_type=OperationType.GET_STATE,
            agent_id=agent_id,
            args={},
            response_queue=resp_q,
        )
        return self.send_operation(op)["state"]
