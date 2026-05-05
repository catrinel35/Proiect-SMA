import threading
import queue
import time
import random
from typing import Optional, Dict, List, Tuple, Any
from environment import Environment, Operation, OperationType, Direction


DIRECTIONS = [Direction.NORTH, Direction.SOUTH, Direction.EAST, Direction.WEST]

DIR_DELTA = {
    Direction.NORTH: (0, -1),
    Direction.SOUTH: (0, 1),
    Direction.EAST: (1, 0),
    Direction.WEST: (-1, 0),
}


class Message:
    def __init__(self, sender_id: str, receiver_id: str, content: dict):
        self.sender_id = sender_id
        self.receiver_id = receiver_id
        self.content = content
        self.timestamp = time.time()


class Agent:
    def __init__(self, agent_id: str, color: str, start_pos: Tuple[int, int], env: Environment):
        self.agent_id = agent_id
        self.color = color
        self.position = list(start_pos)
        self.env = env
        self.points = 0
        self.carried_tile: Optional[str] = None

        self.inbox: queue.Queue = queue.Queue()
        self.message_log: List[str] = []

        self._running = False
        self._thread = None

        self.start_time = None
        self._last_state_op_step = -1
        self._op_step = 0

        self._current_goal = None
        self._path = []

    def _log(self, msg: str):
        if self.start_time:
            elapsed = time.time() - self.start_time
            line = f"[{elapsed:.3f}][AGT][{self.color}] {msg}"
        else:
            line = f"[AGT][{self.color}] {msg}"
        self.message_log.append(line)
        print(line, flush=True)

    def send_message(self, other_agent: 'Agent', content: dict):
        elapsed = time.time() - self.start_time if self.start_time else 0
        msg = Message(sender_id=self.agent_id, receiver_id=other_agent.agent_id, content=content)
        other_agent.inbox.put(msg)
        self._log(f"MSG -> {other_agent.color}: {content.get('type', '?')}")
        self.env._log(f"[NEG][{self.color} -> {other_agent.color}] {content}")

    def receive_messages(self) -> List[Message]:
        messages = []
        while not self.inbox.empty():
            try:
                messages.append(self.inbox.get_nowait())
            except queue.Empty:
                break
        return messages

    def _do_operation(self, op_type: OperationType, **kwargs) -> dict:
        resp_q = queue.Queue()
        op = Operation(
            op_type=op_type,
            agent_id=self.agent_id,
            args=kwargs,
            response_queue=resp_q,
        )
        self._op_step += 1
        result = self.env.send_operation(op)
        if result["success"]:
            self._update_local_state_after_op(op_type, kwargs)
        return result

    def _update_local_state_after_op(self, op_type: OperationType, args: dict):
        if op_type == OperationType.MOVE:
            direction = args["direction"]
            dx, dy = DIR_DELTA[direction]
            self.position[0] += dx
            self.position[1] += dy
        elif op_type == OperationType.PICK:
            self.carried_tile = args["color"]
        elif op_type == OperationType.DROP:
            self.carried_tile = None
        elif op_type == OperationType.USE_TILE:
            self.carried_tile = None

    def move(self, direction: Direction) -> bool:
        result = self._do_operation(OperationType.MOVE, direction=direction)
        return result["success"]

    def pick(self, color: str) -> bool:
        result = self._do_operation(OperationType.PICK, color=color)
        return result["success"]

    def drop_tile(self) -> bool:
        result = self._do_operation(OperationType.DROP)
        return result["success"]

    def use_tile(self, direction: Direction) -> bool:
        result = self._do_operation(OperationType.USE_TILE, direction=direction)
        return result["success"]

    def transfer_points(self, target_agent_id: str, points: int) -> bool:
        result = self._do_operation(
            OperationType.TRANSFER_POINTS,
            agent_id=target_agent_id,
            points=points,
        )
        return result["success"]

    def get_state(self) -> dict:
        return self.env.get_state(self.agent_id)


    def _bfs(self, start: Tuple[int, int], goal: Tuple[int, int], state: dict) -> List[Direction]:
        obstacles = set(tuple(o) for o in state["obstacles"])
        holes = {
            tuple(int(x) for x in k.split(",")): v
            for k, v in state["holes"].items()
        }
        width = state["width"]
        height = state["height"]

        def passable(x, y):
            if x < 0 or x >= width or y < 0 or y >= height:
                return False
            if (x, y) in obstacles:
                return False
            pos = (x, y)
            if pos in holes and holes[pos]["depth"] > 0 and pos != goal:
                return False
            return True

        from collections import deque
        visited = {start: None}  # pos -> (prev_pos, direction)
        q = deque([start])
        parent = {start: (None, None)}

        while q:
            cur = q.popleft()
            if cur == goal:
                path = []
                while parent[cur][0] is not None:
                    path.append(parent[cur][1])
                    cur = parent[cur][0]
                path.reverse()
                return path

            for direction in DIRECTIONS:
                dx, dy = DIR_DELTA[direction]
                nx, ny = cur[0] + dx, cur[1] + dy
                npos = (nx, ny)
                if npos not in parent and passable(nx, ny):
                    parent[npos] = (cur, direction)
                    q.append(npos)

        return []  # no path found

    def _navigate_to(self, target: Tuple[int, int], state: dict) -> bool:
        pos = tuple(self.position)
        if pos == target:
            return True
        path = self._bfs(pos, target, state)
        if path:
            self.move(path[0])
        else:
            self.move(random.choice(DIRECTIONS))
        return False

    def _simple_strategy(self):
        state = self.get_state()

        my_color = self.color
        pos = tuple(self.position)
        obstacles = set(tuple(o) for o in state["obstacles"])
        width = state["width"]
        height = state["height"]

        def cell_accessible(x, y):
            if x < 0 or x >= width or y < 0 or y >= height:
                return False
            if (x, y) in obstacles:
                return False
            hole_key = f"{x},{y}"
            if hole_key in state["holes"] and state["holes"][hole_key]["depth"] > 0:
                return False
            return True

        # Parse holes and tiles
        holes = {
            tuple(int(x) for x in k.split(",")): v
            for k, v in state["holes"].items()
        }
        tiles = {
            tuple(int(x) for x in k.split(",")): v
            for k, v in state["tiles"].items()
        }

        my_holes = [(p, h) for p, h in holes.items() if h["color"] == my_color and h["depth"] > 0]
        any_holes = [(p, h) for p, h in holes.items() if h["depth"] > 0]
        # Find tiles of any color
        my_tiles = [(p, colors) for p, colors in tiles.items() if my_color in colors]
        any_tiles = [(p, colors) for p, colors in tiles.items() if colors]

        def get_valid_adjacent(hole_pos):
            valid = []
            for direction in DIRECTIONS:
                dx, dy = DIR_DELTA[direction]
                adj = (hole_pos[0] - dx, hole_pos[1] - dy)
                if cell_accessible(adj[0], adj[1]):
                    valid.append((adj, direction))
            return valid

        if not self.carried_tile:
            target_tiles = my_tiles if my_tiles else any_tiles
            if not target_tiles:
                # Nothing to pick anywhere, wander randomly
                self.move(random.choice(DIRECTIONS))
                return

            # Pick nearest tile cell
            target_pos = min(
                target_tiles,
                key=lambda x: abs(x[0][0] - pos[0]) + abs(x[0][1] - pos[1])
            )[0]

            if pos == target_pos:
                tile_colors = tiles.get(pos, [])
                if my_color in tile_colors:
                    self.pick(my_color)
                elif tile_colors:
                    self.pick(tile_colors[0])
                else:
                    # Tile disappeared, recalculate next step
                    self.move(random.choice(DIRECTIONS))
            else:
                self._navigate_to(target_pos, state)

        else:
            target_holes = my_holes if my_holes else any_holes

            if not target_holes:
                self.drop_tile()
                return
            best_hole = None
            best_adj = None
            best_dir = None
            best_dist = float('inf')

            for hole_pos, hole_info in target_holes:
                valid_adjs = get_valid_adjacent(hole_pos)
                for adj_pos, direction in valid_adjs:
                    dist = abs(adj_pos[0] - pos[0]) + abs(adj_pos[1] - pos[1])
                    if dist < best_dist:
                        best_dist = dist
                        best_hole = hole_pos
                        best_adj = adj_pos
                        best_dir = direction

            if best_adj is None:
                self.move(random.choice(DIRECTIONS))
                return

            if pos == best_adj:
                self.use_tile(best_dir)
            else:
                self._navigate_to(best_adj, state)

    def _communicate_intentions(self, other_agents: List['Agent'], state: dict):
        holes = {
            tuple(int(x) for x in k.split(",")): v
            for k, v in state["holes"].items()
        }
        my_holes = [p for p, h in holes.items() if h["color"] == self.color and h["depth"] > 0]

        if my_holes:
            target = my_holes[0]
            for other in other_agents:
                if other.agent_id != self.agent_id:
                    self.send_message(other, {
                        "type": "intention",
                        "agent": self.color,
                        "action": f"heading to fill hole at {target}",
                    })

    def run(self, other_agents: List['Agent'], total_time_s: float):
        self._running = True
        self.start_time = self.env.start_time
        end_time = self.start_time + total_time_s

        self._log(f"Started at position {tuple(self.position)}")

        state = self.get_state()
        self._communicate_intentions(other_agents, state)

        step = 0
        while self._running and time.time() < end_time:
            # Process incoming messages
            msgs = self.receive_messages()
            for msg in msgs:
                self._log(f"MSG from {msg.sender_id}: {msg.content}")

            self._simple_strategy()
            step += 1

        self._log(f"Stopped. Final points: {self.env.agents[self.agent_id].points}")

    def start(self, other_agents: List['Agent'], total_time_s: float):
        self._thread = threading.Thread(
            target=self.run,
            args=(other_agents, total_time_s),
            daemon=True,
        )
        self._thread.start()

    def stop(self):
        self._running = False
        if self._thread:
            self._thread.join(timeout=2)
