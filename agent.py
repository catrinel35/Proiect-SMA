import threading
import queue
import time
import random
from typing import Optional, Dict, List, Tuple
from dataclasses import dataclass, field
from enum import Enum, auto
from environment import Environment, Operation, OperationType, Direction


DIRECTIONS = [Direction.NORTH, Direction.SOUTH, Direction.EAST, Direction.WEST]

DIR_DELTA = {
    Direction.NORTH: (0, -1),
    Direction.SOUTH: (0, 1),
    Direction.EAST:  (1, 0),
    Direction.WEST:  (-1, 0),
}

OUTSOURCE_THRESHOLD = 5   # Manhattan distance before considering outsourcing

class StepType(Enum):
    MOVE     = auto()
    PICK     = auto()
    USE_TILE = auto()
    DROP     = auto()


@dataclass
class PlanStep:
    step_type: StepType
    direction: Optional[Direction] = None   # for MOVE and USE_TILE
    color: Optional[str] = None             # for PICK


@dataclass
class Task:
    tile_pos:   Tuple[int, int]
    tile_color: str
    hole_pos:   Tuple[int, int]
    adj_pos:    Tuple[int, int]
    use_dir:    Direction
    steps:      List[PlanStep] = field(default_factory=list)
    outsourced: bool           = False
    reward:     int            = 0
    requester:  str            = ""

    def describe(self) -> str:
        tag = f" [NEGOTIATED +{self.reward}pts from {self.requester}]" \
              if self.outsourced else ""
        return (f"pick {self.tile_color} at {self.tile_pos} "
                f"-> fill hole at {self.hole_pos} from {self.adj_pos}{tag}")


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

        self._plan:           List[Task]     = []
        self._step_queue:     List[PlanStep] = []
        self._reserved_tiles: set            = set()
        self._all_agents:     List           = []

    def _log(self, msg: str, tag: str = "AGT"):
        elapsed = time.time() - self.start_time if self.start_time else 0
        line = f"[{elapsed:.3f}][{tag}][{self.color}] {msg}"
        self.message_log.append(line)
        print(line, flush=True)

    def send_message(self, other: 'Agent', content: dict):
        other.inbox.put(Message(self.agent_id, other.agent_id, content))
        self.env._log(f"[NEG][{self.color} -> {other.color}] "
                      f"{content.get('type','?')}: {content}")

    def receive_messages(self) -> List[Message]:
        msgs = []
        while not self.inbox.empty():
            try:    msgs.append(self.inbox.get_nowait())
            except queue.Empty: break
        return msgs

    def _find_agent(self, agent_id: str) -> Optional['Agent']:
        for ag in self._all_agents:
            if ag.agent_id == agent_id:
                return ag
        return None

    def _do_operation(self, op_type: OperationType, **kwargs) -> dict:
        resp_q = queue.Queue()
        op = Operation(op_type=op_type, agent_id=self.agent_id,
                       args=kwargs, response_queue=resp_q)
        self._op_step += 1
        result = self.env.send_operation(op)
        if result["success"]:
            self._sync_local(op_type, kwargs)
        return result

    def _sync_local(self, op_type: OperationType, args: dict):
        if op_type == OperationType.MOVE:
            dx, dy = DIR_DELTA[args["direction"]]
            self.position[0] += dx
            self.position[1] += dy
        elif op_type == OperationType.PICK:
            self.carried_tile = args["color"]
        elif op_type in (OperationType.DROP, OperationType.USE_TILE):
            self.carried_tile = None

    def move(self, d: Direction) -> bool:
        return self._do_operation(OperationType.MOVE, direction=d)["success"]
    def pick(self, c: str) -> bool:
        return self._do_operation(OperationType.PICK, color=c)["success"]
    def drop_tile(self) -> bool:
        return self._do_operation(OperationType.DROP)["success"]
    def use_tile(self, d: Direction) -> bool:
        return self._do_operation(OperationType.USE_TILE, direction=d)["success"]
    def transfer_points(self, target_id: str, pts: int) -> bool:
        return self._do_operation(OperationType.TRANSFER_POINTS,
                                  agent_id=target_id, points=pts)["success"]
    def get_state(self) -> dict:
        return self.env.get_state(self.agent_id)


    def _bfs(self, start: Tuple[int,int], goal: Tuple[int,int],
             state: dict) -> List[Direction]:
        from collections import deque
        obs   = set(tuple(o) for o in state["obstacles"])
        holes = {tuple(int(c) for c in k.split(",")): v
                 for k, v in state["holes"].items()}
        W, H  = state["width"], state["height"]

        def ok(x, y):
            if not (0 <= x < W and 0 <= y < H): return False
            if (x, y) in obs: return False
            if (x, y) in holes and holes[(x,y)]["depth"] > 0 and (x,y) != goal:
                return False
            return True

        parent = {start: (None, None)}
        q = deque([start])
        while q:
            cur = q.popleft()
            if cur == goal:
                path = []
                while parent[cur][0] is not None:
                    path.append(parent[cur][1]); cur = parent[cur][0]
                path.reverse(); return path
            for d in DIRECTIONS:
                dx, dy = DIR_DELTA[d]; nxt = (cur[0]+dx, cur[1]+dy)
                if nxt not in parent and ok(*nxt):
                    parent[nxt] = (cur, d); q.append(nxt)
        return []

    def _manhattan(self, a: Tuple[int,int], b: Tuple[int,int]) -> int:
        return abs(a[0]-b[0]) + abs(a[1]-b[1])

    def _parse_state(self, state: dict):
        holes = {tuple(int(c) for c in k.split(",")): v
                 for k, v in state["holes"].items()}
        tiles = {tuple(int(c) for c in k.split(",")): v
                 for k, v in state["tiles"].items()}
        obs   = set(tuple(o) for o in state["obstacles"])
        return holes, tiles, obs, state["width"], state["height"]

    def _cell_accessible(self, x, y, obs, holes, W, H) -> bool:
        if not (0 <= x < W and 0 <= y < H): return False
        if (x, y) in obs: return False
        if (x, y) in holes and holes[(x,y)]["depth"] > 0: return False
        return True

    def _valid_adjacent(self, hole_pos, obs, holes, W, H) -> list:
        result = []
        for d in DIRECTIONS:
            dx, dy = DIR_DELTA[d]
            adj = (hole_pos[0]-dx, hole_pos[1]-dy)
            if self._cell_accessible(*adj, obs, holes, W, H):
                result.append((adj, d))
        return result

    def _build_task_steps(self, task: Task, state: dict) -> List[PlanStep]:
        steps = []
        pos   = tuple(self.position)
        for d in self._bfs(pos, task.tile_pos, state):
            steps.append(PlanStep(StepType.MOVE, direction=d))
        steps.append(PlanStep(StepType.PICK, color=task.tile_color))
        for d in self._bfs(task.tile_pos, task.adj_pos, state):
            steps.append(PlanStep(StepType.MOVE, direction=d))
        steps.append(PlanStep(StepType.USE_TILE, direction=task.use_dir))
        return steps

    def _build_plan(self, state: dict, other_agents: List['Agent']) -> List[Task]:
        holes, tiles, obs, W, H = self._parse_state(state)

        my_holes = sorted(
            [(p, h) for p, h in holes.items()
             if h["color"] == self.color and h["depth"] > 0],
            key=lambda x: x[1]["depth"], reverse=True
        )

        if not my_holes:
            self._log("No holes to fill. Empty plan.", "PLN")
            return []

        reserved_by_others = set()
        for ag in other_agents:
            if ag.agent_id != self.agent_id:
                reserved_by_others.update(ag._reserved_tiles)

        my_tiles_expanded = []
        for p, cs in tiles.items():
            if p in reserved_by_others: continue
            for _ in range(cs.count(self.color)):
                my_tiles_expanded.append(p)

        pos         = tuple(self.position)
        plan        = []
        used_counts = {}

        for hole_pos, hole_info in my_holes:
            adjs = self._valid_adjacent(hole_pos, obs, holes, W, H)
            if not adjs:
                self._log(f"Hole {hole_pos} has no adjacent cells, skipping.", "PLN")
                continue

            for _ in range(hole_info["depth"]):
                available = [p for p in set(my_tiles_expanded)
                             if my_tiles_expanded.count(p) > used_counts.get(p, 0)]
                if not available:
                    self._log(f"No more tiles for hole {hole_pos}.", "PLN")
                    break

                sim_pos   = plan[-1].adj_pos if plan else pos
                best_tile = min(available,
                    key=lambda p: self._manhattan(p, sim_pos))
                best_adj, best_dir = min(adjs,
                    key=lambda a: self._manhattan(a[0], best_tile))

                dist = self._manhattan(pos, best_tile) + \
                       self._manhattan(best_tile, best_adj)

                task = Task(
                    tile_pos=best_tile, tile_color=self.color,
                    hole_pos=hole_pos, adj_pos=best_adj, use_dir=best_dir,
                )

                outsourced = self._negotiate_in_plan(
                    task, dist, state, other_agents)

                if not outsourced:
                    plan.append(task)

                used_counts[best_tile] = used_counts.get(best_tile, 0) + 1

        self._reserved_tiles = set(used_counts.keys())

        self._log(f"New plan: {len(plan)} own task(s).", "PLN")
        for i, t in enumerate(plan):
            self._log(f"  Task {i+1}: {t.describe()}", "PLN")

        for ag in other_agents:
            if ag.agent_id != self.agent_id:
                self.send_message(ag, {
                    "type": "plan_announcement",
                    "agent": self.color,
                    "tasks": [t.describe() for t in plan],
                    "reserved_tiles": [list(p) for p in used_counts.keys()],
                })

        return plan

    def _plan_still_valid(self, state: dict) -> bool:
        if not self._plan: return False
        task = self._plan[0]
        holes, tiles, _, _, _ = self._parse_state(state)
        tile_ok = (self.carried_tile == task.tile_color or
                   (task.tile_pos in tiles and
                    task.tile_color in tiles[task.tile_pos]))
        hole_ok = (task.hole_pos in holes and
                   holes[task.hole_pos]["depth"] > 0)
        return tile_ok and hole_ok

    def _execute_next_step(self, state: dict):
        if not self._step_queue: return
        step = self._step_queue[0]

        if step.step_type == StepType.MOVE:
            if self.move(step.direction):
                self._step_queue.pop(0)
            else:
                self._log("Move blocked. Replanning.", "PLN")
                self._invalidate_plan()

        elif step.step_type == StepType.PICK:
            if self.pick(step.color):
                self._log(f"Picked {step.color} tile successfully.", "PLN")
                self._step_queue.pop(0)
            else:
                self._log("Pick failed (tile gone). Replanning.", "PLN")
                self._invalidate_plan()

        elif step.step_type == StepType.USE_TILE:
            if self.use_tile(step.direction):
                task = self._plan[0] if self._plan else None
                self._log("Hole filled successfully!", "PLN")
                self._step_queue.pop(0)
                self._on_task_complete(task)
                if self._plan: self._plan.pop(0)
                self._step_queue = []
            else:
                self._log("UseTile failed. Replanning.", "PLN")
                self._invalidate_plan()

        elif step.step_type == StepType.DROP:
            self.drop_tile(); self._step_queue.pop(0)

    def _invalidate_plan(self):
        self._plan           = []
        self._step_queue     = []
        self._reserved_tiles = set()
        self._on_invalidate()
        if self.carried_tile:
            self.drop_tile()


    def _negotiate_in_plan(self, task: Task, dist: int,
                           state: dict,
                           other_agents: List['Agent']) -> bool:
        return False

    def _handle_negotiation_message(self, msg: Message, state: dict):
        self._log(f"MSG from {msg.sender_id}: {msg.content}", "AGT")

    def _pre_execute_hook(self, state: dict, other_agents: List['Agent']):
        pass

    def _on_task_complete(self, task: Optional[Task]):
        pass

    def _on_invalidate(self):
        pass


    def _communicate_intentions(self, other_agents: List['Agent'], state: dict):
        holes = {tuple(int(x) for x in k.split(",")): v
                 for k, v in state["holes"].items()}
        my_holes = [p for p, h in holes.items()
                    if h["color"] == self.color and h["depth"] > 0]
        if my_holes:
            for other in other_agents:
                if other.agent_id != self.agent_id:
                    self.send_message(other, {
                        "type":   "intention",
                        "agent":  self.color,
                        "action": f"heading to fill hole at {my_holes[0]}",
                    })

    def _handle_message(self, msg: Message, state: dict,
                        other_agents: List['Agent']):
        t = msg.content.get("type", "")
        if t == "plan_announcement":
            self._log(f"Received plan from {msg.sender_id}: "
                      f"{len(msg.content.get('tasks',[]))} task(s).", "NEG")
        elif t == "intention":
            self._log(f"Intention from {msg.sender_id}: "
                      f"{msg.content.get('action')}", "NEG")
        else:
            self._handle_negotiation_message(msg, state)


    def run(self, other_agents: List['Agent'], total_time_s: float):
        self._running    = True
        self.start_time  = self.env.start_time
        self._all_agents = other_agents
        end_time         = self.start_time + total_time_s

        self._log(f"Started at position {tuple(self.position)}")

        state = self.get_state()
        self._communicate_intentions(other_agents, state)

        while self._running and time.time() < end_time:

            state = self.get_state()

            for msg in self.receive_messages():
                self._handle_message(msg, state, other_agents)

            # Subclass hook (e.g. resolve CFPs, check pending offers)
            self._pre_execute_hook(state, other_agents)

            if not self._plan or not self._plan_still_valid(state):
                if self._plan:
                    self._log("Plan invalid. Replanning.", "PLN")
                self._plan       = self._build_plan(state, other_agents)
                self._step_queue = []

            if self._plan and not self._step_queue:
                task             = self._plan[0]
                task.steps       = self._build_task_steps(task, state)
                self._step_queue = list(task.steps)
                self._log(f"Executing task: {task.describe()} "
                          f"({len(self._step_queue)} steps)", "PLN")

            if self._step_queue:
                self._execute_next_step(state)
            else:
                time.sleep(0.05)

        self._log(f"Stopped. Final points: "
                  f"{self.env.agents[self.agent_id].points}")

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