from typing import Optional, Dict, List, Tuple
from agent import Agent, Task, Message, OUTSOURCE_THRESHOLD
from environment import Environment, Direction
import time


BID_WINDOW = 0.6   # seconds to collect bids
MIN_REWARD = 5
MAX_REWARD = 30


class AgentCNP(Agent):
    """
    Agent using Contract Net Protocol for negotiation.
    """

    def __init__(self, agent_id, color, start_pos, env):
        super().__init__(agent_id, color, start_pos, env)

        self._emitted_cfp_keys: set       = set()
        self._open_cfps:        Dict      = {}   # task_id -> cfp info + bids
        self._awarded_to:       Dict[str,str] = {}

    # ── Hook: outsource task if too far ──────────────────────

    def _negotiate_in_plan(self, task: Task, dist: int,
                           state: dict, other_agents: list) -> bool:
        """
        If the task distance exceeds threshold, emit a CFP.
        Returns True = outsourced (skip from own plan).
        """
        cfp_key = (task.tile_pos, task.hole_pos)
        if dist > OUTSOURCE_THRESHOLD and other_agents \
                and cfp_key not in self._emitted_cfp_keys:
            self._emitted_cfp_keys.add(cfp_key)
            reward = min(MAX_REWARD, max(MIN_REWARD, dist * 2))
            self._issue_cfp(task, reward, other_agents)
            return True   # don't add to own plan
        return False

    # ── Hook: resolve CFPs each iteration ────────────────────

    def _pre_execute_hook(self, state: dict, other_agents: list):
        self._resolve_cfps(other_agents, state)

    # ── Hook: pay reward after completing outsourced task ────

    def _on_task_complete(self, task):
        if task and task.outsourced and task.requester:
            req = self._find_agent(task.requester)
            if req:
                self._log(f"Paying {task.reward} pts to {task.requester}.", "NEG")
                self.transfer_points(req.agent_id, task.reward)

    # ── Hook: reset CFP state on replan ──────────────────────

    def _on_invalidate(self):
        self._emitted_cfp_keys = set()

    # ── CFP emission ──────────────────────────────────────────

    def _issue_cfp(self, task: Task, reward: int, agents: list):
        task_id = (f"{self.agent_id}_{task.hole_pos}"
                   f"_{task.tile_pos}_{time.time():.3f}")
        cfp_msg = {
            "type":       "cfp",
            "task_id":    task_id,
            "tile_pos":   list(task.tile_pos),
            "tile_color": task.tile_color,
            "hole_pos":   list(task.hole_pos),
            "adj_pos":    list(task.adj_pos),
            "use_dir":    task.use_dir.value,
            "reward":     reward,
            "requester":  self.agent_id,
        }
        self._log(
            f"CFP: outsourcing pick {task.tile_color} at {task.tile_pos} "
            f"-> hole {task.hole_pos} for {reward} pts", "NEG")

        for ag in agents:
            if ag.agent_id != self.agent_id:
                self.send_message(ag, cfp_msg)

        self._open_cfps[task_id] = {
            "task":     task,
            "reward":   reward,
            "bids":     {},
            "deadline": time.time() + BID_WINDOW,
            "awarded":  False,
        }

    # ── Bid evaluation (bidder side) ──────────────────────────

    def _evaluate_bid(self, cfp: dict, state: dict) -> Optional[int]:
        tile_pos = tuple(cfp["tile_pos"])
        adj_pos  = tuple(cfp["adj_pos"])
        pos      = tuple(self.position)

        p1 = self._bfs(pos, tile_pos, state)
        p2 = self._bfs(tile_pos, adj_pos, state)
        if not p1 or not p2:
            return None
        return len(p1) + len(p2)

    def _handle_cfp(self, msg: Message, state: dict):
        c = msg.content
        task_id = c["task_id"]

        if self.carried_tile is not None:
            self._send_refuse(msg, task_id)
            return

        cost = self._evaluate_bid(c, state)
        if cost is None:
            self._send_refuse(msg, task_id)
            return

        reward = c["reward"]
        if reward < cost // 2:
            self._log(f"CFP {task_id[:14]}: reward {reward} too low (cost {cost}). Refuse.", "NEG")
            self._send_refuse(msg, task_id)
            return

        req = self._find_agent(msg.sender_id)
        if req:
            self.send_message(req, {
                "type":    "bid",
                "task_id": task_id,
                "bidder":  self.agent_id,
                "cost":    cost,
            })
            self._log(f"BID on {task_id[:14]}: cost={cost}, reward={reward}", "NEG")

    def _send_refuse(self, msg: Message, task_id: str):
        req = self._find_agent(msg.sender_id)
        if req:
            self.send_message(req, {
                "type":    "refuse",
                "task_id": task_id,
                "bidder":  self.agent_id,
            })

    # ── Resolve expired CFPs (requester side) ─────────────────

    def _resolve_cfps(self, agents: list, state: dict):
        now     = time.time()
        expired = [tid for tid, c in self._open_cfps.items()
                   if now >= c["deadline"] and not c["awarded"]]

        for task_id in expired:
            cfp = self._open_cfps[task_id]
            bids = cfp["bids"]

            if not bids:
                self._log(f"CFP {task_id[:14]}: no bids. Adding to own plan.", "NEG")
                self._plan.append(cfp["task"])
                cfp["awarded"] = True
                continue

            winner_id, winner_cost = min(bids.items(), key=lambda x: x[1])
            winner = self._find_agent(winner_id)

            self._log(
                f"CFP {task_id[:14]}: award to {winner_id} "
                f"(cost={winner_cost}, reward={cfp['reward']})", "NEG")

            if winner:
                t = cfp["task"]
                self.send_message(winner, {
                    "type":       "award",
                    "task_id":    task_id,
                    "tile_pos":   list(t.tile_pos),
                    "tile_color": t.tile_color,
                    "hole_pos":   list(t.hole_pos),
                    "adj_pos":    list(t.adj_pos),
                    "use_dir":    t.use_dir.value,
                    "reward":     cfp["reward"],
                    "requester":  self.agent_id,
                })

            for ag_id in bids:
                if ag_id != winner_id:
                    loser = self._find_agent(ag_id)
                    if loser:
                        self.send_message(loser, {"type": "reject", "task_id": task_id})

            cfp["awarded"] = True

    def _handle_award(self, msg: Message):
        c = msg.content
        use_dir = next(d for d in Direction if d.value == c["use_dir"])
        task = Task(
            tile_pos   = tuple(c["tile_pos"]),
            tile_color = c["tile_color"],
            hole_pos   = tuple(c["hole_pos"]),
            adj_pos    = tuple(c["adj_pos"]),
            use_dir    = use_dir,
            outsourced = True,
            reward     = c["reward"],
            requester  = c["requester"],
        )
        self._log(f"AWARD: {task.describe()}", "NEG")
        self._plan.insert(0, task)
        self._step_queue = []

    # ── Message handler ───────────────────────────────────────

    def _handle_negotiation_message(self, msg: Message, state: dict):
        t = msg.content.get("type", "")
        if t == "cfp":
            self._handle_cfp(msg, state)
        elif t == "bid":
            task_id = msg.content["task_id"]
            if task_id in self._open_cfps:
                self._open_cfps[task_id]["bids"][msg.content["bidder"]] = msg.content["cost"]
                self._log(f"BID from {msg.content['bidder']}: "
                          f"cost={msg.content['cost']} on {task_id[:14]}", "NEG")
        elif t == "refuse":
            self._log(f"REFUSE from {msg.content['bidder']} on {msg.content['task_id'][:14]}", "NEG")
        elif t == "award":
            self._handle_award(msg)
        elif t == "reject":
            self._log(f"REJECT: lost contract {msg.content['task_id'][:14]}", "NEG")
        else:
            self._log(f"Unknown MSG from {msg.sender_id}: {msg.content}", "AGT")