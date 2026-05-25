from typing import Optional, Dict, List, Tuple
from agent import Agent, Task, Message, OUTSOURCE_THRESHOLD
from environment import Direction
import time


OFFER_TIMEOUT   = 1.0   # seconds to wait for a response to a proposal
MAX_ROUNDS      = 2     # max counteroffer rounds before giving up
MIN_REWARD      = 5
MAX_REWARD      = 40


class AgentOffer(Agent):
    """
    Agent using Offer/Counteroffer negotiation.
    """

    def __init__(self, agent_id, color, start_pos, env):
        super().__init__(agent_id, color, start_pos, env)

        self._pending_proposals: Dict[str, dict] = {}
        self._accepted_tasks:    Dict[str, Task]  = {}
        self._negotiated_keys:   set              = set()

    def _negotiate_in_plan(self, task: Task, dist: int,
                           state: dict, other_agents: list) -> bool:
        neg_key = (task.tile_pos, task.hole_pos)
        if dist <= OUTSOURCE_THRESHOLD or not other_agents \
                or neg_key in self._negotiated_keys:
            return False

        candidates = [ag for ag in other_agents
                      if ag.agent_id != self.agent_id
                      and ag.carried_tile is None]
        if not candidates:
            return False

        target = min(candidates, key=lambda ag: self._manhattan(
            tuple(self.position), tuple(ag.position)))

        reward  = min(MAX_REWARD, max(MIN_REWARD, dist * 2))
        neg_id  = f"{self.agent_id}_{task.hole_pos}_{task.tile_pos}_{time.time():.3f}"

        self._log(
            f"PROPOSE to {target.color}: do pick {task.tile_color} "
            f"at {task.tile_pos} -> hole {task.hole_pos} for {reward} pts", "NEG")

        self.send_message(target, {
            "type":       "propose",
            "neg_id":     neg_id,
            "tile_pos":   list(task.tile_pos),
            "tile_color": task.tile_color,
            "hole_pos":   list(task.hole_pos),
            "adj_pos":    list(task.adj_pos),
            "use_dir":    task.use_dir.value,
            "reward":     reward,
            "proposer":   self.agent_id,
            "round":      0,
        })

        self._pending_proposals[neg_id] = {
            "task":     task,
            "reward":   reward,
            "target":   target.agent_id,
            "deadline": time.time() + OFFER_TIMEOUT,
            "resolved": False,
            "round":    0,
        }
        self._negotiated_keys.add(neg_key)
        return True   # temporarily remove from own plan

    def _pre_execute_hook(self, state: dict, other_agents: list):
        self._check_pending_proposals()

    def _on_task_complete(self, task):
        if task and task.outsourced and task.requester:
            req = self._find_agent(task.requester)
            if req:
                self._log(f"Paying {task.reward} pts to {task.requester}.", "NEG")
                self.transfer_points(req.agent_id, task.reward)

    def _on_invalidate(self):
        self._negotiated_keys = set()

    def _check_pending_proposals(self):
        now = time.time()
        for neg_id, p in list(self._pending_proposals.items()):
            if p["resolved"]: continue
            if now >= p["deadline"]:
                self._log(
                    f"PROPOSE {neg_id[:14]}: timeout, doing task myself.", "NEG")
                self._plan.append(p["task"])
                p["resolved"] = True

    def _handle_propose(self, msg: Message, state: dict):
        c       = msg.content
        neg_id  = c["neg_id"]
        reward  = c["reward"]
        rnd     = c.get("round", 0)

        if self.carried_tile is not None:
            self._send_reject(msg, neg_id, "busy")
            return

        # Evaluate cost
        tile_pos = tuple(c["tile_pos"])
        adj_pos  = tuple(c["adj_pos"])
        pos      = tuple(self.position)

        p1 = self._bfs(pos, tile_pos, state)
        p2 = self._bfs(tile_pos, adj_pos, state)
        if not p1 or not p2:
            self._send_reject(msg, neg_id, "no path")
            return

        cost = len(p1) + len(p2)

        if reward >= cost:
            # Profitable — ACCEPT
            self._accept_proposal(msg, c, cost)

        elif rnd < MAX_ROUNDS:
            # Not enough — COUNTER with desired reward
            desired = min(MAX_REWARD, cost + 2)
            self._send_counter(msg, c, desired, rnd + 1)

        else:
            # Max rounds reached — REJECT
            self._send_reject(msg, neg_id, "max rounds")

    def _accept_proposal(self, msg: Message, c: dict, cost: int):
        use_dir = next(d for d in Direction if d.value == c["use_dir"])
        task = Task(
            tile_pos   = tuple(c["tile_pos"]),
            tile_color = c["tile_color"],
            hole_pos   = tuple(c["hole_pos"]),
            adj_pos    = tuple(c["adj_pos"]),
            use_dir    = use_dir,
            outsourced = True,
            reward     = c["reward"],
            requester  = c["proposer"],
        )
        self._log(
            f"ACCEPT: {task.describe()} (cost={cost})", "NEG")

        proposer = self._find_agent(msg.sender_id)
        if proposer:
            self.send_message(proposer, {
                "type":   "accept",
                "neg_id": c["neg_id"],
            })

        self._plan.insert(0, task)
        self._step_queue = []
        self._accepted_tasks[c["neg_id"]] = task

    def _send_counter(self, msg: Message, c: dict, desired: int, rnd: int):
        self._log(
            f"COUNTER on {c['neg_id'][:14]}: want {desired} pts "
            f"(round {rnd})", "NEG")
        proposer = self._find_agent(msg.sender_id)
        if proposer:
            self.send_message(proposer, {
                "type":       "counter",
                "neg_id":     c["neg_id"],
                "tile_pos":   c["tile_pos"],
                "tile_color": c["tile_color"],
                "hole_pos":   c["hole_pos"],
                "adj_pos":    c["adj_pos"],
                "use_dir":    c["use_dir"],
                "reward":     desired,
                "proposer":   c["proposer"],
                "round":      rnd,
            })

    def _send_reject(self, msg: Message, neg_id: str, reason: str):
        self._log(f"REJECT {neg_id[:14]}: {reason}", "NEG")
        proposer = self._find_agent(msg.sender_id)
        if proposer:
            self.send_message(proposer, {
                "type":   "reject",
                "neg_id": neg_id,
                "reason": reason,
            })

    def _handle_accept(self, msg: Message):
        neg_id = msg.content["neg_id"]
        p = self._pending_proposals.get(neg_id)
        if p and not p["resolved"]:
            self._log(
                f"ACCEPT received from {msg.sender_id} "
                f"on {neg_id[:14]}. Task delegated.", "NEG")
            p["resolved"] = True

    def _handle_counter(self, msg: Message, state: dict):
        c       = msg.content
        neg_id  = c["neg_id"]
        desired = c["reward"]
        rnd     = c.get("round", 1)

        p = self._pending_proposals.get(neg_id)
        if not p or p["resolved"]:
            return

        self._log(
            f"COUNTER from {msg.sender_id}: wants {desired} pts "
            f"(round {rnd})", "NEG")

        # Accept if desired reward is still less than MAX_REWARD
        if desired <= MAX_REWARD:
            self._log(f"Accepting counter offer of {desired} pts.", "NEG")
            p["reward"] = desired
            p["resolved"] = True

            responder = self._find_agent(msg.sender_id)
            if responder:
                self.send_message(responder, {
                    "type":       "propose",
                    "neg_id":     neg_id,
                    "tile_pos":   c["tile_pos"],
                    "tile_color": c["tile_color"],
                    "hole_pos":   c["hole_pos"],
                    "adj_pos":    c["adj_pos"],
                    "use_dir":    c["use_dir"],
                    "reward":     desired,
                    "proposer":   self.agent_id,
                    "round":      MAX_ROUNDS,  # force accept next time
                })
        else:
            self._log(f"Counter too expensive ({desired} pts). Reject.", "NEG")
            p["resolved"] = True
            self._plan.append(p["task"])
            responder = self._find_agent(msg.sender_id)
            if responder:
                self.send_message(responder, {
                    "type":   "reject",
                    "neg_id": neg_id,
                    "reason": "too expensive",
                })

    def _handle_reject(self, msg: Message):
        neg_id = msg.content["neg_id"]
        p = self._pending_proposals.get(neg_id)
        if p and not p["resolved"]:
            self._log(
                f"REJECT from {msg.sender_id}: "
                f"{msg.content.get('reason','')}. Doing task myself.", "NEG")
            p["resolved"] = True
            self._plan.append(p["task"])

    def _handle_negotiation_message(self, msg: Message, state: dict):
        t = msg.content.get("type", "")
        if   t == "propose": self._handle_propose(msg, state)
        elif t == "accept":  self._handle_accept(msg)
        elif t == "counter": self._handle_counter(msg, state)
        elif t == "reject":  self._handle_reject(msg)
        else:
            self._log(f"Unknown MSG from {msg.sender_id}: {msg.content}", "AGT")