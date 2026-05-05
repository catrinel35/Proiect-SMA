import sys
import time

from parser import parse_input
from environment import Environment
from agent import Agent
from display import display_state

DISPLAY_INTERVAL_MS = 2000


def main():
    if len(sys.argv) >= 2:
        input_file = sys.argv[1]
    else:
        input_file = "tests/system.txt"

    print(f"[MAIN] Loading configuration from: {input_file}")
    try:
        config = parse_input(input_file)
    except FileNotFoundError:
        print(f"[ERROR] Input file not found: {input_file}")
        sys.exit(1)

    print(f"[MAIN] Config: {config.N} agents, grid {config.W}x{config.H}, "
          f"t={config.t}ms, T={config.T}ms")

    env = Environment(width=config.W, height=config.H, operation_time_ms=config.t)

    for obs in config.obstacles:
        env.add_obstacle(obs[0], obs[1])

    for count, color, x, y in config.tiles:
        env.add_tile(x, y, color, count)

    for depth, color, x, y in config.holes:
        env.add_hole(x, y, color, depth)

    agents: list[Agent] = []
    for i in range(config.N):
        color = config.colors[i]
        pos = config.agent_positions[i]
        agent = Agent(agent_id=color, color=color, start_pos=pos, env=env)
        env.add_agent(color, color, pos[0], pos[1])
        agents.append(agent)
        print(f"[MAIN] Agent '{color}' created at position {pos}")

    env.start_time = time.time()
    env.start()
    print("[MAIN] Environment started.")

    display_state(env, [])

    total_time_s = config.T / 1000.0
    for agent in agents:
        agent.start(agents, total_time_s)
    print(f"[MAIN] All {config.N} agents started.")

    end_time = env.start_time + total_time_s
    display_interval_s = DISPLAY_INTERVAL_MS / 1000.0

    try:
        while time.time() < end_time:
            time.sleep(display_interval_s)
            log_buf = env.get_log_buffer()
            display_state(env, log_buf)
    except KeyboardInterrupt:
        print("\n[MAIN] Interrupted by user.")

    for agent in agents:
        agent.stop()
    env.stop()

    print("\n" + "=" * 60)
    print("SIMULATION ENDED")
    print("=" * 60)
    display_state(env, env.get_log_buffer())

    print("\nFinal Scores:")
    for aid, a in env.agents.items():
        print(f"  {a.color.capitalize()}: {a.points} points")


if __name__ == "__main__":
    main()
