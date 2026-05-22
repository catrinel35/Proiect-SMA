import sys
import time
from parser import parse_input
from environment import Environment
from display import display_state
from agent_cnp   import AgentCNP
from agent_offer import AgentOffer

DISPLAY_INTERVAL_MS = 2000


def main():
    input_file = sys.argv[1] if len(sys.argv) >= 2 else "system.txt"
    agent_mode = sys.argv[2] if len(sys.argv) >= 3 else "mixed"

    print(f"[MAIN] Loading configuration from: {input_file}")
    try:
        config = parse_input(input_file)
    except FileNotFoundError:
        print(f"[ERROR] File not found: {input_file}")
        sys.exit(1)

    print(f"[MAIN] Config: {config.N} agents, grid {config.W}x{config.H}, "
          f"t={config.t}ms, T={config.T}ms, mode={agent_mode}")

    env = Environment(width=config.W, height=config.H, operation_time_ms=config.t)

    for obs in config.obstacles:
        env.add_obstacle(obs[0], obs[1])
    for count, color, x, y in config.tiles:
        env.add_tile(x, y, color, count)
    for depth, color, x, y in config.holes:
        env.add_hole(x, y, color, depth)

    agents = []
    for i in range(config.N):
        color = config.colors[i]
        pos   = config.agent_positions[i]

        if agent_mode == "cnp":
            AgentClass = AgentCNP
        elif agent_mode == "occ":
            AgentClass = AgentOffer
        else:
            # mixed: even indices -> CNP, odd -> OCC (they compete)
            AgentClass = AgentCNP if i % 2 == 0 else AgentOffer

        agent = AgentClass(agent_id=color, color=color,
                           start_pos=pos, env=env)
        env.add_agent(color, color, pos[0], pos[1])
        agents.append(agent)

        kind = "CNP" if AgentClass is AgentCNP else "OCC"
        print(f"[MAIN] Agent '{color}' ({kind}) created at {pos}")

    env.start_time = time.time()
    env.start()
    print("[MAIN] Environment started.")

    display_state(env, [])

    total_time_s = config.T / 1000.0
    for agent in agents:
        agent.start(agents, total_time_s)
    print(f"[MAIN] All {config.N} agents started.")

    end_time           = env.start_time + total_time_s
    display_interval_s = DISPLAY_INTERVAL_MS / 1000.0

    try:
        while time.time() < end_time:
            time.sleep(display_interval_s)
            display_state(env, env.get_log_buffer())
    except KeyboardInterrupt:
        print("\n[MAIN] Interrupted.")

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