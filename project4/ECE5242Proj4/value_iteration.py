from maze import Maze
import numpy as np


def is_goal_state(state, env):
    goal_states = [goal_state for goal_state in range(*env.goal)]
    return state in goal_states


def main():

    # Create environments
    env = Maze()
    env1 = Maze(slippage=0)
    env2 = Maze(slippage=1)

    # Calculate V
    gamma = 0.9
    theta = 0.001
    delta = np.inf
    V = np.zeros(env.snum)
    while delta > theta:

        delta_new = 0
        for state_in in range(env.snum):
            # Save old v
            v_old = V[state_in]

            # Save maximum v
            sum_max = 0
            for action in range(env.anum):
                # Calculate possible outcomes
                reward1, state_out1, _ = env1.step(state_in, action)
                reward2, state_out2, _ = env2.step(state_in, action)
                sum = 0.9 * (reward1 + gamma * V[state_out1]) + \
                      0.1 * (reward2 + gamma * V[state_out2])
                sum_max = max(sum_max, sum)

            V[state_in] = sum_max
            delta_new = max(delta_new, abs(v_old - V[state_in]))

        # Update delta
        print(delta)
        delta = delta_new

    # Calculate optimal policy
    pi = np.zeros(env.snum, dtype=int)
    for state_in in range(env.snum):
        action_max = -1
        sum_max = -1
        for action in range(env.anum):
            reward1, state_out1, _ = env1.step(state_in, action)
            reward2, state_out2, _ = env2.step(state_in, action)

            # Save best action
            sum = 0.9 * (reward1 + gamma * V[state_out1]) + \
                  0.1 * (reward2 + gamma * V[state_out2])
            if (sum > sum_max):
                sum_max = sum
                action_max = action

        # Save optimal action for the state
        pi[state_in] = action_max

    # Print optimal path (assuming env1)
    state = 0
    while not is_goal_state(state, env):
        action = pi[state]
        _, state, _ = env1.step(state, action)
        env.plot(state, action)


if __name__ == '__main__':
    main()
