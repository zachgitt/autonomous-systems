from maze import Maze
from evaluation import *
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error


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
    Q = np.zeros(shape=(env.snum, env.anum))
    pi = np.zeros(env.snum, dtype=int)
    for state_in in range(env.snum):
        action_max = -1
        sum_max = -1
        for action in range(env.anum):
            reward1, state_out1, _ = env1.step(state_in, action)
            reward2, state_out2, _ = env2.step(state_in, action)
            sum = 0.9 * (reward1 + gamma * V[state_out1]) + \
                  0.1 * (reward2 + gamma * V[state_out2])

            # Store all sums in Q
            Q[state_in][action] = sum

            # Save best action
            if (sum > sum_max):
                sum_max = sum
                action_max = action

        # Save optimal action for the state
        pi[state_in] = action_max

    # Save optimal Q values
    np.save('q.npy', Q)

    # Print optimal path (assuming no slippage)
    state = 0
    while not is_goal_state(state, env):
        action = pi[state]
        _, state, _ = env1.step(state, action)
        env.plot(state, action)

    # Save Q-plots
    plots = []
    alphas = [.96]
    epsilons = [.75,.8,.85,.9,.95]
    num_episodes = 5000
    step_size = 50

    # Q learning with different learing rates
    for alpha in alphas:
        for epsilon in epsilons:

            # Q learning
            eval_steps, eval_reward = [], []
            Q_table = np.zeros(shape=(env.snum, env.anum))
            episode = 0
            rmse = []
            while episode < num_episodes:

                # Reset env
                state = env.reset()
                done = False
                while not done:

                    # Take step
                    action = get_action_egreedy(Q_table[state], epsilon, env.anum)
                    reward, next_state, done = env.step(state, action)

                    # Update Q-table
                    next_action = np.argmax(Q_table[next_state])
                    next_q = reward + gamma * Q_table[next_state][next_action]
                    Q_table[state][action] += alpha * (next_q - Q_table[state][action])
                    state = next_state

                    # Plot Q vs Q* RMSE
                    rmse.append(mean_squared_error(Q, Q_table))

                # Evaluate Q table
                if episode % step_size == 0:
                    avg_step, avg_reward = evaluation(Maze(), Q_table, step_bound=100, num_itr=10)
                    eval_steps.append(avg_step)
                    eval_reward.append(avg_reward)
                    print('Episode: ' + str(episode) + ' Avg-Step: ' + str(avg_step) + ' Avg-Rew: ' + str(avg_reward))

                episode += 1

            # Save eval and rmse
            plots.append((eval_steps, eval_reward, rmse))


    # Plot avg steps
    idx = 0
    plt.plot([], [], ' ', label='LR=' + str(alphas[0]))
    for alpha in alphas:
        for epsilon in epsilons:
            plt.plot(np.arange(0, num_episodes, step_size), plots[idx][0], label='Epsilon=' + str(epsilon)) #+ ' Eps=' + str(epsilon))
            idx += 1
    plt.xlabel('Episode')
    plt.ylabel('Avg Steps')
    plt.legend()
    plt.savefig('avg_steps.png')
    plt.show()

    # Plot avg reward
    idx = 0
    plt.plot([], [], ' ', label='LR=' + str(alphas[0]))
    for alpha in alphas:
        for epsilon in epsilons:
            plt.plot(np.arange(0, num_episodes, step_size), plots[idx][1], label='Epsilon=' + str(epsilon)) #+ ' Eps=' + str(epsilon))
            idx += 1
    plt.xlabel('Episode')
    plt.ylabel('Avg Reward')
    plt.legend()
    plt.savefig('avg_rewards.png')
    plt.show()

    # Plot rmse
    idx = 0
    plt.plot([], [], ' ', label='LR=' + str(alphas[0]))
    for alpha in alphas:
        for epsilon in epsilons:
            plt.plot(np.arange(0, len(plots[idx][2])), plots[idx][2], label='Epsilon=' + str(epsilon)) #+ ' Eps=' + str(epsilon))
            idx += 1
    plt.xlabel('t')
    plt.ylabel('RMSE')
    plt.legend()
    plt.savefig('rmses.png')
    plt.show()


if __name__ == '__main__':
    main()
