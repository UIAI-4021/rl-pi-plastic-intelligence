import gym
import gym_maze
import numpy as np


# Create an environment
env = gym.make("maze-random-10x10-plus-v0")
observation = env.reset()

#defining Q table and hyperparameters
q_table = np.zeros((100, 4))
learning_rate = 0.1
discount_factor = 0.9

#number of wins
k = 0

# Define the maximum number of iterations
NUM_EPISODES = 1000

for episode in range(NUM_EPISODES):
    state = env.reset()

    for t in range(100):

        row = state[0]
        col = state[1]
        state = int(row*10 + col)

        #choosing best action based on our policy
        action = np.argmax(q_table[state])

        next_state, reward, done, info = env.step(action)
        next_max = np.max(q_table[next_state[0] * 10 + next_state[1]])

        # Update Q-value for the current state-action pair
        q_table[state][action] = (1 - learning_rate) * q_table[state][action] + learning_rate * (reward + discount_factor * next_max)
        state = next_state

        if done:
            k += 1
            print(f'Wins : {k}')
            break

        env.render()



# Close the environment
env.close()