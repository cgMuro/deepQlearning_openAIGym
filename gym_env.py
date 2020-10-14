import gym
import numpy as np
from dqn import DQNAgent

# Config
EPISODES = 5000         # Number of games we want the agent to play
BATCH_SIZE = 20


env = gym.make('CartPole-v0')

initial_state = env.reset().shape[0]
initial_actions = env.action_space.n

agent = DQNAgent(initial_state, initial_actions)



for e in range(EPISODES):
    env.render()

    state = env.reset()
    state = np.reshape(state, [1, initial_state])

    # for time in range(500):

    # Let the agent decide what action to take
    decision = agent.take_action(state)
    if decision == 'explore':
        action = env.action_space.sample()
    else:
        action = decision
    # Perform action
    next_state, reward, done, _ = env.step(action)

    # Memorize experience
    next_state = np.reshape(next_state, [1, initial_state])
    agent.memorize(state=state, action=action, reward=reward, next_state=next_state, done=done)

    # Check if the game is ended
    if done:
        agent.update_target_model()
        # print(f'EPISODE: {e}/{EPISODES}, SCORE: {time}, EPSILON:{agent.epsilon}')
        print(f'EPISODE: {e}/{EPISODES}, EPSILON:{agent.epsilon}')
        break

    # Train agent with past experiences
    if len(agent.memory) > BATCH_SIZE:
        agent.experince_replay(batch_size=BATCH_SIZE)



env.close()
