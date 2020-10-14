import random
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Dense, Activation, Dropout
from tensorflow.keras.optimizers import Adam
import numpy as np
from collections import deque

# General Config
MAX_MEMORY = 2000
BATCH_SIZE = 30


class DQNAgent():

    def __init__(self, state, actions):
        # INPUTS
        self.state = state
        self.actions = actions
        # CONFIG
        self.memory = deque(maxlen=MAX_MEMORY)
        self.alpha = 0.0001         # Learning rate --> determines how much neural network learns at each iteration
        self.gamma = 0.99           # Discount rate --> calculates the future discounted reward
        self.epsilon = 1.0          # Exploration rate --> the rate at which the agent randomly picks the action
        self.epsilon_min = 0.01     # Min eploratio rate --> the minimum rate of exploration by the agent that we want
        self.epsilon_decay = 0.99   # Exploration decay rate --> the rate at which we want the exploration rate to decrease
    
        # Models
        self.core_model = self.create_model()
        self.target_model = self.create_model()
    
    def create_model(self):
        ''' Create the Neural Network '''
        # Init model
        model = Sequential()
        # Add input
        model.add(Dense(24, input_dim=self.state, activation='relu'))
        # Hidden layer
        model.add(Dense(64, activation='relu'))
        model.add(Dropout(0.5))
        # Output layer
        model.add(Dense(self.actions, activation='linear'))
        # Compile
        model.compile(
            optimizer=Adam(learning_rate=self.alpha),
            loss='mse',
            metrics=['mae']
        )
        return model

    def memorize(self, state, action, reward, next_state, done):
        ''' Record past experiences '''
        self.memory.append((state, action, reward, next_state, done))

    def update_target_model(self):
        self.target_model.set_weights(self.core_model.get_weights())

    def take_action(self, state):
        ''' Take an action based on the exploit/explore trade off '''
        # Eplore --> use epsilon to check if you should explore or not
        if np.random.rand() <= self.epsilon:
            # More general approach --> choose random action between all actions
            # return random.randrange(self.actions)
            # OpenAI Gym specific (use env.action_space.sample() inside gym env)
            return 'explore'

        # Exlploit --> Let the Neural Network decide what action to take
        actions = self.core_model.predict(state)

        return np.argmax(actions[0])        # Take the best possible action

    def experince_replay(self, batch_size):
        ''' Train Neural Network with past experiences '''
        # Random batch of size n from the memory
        minibatch = random.sample(self.memory, batch_size)
        
        for state, action, reward, next_state, done in minibatch:
            # We calculate the current Q value for this (state, action)          
            target = self.core_model.predict(state)

            if done:
                new_q = reward
            else:
                new_q = reward + self.gamma * max(self.target_model.predict(next_state)[0])

            # We update the current action Q-value with the new Q-value calculated through the target network
            target[0][action] = new_q
            # Train the core neural network
            self.core_model.fit(state, target, epochs=1)

            # Update the weights on the target_model
            self.update_target_model()

            # Decrease the exploration rate (epsilon) if it is still too high
            if self.epsilon > self.epsilon_min:
                self.epsilon *= self.epsilon_decay


    