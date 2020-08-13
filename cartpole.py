import random
import gym
import numpy as np
from collections import deque
from keras.models import Sequential, load_model, save_model
from keras.layers import Dense
from keras.optimizers import Adam


from scores.score_logger import ScoreLogger

ENV_NAME = "CartPole-v1"

GAMMA = 0.95
LEARNING_RATE = 0.001

MEMORY_SIZE = 1000000
BATCH_SIZE = 20

EXPLORATION_MAX = 1.0
EXPLORATION_MIN = 0.01
EXPLORATION_DECAY = 0.995

RESUME = True
SAVE_FREQUENCY = 10


class DQNSolver:

    def __init__(self, observation_space, action_space):
        self.exploration_rate = EXPLORATION_MAX  # epsilon in epsilon-greedy, optimism in the face of uncertainty

        self.action_space = action_space
        self.memory = deque(maxlen=MEMORY_SIZE)

        if RESUME:
            self.model = load_model('saved_model')
        else:
            self.model = Sequential()
            self.model.add(Dense(24, input_shape=(observation_space,), activation="relu"))
            self.model.add(Dense(24, activation="relu"))
            self.model.add(Dense(self.action_space, activation="linear"))
            self.model.compile(loss="mse", optimizer=Adam(lr=LEARNING_RATE))

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def act(self, state):
        if np.random.rand() < self.exploration_rate:
            return random.randrange(self.action_space)  # choose to explore with probability epsilon
        q_values = self.model.predict(state)
        return np.argmax(q_values[0])  # choose the greedy action with probability (1 - epsilon)

    def experience_replay(self):  # the central func of Deep-Q Net
        if len(self.memory) < BATCH_SIZE:  # stop experience if don't have enough experience
            return
        batch = random.sample(self.memory, BATCH_SIZE)  # sample mini-batch of transitions (s, a, r, s')
        for state, action, reward, state_next, terminal in batch:
            q_update = reward  # if terminal
            if not terminal:
                q_update = (reward + GAMMA * np.amax(self.model.predict(state_next)[0]))  # q-target, not fixed here
            q_values = self.model.predict(state)
            q_values[0][action] = q_update
            self.model.fit(state, q_values, verbose=0)  # optimize MSE(q-network, q-target)
        self.exploration_rate *= EXPLORATION_DECAY
        self.exploration_rate = max(EXPLORATION_MIN, self.exploration_rate)


def cartpole():
    env = gym.make(ENV_NAME)
    score_logger = ScoreLogger(ENV_NAME)
    observation_space = env.observation_space.shape[0]
    action_space = env.action_space.n
    dqn_solver = DQNSolver(observation_space, action_space)
    run = 0
    while True:
        run += 1
        state = env.reset()
        state = np.reshape(state, [1, observation_space])  # reshape to row array
        step = 0
        while True:  # run the Deep-Q Net
            step += 1
            env.render()
            action = dqn_solver.act(state)  # make action according to e-greedy
            state_next, reward, terminal, info = env.step(action)  # observe SARS from env
            reward = reward if not terminal else -reward  # ? negative
            state_next = np.reshape(state_next, [1, observation_space])
            dqn_solver.remember(state, action, reward, state_next, terminal)  # store transition
            state = state_next  # advance to next state
            if terminal:
                print("Run: " + str(run) + ", exploration: " + str(dqn_solver.exploration_rate) + ", score: " + str(step))
                score_logger.add_score(step, run)
                break
            dqn_solver.experience_replay()
        if run % SAVE_FREQUENCY == 0:
            save_model(dqn_solver.model, "saved_model")  # save the model parameters every once in a while


if __name__ == "__main__":
    cartpole()
