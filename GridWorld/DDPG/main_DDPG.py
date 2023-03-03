from GridWorld.DDPG.ddpg_backend import Agent
import numpy as np
import gymnasium as gym
from gymnasium.wrappers import FlattenObservation

SIZE = 5
gym.register("GridWorldEnv-v0", entry_point="CGridWorldEnv:GridWorldEnv")
env_ = gym.make("GridWorldEnv-v0", size=SIZE)
env = FlattenObservation(env_)

agent = Agent(alpha=0.000025, beta=0.00025, input_dims=[2], tau=0.001, env=env,
              batch_size=64,  layer1_size=64, layer2_size=64, n_actions=2)

#agent.load_models()
np.random.seed(0)
score_history = []

for i in range(10):
    obs, info = env.reset()
    done = False
    score = 0
    while not done:
        act = agent.choose_action(obs)
        new_state, reward, done, _, info = env.step(act)
        agent.remember(obs, act, reward, new_state, int(done))
        agent.learn()
        score += reward
        obs = new_state
        env.render()
    score_history.append(score)

    # if i % 25 == 0:
    #    agent.save_models()

    print('episode ', i, 'score %.2f' % score,
          'trailing 100 games avg %.3f' % np.mean(score_history[-100:]))
env.close()
# filename = 'gridworld_random_target.png'
# plotLearning(score_history, filename, window=100)
