import control as ct
import control.optimal as opt
from GridWorld.DDPG.ddpg_backend import Agent
import numpy as np
import gymnasium as gym
import gym as old_gym
from gymnasium.wrappers import FlattenObservation

SIZE = 10
gym.register("GridWorldEnv-v0", entry_point="CGridWorldEnv:GridWorldEnv")
env_ = gym.make("GridWorldEnv-v0", size=SIZE, random_start=True, random_target=False)
env = FlattenObservation(env_)

#  RL agent declared
agent = Agent(alpha=0.000025, beta=0.00025, input_dims=[2], tau=0.001, env=env,
              batch_size=64, layer1_size=32, layer2_size=32, n_actions=2)

#  MPC agent declared
MPC = True
A = np.eye(2)
B = A
C = np.eye(2)
eps = 0.7
Ts = 0.25

sys = ct.ss(A, B, C, 0, Ts)
x0 = np.array([0, 0])

Q = np.diag([1, 1])
R = np.diag([1, 1])
S = np.diag([100, 100])

in_constraints = opt.input_range_constraint(sys, [-1, -1], [1, 1])
state_constraints = opt.state_range_constraint(sys, [0, 0], [SIZE - 1, SIZE - 1])
all_constraints = [in_constraints, state_constraints]
horizon = 5  # Horizon Length
# agent.load_models()
np.random.seed(0)
score_history = []

for i in range(50):
    obs, info = env.reset()
    xf = info["target_location"]
    traj_cost = opt.quadratic_cost(sys=sys, Q=Q, R=None, x0=xf)
    terminal_cost = opt.quadratic_cost(sys, Q=S, R=0, x0=xf)
    done = False
    truncated = False
    score = 0
    k = 0
    while not done:
        eval_pts = np.arange(k, k + horizon) * Ts
        if MPC:
            result = opt.solve_ocp(sys, eval_pts, X0=obs, cost=traj_cost,
                                   constraints=all_constraints,
                                   return_states=True,
                                   terminal_cost=terminal_cost,
                                   )
            _action = result.inputs[0][0], result.inputs[1][0]
            act = _action
        else:
            act = agent.choose_action(obs)

        new_state, reward, done, truncated, info = env.step(act)
        if truncated:
            done = True
        agent.remember(obs, act, reward, new_state, int(done))
        agent.learn()
        score += reward
        k += 1
        MPC = True if eps < np.random.rand(1) and i < 30 else False

        obs = new_state
        env.render()
    score_history.append(score)

    # if i % 25 == 0:
    #    agent.save_models()

    print('episode ', i, 'score %.2f' % score,
          'trailing 100 games avg %.3f' % np.mean(score_history[-100:]))

env.close()
