import control as ct
import matplotlib.pyplot as plt
import numpy as np
import control.optimal as opt
import gymnasium as gym
from gymnasium.wrappers import FlattenObservation


SIZE = 10
gym.register("GridWorld-v0", entry_point="GridWorld:GridWorldEnv")
env = gym.make("GridWorld-v0", size=SIZE)


observation, info = env.reset()

A = np.eye(2)
B = A
C = np.eye(2)

Ts = 0.25

x0 = np.array([0, 0])
xf = info["target_location"]

sys = ct.ss(A, B, C, 0, Ts)

Q = np.diag([1, 1])
R = np.diag([1, 1])
S = np.diag([100, 100])

in_constraints = opt.input_range_constraint(sys, [-1, -1], [1, 1])
state_constraints = opt.state_range_constraint(sys, [0, 0], [SIZE - 1, SIZE - 1])
all_constraints = [in_constraints, state_constraints]

timepts = np.linspace(0, 100, 401, endpoint=True)

for idx in range(len(timepts)):
    horizon = timepts[idx:idx+5]
    traj_cost = opt.quadratic_cost(sys=sys, Q=Q, R=None, x0=xf)
    terminal_cost = opt.quadratic_cost(sys, Q=S, R=0, x0=xf)
    # term_constraints = opt.state_range_constraint(sys, xf, xf)
    result = opt.solve_ocp(sys, horizon, X0=observation, cost=traj_cost,
                           constraints=all_constraints,
                           return_states=True,
                           terminal_cost=terminal_cost,
                           )
    _action = result.inputs[0][0], result.inputs[1][0]
    action = _action
    observation, _, done, _, _ = env.step(action)

    if done:
        observation, info = env.reset()
        xf = info["target_location"]

env.close()
