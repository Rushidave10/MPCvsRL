import control as ct
import matplotlib.pyplot as plt
import numpy as np
import control.optimal as opt
import gymnasium as gym
from gymnasium.wrappers import FlattenObservation

SIZE = 5
gym.register("GridWorldEnv-v0", entry_point="GridWorldEnv:GridWorldEnv")
env_ = gym.make("GridWorldEnv-v0", size=SIZE)

env = FlattenObservation(env_)

A = np.eye(2)
B = A
C = np.eye(2)

Ts = 0.25

x0 = np.array([0, 0])
xf = np.array([4, 4])

sys = ct.ss(A, B, C, 0, Ts)

Q = np.diag([1, 1])
R = np.diag([1, 1])
S = np.diag([2, 2])

in_constraints = opt.input_range_constraint(sys, [-1, -1], [1, 1])
term_constraints = opt.state_range_constraint(sys, xf, xf)
state_constraints = opt.state_range_constraint(sys, [0, 0], [SIZE - 1, SIZE - 1])
all_constraints = [in_constraints, state_constraints]

traj_cost = opt.quadratic_cost(sys=sys, Q=Q, R=R, x0=x0, u0=np.array([0, 1]))
terminal_cost = opt.quadratic_cost(sys, S, 0, x0=xf)

observation, info = env.reset()
action_to_key = {
    0: np.array([1, 0]),  # Right
    1: np.array([0, 1]),  # Down
    2: np.array([-1, 0]),  # Left
    3: np.array([0, -1])  # Up
}


def map_action_to_key(act):
    for i, j in action_to_key.items():
        if np.array_equal(j, np.array([act])):
            return i


for idx in range(1000):
    timepts = np.linspace(0, 1, 5, endpoint=True)
    result = opt.solve_ocp(sys, timepts, X0=x0, cost=traj_cost,
                           constraints=all_constraints,
                           return_states=True,
                           terminal_constraints=term_constraints
                           )
    _action = int(result.inputs[0][0]), int(result.inputs[1][0])
    action = map_action_to_key(_action)
    observation, reward, done, _, info = env.step(action)
    x0 = observation
    if done:
        observation, info = env.reset()
env.close()
