
import numpy as np
import pygame

import gymnasium as gym
from gymnasium import spaces


class GridWorldEnv(gym.Env):
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 4}
    Random = True

    def __init__(self, render_mode=None, size=5):
        self.size = size
        self.window_size = 800
        self._step_count = []
        self._visited = []
        self.no_cells = self.size % self.window_size
        self.mesh = np.zeros((self.no_cells, self.no_cells))

        self.observation_space = spaces.Box(0, size - 1, shape=(2,), dtype=int)
        self.action_space = gym.spaces.Box(-1, +1, shape=(2,))

        self._action_to_direction = {
            0: np.array([1, 0]),  # Right
            1: np.array([0, 1]),  # Down
            2: np.array([-1, 0]),  # Left
            3: np.array([0, -1])  # Up
        }

        assert render_mode is None or render_mode in self.metadata["render_modes"]
        self.render_mode = "human"

        self.window = None
        self.clock = None

    def _get_obs(self):
        return self._agent_location

    def _get_info(self):
        return {
            "target_location": self._target_location
        }

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        self._agent_location = np.array([0, 0])

        self._target_location = self._agent_location
        while np.array_equal(self._target_location, np.rint(self._agent_location)):
            self._step_count = []
            self._visited = []
            self.mesh = np.zeros((self.no_cells, self.no_cells))
            if self.Random:
                self._target_location = self.np_random.integers(0, self.size, size=2, dtype=int)
            else:
                self._target_location = np.array([self.size - 1, self.size - 1])

        observation = self._get_obs()
        info = self._get_info()

        if self.render_mode == "human":
            self._render_frame()

        return observation, info

    def step(self, action):
        direction = action
        self._agent_location = np.clip(self._agent_location + direction, a_min=0, a_max=self.size - 1)

        for i in range(self.no_cells):
            for j in range(self.no_cells):
                if (np.rint(self._agent_location) == np.array([i, j])).all():
                    self.mesh[i, j] += 1

        terminated = np.array_equal(np.rint(self._agent_location), self._target_location)
        reward = -np.linalg.norm(self._agent_location - self._target_location, ord=1)
        observation = self._get_obs()
        info = self._get_info()

        if self.render_mode == "human":
            self._render_frame()

        return observation, reward, terminated, False, info

    def render(self):
        if self.render_mode == "rgb_array":
            return self._render_frame()

    def _render_frame(self):
        if self.window is None and self.render_mode == "human":
            pygame.init()
            pygame.display.init()
            self.window = pygame.display.set_mode((self.window_size, self.window_size))

        if self.clock is None and self.render_mode == "human":
            self.clock = pygame.time.Clock()

        canvas = pygame.Surface((self.window_size, self.window_size))

        canvas.fill((255, 255, 255))
        pix_square_size = (self.window_size / self.size)

        for i in range(self.no_cells):
            for j in range(self.no_cells):
                pygame.draw.rect(canvas, (255 - self.mesh[i, j] * 30, 255 - self.mesh[i, j] * 30, 255),
                                 pygame.Rect([pix_square_size * i, pix_square_size * j],
                                             (pix_square_size, pix_square_size)))

        pygame.draw.rect(canvas, (255, 0, 0), pygame.Rect(pix_square_size * self._target_location,
                                                          (pix_square_size, pix_square_size)))

        pygame.draw.circle(canvas, (0, 0, 255), (self._agent_location + 0.5) * pix_square_size, pix_square_size / 3)

        for x in range(self.size + 1):
            pygame.draw.line(canvas, color=0,
                             start_pos=(0, pix_square_size * x),
                             end_pos=(self.window_size, pix_square_size * x),
                             width=3)

            pygame.draw.line(canvas,
                             color=0,
                             start_pos=(pix_square_size * x, 0),
                             end_pos=(pix_square_size * x, self.window_size),
                             width=3)

        if self.render_mode == "human":
            self.window.blit(canvas, canvas.get_rect())
            pygame.event.pump()
            pygame.display.update()

            self.clock.tick(self.metadata["render_fps"])

        else:
            return np.transpose(np.array(pygame.surfarray.pixels3d(canvas)), axes=(1, 0, 2))

    def close(self):
        if self.window is not None:
            pygame.display.quit()
            pygame.quit()
