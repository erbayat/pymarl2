from envs.multiagentenv import MultiAgentEnv
import torch as th
import numpy as np
import random
import pygame
from utils.dict2namedtuple import convert
import os

int_type = np.int16
float_type = np.float32



#sum reward batch 
class UAVEnv(MultiAgentEnv):

    action_labels = {'right': 0, 'down': 1, 'left': 2, 'up': 3, 'weight_increase': 4, 'weight_decrease': 5}
    action_look_to_act = 6

    def __init__(self, batch_size=None, **kwargs):
        print('sasaffs')
        args = kwargs
        if isinstance(args, dict):
            args = convert(args)

        self.args = args
        self.map_index = getattr(args, "map", 1)
        current_dir = os.path.dirname(os.path.abspath(__file__))
        maps_dir = os.path.join(current_dir, 'maps')
        map_file = f'map_{self.map_index}.npy'
        file_path = os.path.join(maps_dir, map_file)
        self.environment_map = np.load(file_path)
        self.event_captured = np.zeros_like(self.environment_map)

        # Configuration for UAVs
        # self.n_uavs = args.n_uavs
        # self.grid_shape = np.asarray(args.world_shape, dtype=int_type)
        # self.x_max, self.y_max = self.grid_shape
        # self.env_max = np.asarray(self.grid_shape, dtype=int_type)
        self.n_uavs = 15
        self.grid_shape = np.array([20,30])
        self.x_max, self.y_max = self.grid_shape
        self.env_max = np.asarray(self.grid_shape, dtype=int_type)

        self.batch_mode = batch_size is not None
        self.batch_size = batch_size if self.batch_mode else 1
        
        # UAV specific attributes
        #self.queue_capacity = args.queue_capacity
        self.queue_capacity = 10
        self.queues = np.zeros((self.n_uavs, self.batch_size), dtype=int_type)
        self.weights = np.ones((self.n_uavs, self.batch_size), dtype=int_type)

        # Base station location and parameters

        #self.base_station_pos = np.asarray(args.base_station_pos, dtype=int_type)
        #self.max_bandwidth = args.max_bandwidth
        self.base_station_pos = np.asarray([10,15], dtype=int_type)
        self.max_bandwidth = 15

        # Actions (move in four directions and weight adjustment)
        self.actions = np.asarray([[0, 1], [1, 0], [0, -1], [-1, 0]], dtype=int_type)
        self.weight_actions = np.array([-1, 1], dtype=int_type)  # reduce, keep, increase
        self.action_names = ["right", "down", "left", "up", "weight_increase", "weight_decrease"]

        # Episode and reward settings
        self.episode_limit = getattr(args, "episode_limit", 100)
        self.time_reward = getattr(args, "reward_time", -0.1)
        #self.capture_event_reward = getattr(args, "reward_event", 5.0)
        self.send_to_base_reward = getattr(args, "reward_base", 10.0)

        # Initialize the internal state
        self.uavs = np.zeros((self.n_uavs, self.batch_size, 2), dtype=int_type)
        self.steps = 0
        self.sum_rewards = 0
        self.reset()

    def reset(self):
        self.queues.fill(0)
        self.weights.fill(1)
        self.event_captured.fill(False)  # Reset event captured status
        self.steps = 0
        self.sum_rewards = 0


        # Place UAVs and events on the grid
        self._place_uavs()
        
        return self.get_obs(), self.get_state()

    def step(self, actions):
        if not self.batch_mode:
            actions = np.expand_dims(np.asarray(actions, dtype=int_type), axis=1)



        reward = np.ones(self.batch_size, dtype=float_type) * self.time_reward
        terminated = [False for _ in range(self.batch_size)]

        # Process weight adjustments
        for u in range(self.n_uavs):
            for b in range(self.batch_size):
                if actions[u, b] > 3:
                    new_weight = self.weights[u, b] + self.weight_actions[actions[u, b]-4]
                    self.weights[u, b] = np.clip(new_weight, 0, 5)
                else:
                    new_pos, collide = self._move_actor(self.uavs[u, b, :], actions[u, b], b, np.asarray([0], dtype=int_type))
                    if not collide:
                        self.uavs[u, b, :] = new_pos
                    if not self.event_captured[self.uavs[u, b, :]] and self.environment_map[self.uavs[u, b, :]]:
                        if self.queues[u, b] < self.queue_capacity:
                            self.queues[u, b] += 1
                            self.event_captured[e, b] = True  # Mark event as captured
        # Distribute bandwidth based on weights
        total_weight = np.sum(self.weights, axis=0)
        bandwidths = (self.weights / total_weight) * self.max_bandwidth

        for u in range(self.n_uavs):
            for b in range(self.batch_size):
                # Calculate the UAV's distance to the base station
                distance_to_base = np.linalg.norm(self.uavs[u, b, :] - self.base_station_pos)

                # Calculate data rate based on the distributed bandwidth and distance
                data_rate = self.calculate_data_rate(bandwidths[u, b], distance_to_base)

                # Simulate sending data to base station
                if self.queues[u, b] > 0:
                    # Calculate how much data is transmitted in this step
                    transmitted_data = min(data_rate, self.queues[u, b])
                    
                    # Decrease the queue by the transmitted data amount
                    self.queues[u, b] -= transmitted_data
                    
                    # Reward for every packet transmitted
                    reward[b] += self.send_to_base_reward * transmitted_data
                    
                    # If the queue is emptied, make sure it's set to zero
                    if self.queues[u, b] <= 0:
                        self.queues[u, b] = 0

        self.sum_rewards += reward[0]
        self.steps += 1

        if self.steps >= self.episode_limit:
            terminated = [True for _ in range(self.batch_size)]

        if self.batch_mode:
            return reward, terminated, {}
        else:
            return reward[0].item(), int(terminated[0]), {}

    def calculate_data_rate(self, bandwidth, distance):
        return bandwidth / (1 + distance)

    def get_obs(self):
        uavs_obs = [self.get_obs_uav(i) for i in range(self.n_uavs)]
        return uavs_obs

    def get_obs_uav(self, uav_id, batch=0):
        obs = np.zeros((self.x_max, self.y_max, 2), dtype=float_type)
        obs[self.uavs[uav_id, batch, 0], self.uavs[uav_id, batch, 1], 0] = 1.0
        for e in range(self.events.shape[0]):
            if not self.event_captured[e, batch]:
                obs[self.events[e, batch, 0], self.events[e, batch, 1], 1] = 1.0
        return obs.flatten()

    def get_state(self):
        return self.grid.copy().reshape(self.grid.size)

    def get_total_actions(self):
        return len(self.action_labels)

    def _place_uavs(self):
        for b in range(self.batch_size):
            for a in range(self.uavs.shape[0]):
                is_free = False
                while not is_free:
                    self.uavs[a, b, 0] = np.random.randint(self.env_max[0])
                    self.uavs[a, b, 1] = np.random.randint(self.env_max[1])

    def _move_actor(self, pos, action, batch, collision_mask):
        new_pos = self._env_bounds(pos + self.actions[action])
        collision = np.sum(self.grid[batch, new_pos[0], new_pos[1], collision_mask]) > 0
        return new_pos, collision

    def _env_bounds(self, positions):
        positions = np.minimum(positions, self.env_max - 1)
        positions = np.maximum(positions, 0)
        return positions

    def close(self):
        print("Closing UAV Environment")
