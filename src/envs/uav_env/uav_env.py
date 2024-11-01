from envs.multiagentenv import MultiAgentEnv
import torch as th
import numpy as np
import random
import pygame
from utils.dict2namedtuple import convert
import os
from envs.uav_env.generate_maps import *
int_type = np.int16
float_type = np.float32

HEIGHT = 100

#sum reward batch 
class UAVEnv(MultiAgentEnv):



    def __init__(self, batch_size=None, **kwargs):
        # Unpack arguments from sacred
        args = kwargs
        if isinstance(args, dict):
            args = convert(args)
        self.args = args
        # Define the agents
        self.n_uavs = 15
        self.n_feats = 2
        self.grid_shape = np.array([20,30])
        self.x_max, self.y_max = self.grid_shape
        self.env_max = np.asarray(self.grid_shape, dtype=int_type)
        self.state_size = int(self.x_max * self.y_max * self.n_feats + self.n_uavs*2)
        # Define the internal state
        self.map_index = getattr(args, "map", 1)
        #current_dir = os.path.dirname(os.path.abspath(__file__))
        #maps_dir = os.path.join(current_dir, 'maps')
        #map_file = f'map_{self.map_index}.npy'
        #file_path = os.path.join(maps_dir, map_file)
        #self.environment_map = np.load(file_path)
        rows, cols = self.grid_shape[0], self.grid_shape[1]
        ratio_of_ones = 0.7  # Example ratio
        self.environment_map = np.zeros((self.batch_size, self.x_max, self.y_max), dtype=float_type)
        self.steps = 0
        action_labels = {'right': 0, 'down': 1, 'left': 2, 'up': 3, 'weight_increase': 4, 'weight_decrease': 5}
        self.action_move = 4
        self.n_actions = len(action_labels)

        self.batch_mode = batch_size is not None
        self.batch_size = batch_size if self.batch_mode else 1
        self.grid = np.zeros((self.batch_size, self.x_max, self.y_max, self.n_feats), dtype=float_type)
        for i in range(self.grid.shape[0]):
            self.environment_map[i,:,:] =  generate_binary_map_with_ratio(rows, cols, ratio_of_ones)
            self.grid[i, :, :, 0] = self.environment_map[i,:,:]

        # 0=events, 1=uavs, 2=queues, 3=weights,

        # UAV specific attributes
        #self.queue_capacity = args.queue_capacity
        self.queue_capacity = 10
        self.float_queues = np.zeros((self.n_uavs, self.batch_size), dtype=float_type)
        self.int_queues = np.zeros((self.n_uavs, self.batch_size), dtype=int_type)
        self.weights = np.ones((self.n_uavs, self.batch_size), dtype=int_type)


        # Base station location and parameters
        self.base_station_pos = np.asarray([10,15], dtype=int_type)
        self.max_bandwidth = 15

        # Actions (move in four directions and weight adjustment)
        self.actions = np.asarray([[0, 1], [1, 0], [0, -1], [-1, 0]], dtype=int_type)
        self.weight_actions = np.array([-1, 1], dtype=int_type)  # reduce, keep, increase
        self.action_names = ["right", "down", "left", "up", "weight_increase", "weight_decrease"]

        # Episode and reward settings
        self.episode_limit = getattr(args, "episode_limit", 100)
        self.time_reward = getattr(args, "reward_time", -0.1)
        self.send_to_base_reward = getattr(args, "reward_base", 10.0)

        # Initialize the internal state
        self.uavs = np.zeros((self.n_uavs, self.batch_size, 2), dtype=int_type)
        self.steps = 0
        self.sum_rewards = 0
        self.obs_size = int(self.state_size + self.grid_shape[0] * self.grid_shape[1])
        self.reset()


    def reset(self):
        self.int_queues.fill(0)
        self.float_queues.fill(0)
        self.weights.fill(1)
        self.steps = 0
        self.sum_rewards = 0
        self.uavs = np.zeros((self.n_uavs, self.batch_size, 2), dtype=int_type)
        self.grid.fill(0.0)
        self._place_actors(self.uavs, 1)
        rows, cols = self.grid_shape[0], self.grid_shape[1]
        ratio_of_ones = 0.7  # Example ratio
        self.environment_map = np.zeros((self.batch_size, self.x_max, self.y_max), dtype=float_type)
        for i in range(self.grid.shape[0]):
            self.environment_map[i,:,:] =  generate_binary_map_with_ratio(rows, cols, ratio_of_ones)
            self.grid[i, :, :, 0] = self.environment_map[i,:,:]

        return self.get_obs(), self.get_state()

    def step(self, actions):

        if not self.batch_mode:
            actions = np.expand_dims(np.asarray(actions, dtype=int_type), axis=1)


        reward = np.ones(self.batch_size, dtype=float_type) * self.time_reward
        terminated = [False for _ in range(self.batch_size)]


        # Move the UAVs and handle events
        for b in range(self.batch_size):
            for u in range(self.n_uavs):
                if actions[u, b] >= self.action_move:
                    new_weight = self.weights[u, b] + self.weight_actions[actions[u, b]-self.action_move]
                    self.weights[u, b] = np.clip(new_weight, 0, 5)
                else:
                    new_pos, collide = self._move_actor(self.uavs[u, b, :], actions[u, b], b, np.asarray([1], dtype=int_type))
                    if not collide:
                        self.grid[b, self.uavs[u, b, 0], self.uavs[u, b, 1], 1] = 0
                        self.grid[b, new_pos[0], new_pos[1], 1] = u+1
                        self.uavs[u, b, :] = new_pos
                        if self.grid[b, new_pos[0], new_pos[1], 0]:
                        	if self.float_queues[u, b] < self.queue_capacity:
                            		self.float_queues[u, b] += 1
                            		self.grid[b, new_pos[0], new_pos[1], 0] = 0  

        # Distribute bandwidth based on weights
        total_weight = np.sum(self.weights, axis=0)
        if total_weight == 0:
                self.weights.fill(1)
                total_weight = np.sum(self.weights, axis=0)
        bandwidths = (self.weights / total_weight) * self.max_bandwidth

        for b in range(self.batch_size):
            for u in range(self.n_uavs):
                # Calculate the UAV's distance to the base station
                distance_to_base_2D = np.linalg.norm(self.uavs[u, b, :] - self.base_station_pos[:2])
                distance_to_base_3D = np.sqrt(distance_to_base_2D**2 + HEIGHT**2)


                # Calculate data rate based on the distributed bandwidth and distance
                data_rate = self.calculate_data_rate(bandwidths[u, b], distance_to_base_3D)
                #print(total_weight,distance_to_base)
                # Simulate sending data to base station
                if self.float_queues[u, b] > 0:

                    # Store the previous integer size for comparison
                    previous_int_size = int(self.float_queues[u, b])

                    # Calculate how much data is transmitted in this step
                    transmitted_data = min(data_rate, self.float_queues[u, b])

                    # Decrease the queue by the transmitted data amount
                    self.float_queues[u, b] -= transmitted_data

                    # Check if the integer magnitude has changed
                    current_int_size = int(self.float_queues[u, b])
                    if current_int_size < previous_int_size:
                        completed_packets = previous_int_size - current_int_size
                        reward[b] += self.send_to_base_reward*completed_packets  # Reward for integer magnitude change

                    # If the queue is emptied, give the reward for completing the entire packet
                    if self.float_queues[u, b] <= 0:
                        self.float_queues[u, b] = 0  # Ensure the queue is fully cleared
                        reward[b] += self.send_to_base_reward  # Reward for completing the entire transmission
                    self.int_queues[u, b] = int(self.float_queues[u, b])


        self.sum_rewards += reward[0]
        self.steps += 1

        if self.steps >= self.episode_limit:
            terminated = [True for _ in range(self.batch_size)]

        if self.batch_mode:
            return reward, terminated, {}
        else:
            return reward[0].item(), int(terminated[0]), {}


    def get_obs_agent(self, agent_id, batch=0):

        flattened_grid = self.grid[batch,:,:,:].flatten()  
        flattened_queues = self.int_queues[:,batch].flatten()
        flattened_weights = self.weights[:,batch].flatten()
        position = np.zeros(self.grid_shape)

        position[self.uavs[agent_id,batch, 0], self.uavs[agent_id,batch, 1]] = 1.0
        concatenated_array = np.concatenate((flattened_grid, flattened_queues, flattened_weights,position.flatten()))
        return concatenated_array

    def calculate_data_rate(self, bandwidth, distance):
        if bandwidth == 0:
            return 0
        p_n = 10  # Transmit power (in watts)
        alpha_u = 1e-3  # Channel power gain at reference distance
        B = 0.01  # Total bandwidth (in Hz)
        N_0 = 1e-9  # Noise power spectral density (in watts/Hz)
	# Calculate the path loss (g_n_u_O)
        g_n_u_O = alpha_u / distance**2
    
	# Calculate the SNR (gamma_n_u_O)
        gamma_n_u_O = (p_n * g_n_u_O) / (B * bandwidth * N_0)
    
	# Calculate the channel capacity using the Shannon-Hartley theorem
        C_n_u_O = B * bandwidth * np.log2(1 + gamma_n_u_O)
        #print(C_n_u_O,gamma_n_u_O,bandwidth)
    
        return C_n_u_O




    def get_obs(self):
        agents_obs = [self.get_obs_agent(i) for i in range(self.n_uavs)]
        return agents_obs

    def get_obs_size(self):
        return self.obs_size

    def get_state_each_batch(self, batch = 0):
        flattened_grid = self.grid[batch,:,:,:].flatten()  
        flattened_queues = self.int_queues[:,batch].flatten()
        flattened_weights = self.weights[:,batch].flatten()

        concatenated_array = np.concatenate((flattened_grid, flattened_queues, flattened_weights))
        return concatenated_array

    def get_state(self):
        state = np.zeros((self.batch_size,self.state_size))
        for b in range(self.batch_size):
            state[b,:] = self.get_state_each_batch(b)
        return state

    def get_state_size(self):
        return self.state_size

    def get_avail_actions(self):
        avail_actions = []
        for uav_id in range(self.n_uavs):
            avail_uav = self.get_avail_agent_actions(uav_id)
            avail_actions.append(avail_uav)
        return avail_actions

    def get_avail_agent_actions(self, agent_id):
        """ Returns the available actions for agent_id """
        return np.ones(self.n_actions)

    def get_total_actions(self):
        """ Returns the total number of actions an agent could ever take """
        return self.n_actions

    def render(self):
        raise NotImplementedError

    def _place_actors(self, actors, type_id):
        for b in range(self.batch_size):
            for a in range(actors.shape[0]):
                is_free = False
                while not is_free:
                    actors[a, b, 0] = np.random.randint(self.env_max[0])
                    actors[a, b, 1] = np.random.randint(self.env_max[1])
                    is_free = self.grid[b, actors[a, b, 0], actors[a, b, 1], type_id] == 0
                self.grid[b, actors[a, b, 0], actors[a, b, 1], type_id] = a+1
                self.uavs[a, b, :] = actors[a, b,:]
                if self.grid[b, actors[a, b, 0], actors[a, b, 1], 0]:
                    if self.float_queues[a, b] < self.queue_capacity:
                        self.float_queues[a, b] += 1
                        self.int_queues = int(self.float_queues)
                        self.grid[b, actors[a, b, 0], actors[a, b, 1], 0] = 0  

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

    def seed(self):
        raise NotImplementedError

    def get_stats(self):
        pass

    def save_replay(self):
        raise NotImplementedError

    def get_env_info(self):
        env_info = {"state_shape": self.get_state_size(),
                    "obs_shape": self.get_obs_size(),
                    "n_actions": self.get_total_actions(),
                    "n_agents": self.n_uavs,
                    "episode_limit": self.episode_limit}
        return env_info
