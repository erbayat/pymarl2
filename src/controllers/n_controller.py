from modules.agents import REGISTRY as agent_REGISTRY
from components.action_selectors import REGISTRY as action_REGISTRY
from .basic_controller import BasicMAC
import torch as th
from utils.rl_utils import RunningMeanStd
import numpy as np
from estimator.estimate import estimate
from copy import deepcopy

# This multi-agent controller shares parameters between agents
class NMAC(BasicMAC):
    def __init__(self, scheme, groups, args):
        super(NMAC, self).__init__(scheme, groups, args)
        self.env_args = getattr(self.args,'env_args',{})
        self.grid_shape = self.env_args.get('grid_shape',[20,30])
        self.map_width = self.grid_shape[0]
        self.map_height = self.grid_shape[1]
        self.estimation_threshold = getattr(self.args,'estimation_threshold',40)

    def select_actions(self, step_number,env,ep_batch, t_ep, t_env, bs=slice(None), test_mode=False):
        # Only select actions for the selected batch elements in bs
        avail_actions = ep_batch["avail_actions"][:, t_ep]
        utility_function = getattr(self.args,'utility','expected')
        estimation = getattr(self.args,'estimation','False')
        if utility_function=='random':
            avail_actions_shape = avail_actions.shape
            chosen_actions_shape = (avail_actions_shape[0], avail_actions_shape[1])  # assuming given shape
            return th.randint(0, avail_actions_shape[2], chosen_actions_shape)
        if estimation:
            return self.select_action_with_est(step_number,env,ep_batch, t_ep, t_env, bs, test_mode)
        qvals = self.forward(ep_batch, t_ep, test_mode=test_mode)
        chosen_actions = self.action_selector.select_action(qvals[bs], avail_actions[bs], t_env, test_mode=test_mode)
        return chosen_actions

    def forward(self, ep_batch, t, test_mode=False):
        if test_mode:
            self.agent.eval()
            
        agent_inputs = self._build_inputs(ep_batch, t)
        avail_actions = ep_batch["avail_actions"][:, t]
        agent_outs, self.hidden_states = self.agent(agent_inputs, self.hidden_states)

        return agent_outs

    def get_best_action_expected(self, step_number,env, ep_batch, t_ep, t_env, bs, test_mode):
        model_ep = deepcopy(ep_batch)
        step_number = getattr(self.args,'estimation_step_size',5)
        avail_actions = ep_batch["avail_actions"][:, t_ep]
        full_estimation_intensity = None
        x_loc_event, y_loc_event, x_loc_uav, y_loc_uav= self.get_event_and_uav_locations(ep_batch["obs"][:, t_ep],env.get_knowledge_map())
        if x_loc_event != None and step_number%5==0:
            full_estimation_intensity, intensity_max, cov, std = estimate(10,x_loc_event, y_loc_event,self.args)
            binary_array_events = (full_estimation_intensity > 0.5).astype(int)
            knowledge_map_shape = self.grid_shape
            knowledge_map_size = np.prod(knowledge_map_shape)
            binary_array_events_tensor = th.tensor(binary_array_events.flatten(), dtype=th.float32).to(model_ep["obs"].device)
            print(ep_batch)
            # Loop through each x and assign the same value
            for x in range(ep_batch["obs"][:, t_ep].shape[1]):
                model_ep["obs"][0, t_ep][x][:knowledge_map_size] = binary_array_events_tensor
        elif x_loc_event != None:
            model_ep["obs"][0, t_ep][x][:knowledge_map_size] = model_ep["obs"][0, t_ep-1][x][:knowledge_map_size]    
        agent_outputs = self.forward(model_ep, t_ep, test_mode=test_mode)
        chosen_actions = self.action_selector.select_action(agent_outputs[bs], avail_actions[bs], t_env, test_mode=test_mode)
        print('Action:', chosen_actions,np.count_nonzero(env.get_knowledge_map()))
        return chosen_actions

    def select_action_with_est(self, step_number,env, ep_batch, t_ep, t_env, bs, test_mode):
        utility_function = getattr(self.args,'utility','expected')
        if utility_function=='expected':
            return self.get_best_action_expected(step_number, env, ep_batch, t_ep, t_env, bs, test_mode) 

    def generate_binary_maps_with_probabilities(self,intensity_map, K):
        """
        Generates K samples of NxM binary maps based on a given probability intensity map
        and calculates the probability of observing each sampled map.
        
        Parameters:
        - intensity_map: numpy array of shape (N, M) containing probabilities for each pixel.
        - K: int, the number of samples to generate.
        
        Returns:
        - samples_with_probabilities: list of tuples, where each tuple contains:
            - binary map (numpy array of shape (N, M))
            - probability of observing that map (float)
        """
        N, M = intensity_map.shape
        samples_with_probabilities = []
        
        for _ in range(K):
            # Generate a binary map by sampling each pixel with its corresponding probability
            sample = (np.random.rand(N, M) < intensity_map).astype(int)
            
            # Calculate the probability of this specific binary map
            prob_map = np.where(sample == 1, intensity_map, 1 - intensity_map)
            map_probability = np.prod(prob_map)  # Product of probabilities for this map
            
            samples_with_probabilities.append((sample, map_probability))
        
        return samples_with_probabilities

    def adjust_utilities(self, agent_outputs, current_locations, full_estimation_intensity, utility_function):
        num_actions = agent_outputs.shape[-1]
        num_uavs = agent_outputs.shape[-1]

        adjusted_agent_outputs = agent_outputs.clone()
        for uav_index in range(num_uavs):
            for action in range(num_actions):
                resulting_location = self.get_resulting_location(current_locations[uav_index], action)

                x_res, y_res = resulting_location
                p_event = full_estimation_intensity[y_res, x_res]  # Note: Arrays are indexed as [row, column]

                if utility_function == 'expected':
                    utility = self.calculate_expected_utility(p_event)
                elif utility_function == 'risk_averse':
                    utility = self.calculate_risk_averse_utility(p_event)
            # elif utility_function == 'maximin':
                #    utility = self.calculate_maximin_utility(p_event)
                else:
                    raise ValueError("Invalid utility function specified.")

                adjusted_agent_outputs[0, uav_index, action] = utility

        return adjusted_agent_outputs

    def get_resulting_location(self, current_location, action):
        x, y = int(current_location[0]),int(current_location[1])

        if action == 0:
            x += 1    
        elif action == 1:    
            y += 1
        elif action == 2:    
            x -= 1
        elif action == 3:    
            y -= 1            
        x = np.clip(x, 0, self.map_width - 1)
        y = np.clip(y, 0, self.map_height - 1)
        return x, y

    def get_event_and_uav_locations(self,observations,knowledge_map, queue_shape=(15), weights_shape=(15)):
        # Extract the length of each component from the concatenated array
        knowledge_map_shape = self.grid_shape
        knowledge_map_size = np.prod(knowledge_map_shape)

        uav_positions_size = np.prod(self.grid_shape)
        queue_size = np.prod(queue_shape)
        weights_size = np.prod(weights_shape)

        observations = observations.cpu()

        # Split the concatenated array into its original components
        uav_positions_flat = observations[0][0][knowledge_map_size:knowledge_map_size + uav_positions_size]
        flattened_queues = observations[0][0][knowledge_map_size + uav_positions_size:knowledge_map_size + uav_positions_size + queue_size]
        flattened_weights = observations[0][0][knowledge_map_size + uav_positions_size + queue_size:knowledge_map_size*2+queue_size*2]
        
        # Reshape each component back to its original shape

        uav_positions = uav_positions_flat.reshape(self.grid_shape)
        queues = flattened_queues.reshape(queue_shape)
        weights = flattened_weights.reshape(weights_shape)
        event_indices = np.argwhere(knowledge_map == 1)
        uav_indices = np.argwhere(uav_positions > 0)
        uav_x_coords = uav_indices[0].tolist()
        uav_y_coords = uav_indices[1].tolist()
        if event_indices.shape[0] < self.estimation_threshold:
            return None,None,uav_x_coords,uav_y_coords

        # Split into separate x and y coordinate lists
        event_x_coords = event_indices[:,0].tolist()
        event_y_coords = event_indices[:,1].tolist()
        return event_x_coords, event_y_coords,uav_x_coords, uav_y_coords