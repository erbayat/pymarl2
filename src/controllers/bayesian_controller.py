from modules.agents import REGISTRY as agent_REGISTRY
from components.action_selectors import REGISTRY as action_REGISTRY
import torch as th
import numpy as np
from estimator.estimate import estimate
from copy import deepcopy
# This multi-agent controller shares parameters between agents
class BayesianMAC:
    def __init__(self, scheme, groups, args):
        self.n_agents = args.n_agents
        self.args = args
        input_shape = self._get_input_shape(scheme)
        self._build_agents(input_shape)
        self.agent_output_type = args.agent_output_type
        self.action_selector = action_REGISTRY[args.action_selector](args)
        self.save_probs = getattr(self.args, 'save_probs', False)
        self.grid_shape = getattr(self.args,'grid_shape',[20,30])
        self.map_width = self.grid_shape[0]
        self.map_height = self.grid_shape[1]
        self.hidden_states = None
        self.reward_event = th.tensor(10.0)       
        self.reward_no_event = th.tensor(0.0) 
        self.estimation_threshold = getattr(self.args,'estimation_threshold',20)

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
    
    def select_actions(self, step_number, env, ep_batch, t_ep, t_env, bs=slice(None), test_mode=True):
        model_ep = deepcopy(ep_batch)
        step_number = getattr(self.args,'estimation_step_size',5)
        utility_function = getattr(self.args,'utility','expected')
        avail_actions = ep_batch["avail_actions"][:, t_ep]
        if utility_function=='random':
            avail_actions_shape = avail_actions.shape
            chosen_actions_shape = (avail_actions_shape[0], avail_actions_shape[1])  # assuming given shape
            return th.randint(0, avail_actions_shape[2], chosen_actions_shape)
        # Only select actions for the selected batch elements in bs
        
        # if full_env_obs != None:
        #     agent_outputs = self.forward(ep_batch, t_ep, test_mode=test_mode)
        #     chosen_actions = self.action_selector.select_action(agent_outputs[bs], avail_actions[bs], t_env, test_mode=test_mode)
        #     return chosen_actions
        x_loc_event = None
        full_estimation_intensity = None
        do_estimation = getattr(self.args,'estimation',False)
        if do_estimation:
            x_loc_event, y_loc_event, x_loc_uav, y_loc_uav= self.get_event_and_uav_locations(ep_batch["obs"][:, t_ep],env.get_knowledge_map())
        if x_loc_event != None and step_number%5==0:
            full_estimation_intensity, intensity_max, cov, std = estimate(10,x_loc_event, y_loc_event)
            binary_array_events = (full_estimation_intensity > 0.5).astype(int)
            knowledge_map_shape = self.grid_shape
            knowledge_map_size = np.prod(knowledge_map_shape)
            binary_array_events_tensor = th.tensor(binary_array_events.flatten(), dtype=th.float32).to(model_ep["obs"].device)
            # Loop through each x and assign the same value
            for x in range(ep_batch["obs"][:, t_ep].shape[1]):
                model_ep["obs"][0, t_ep][x][:knowledge_map_size] = binary_array_events_tensor


        agent_outputs = self.forward(model_ep, t_ep, test_mode=test_mode)
        #print('aaa',agent_outputs.shape)
        #current_locations = np.column_stack((x_loc_uav, y_loc_uav))
        #agent_outputs = self.adjust_utilities(agent_outputs, current_locations, full_estimation_intensity, utility_function)
        chosen_actions = self.action_selector.select_action(agent_outputs[bs], avail_actions[bs], t_env, test_mode=test_mode)
        return chosen_actions
    


    def forward(self, ep_batch, t, test_mode=False):
        agent_inputs = self._build_inputs(ep_batch, t)
        avail_actions = ep_batch["avail_actions"][:, t]
        if test_mode:
            self.agent.eval()
        agent_outs, self.hidden_states = self.agent(agent_inputs, self.hidden_states)

        # Softmax the agent outputs if they're policy logits
        if self.agent_output_type == "pi_logits":

            if getattr(self.args, "mask_before_softmax", True):
                # Make the logits for unavailable actions very negative to minimise their affect on the softmax
                agent_outs = agent_outs.reshape(ep_batch.batch_size * self.n_agents, -1)
                reshaped_avail_actions = avail_actions.reshape(ep_batch.batch_size * self.n_agents, -1)
                agent_outs[reshaped_avail_actions == 0] = -1e5

            agent_outs = th.nn.functional.softmax(agent_outs, dim=-1)
            
        return agent_outs.view(ep_batch.batch_size, self.n_agents, -1)

    def init_hidden(self, batch_size):
        self.hidden_states = self.agent.init_hidden()
        if self.hidden_states is not None:
            self.hidden_states = self.hidden_states.unsqueeze(0).expand(batch_size, self.n_agents, -1)  # bav

    def parameters(self):
        return self.agent.parameters()

    def load_state(self, other_mac):
        self.agent.load_state_dict(other_mac.agent.state_dict())

    def cuda(self):
        self.agent.cuda()

    def save_models(self, path):
        th.save(self.agent.state_dict(), "{}/agent.th".format(path))

    def load_models(self, path):
        self.agent.load_state_dict(th.load("{}/agent.th".format(path), map_location=lambda storage, loc: storage))

    def _build_agents(self, input_shape):
        self.agent = agent_REGISTRY[self.args.agent](input_shape, self.args)

    def _build_inputs(self, batch, t):
        # Assumes homogenous agents with flat observations.
        # Other MACs might want to e.g. delegate building inputs to each agent
        bs = batch.batch_size
        inputs = []
        inputs.append(batch["obs"][:, t])  # b1av
        if self.args.obs_last_action:
            if t == 0:
                inputs.append(th.zeros_like(batch["actions_onehot"][:, t]))
            else:
                inputs.append(batch["actions_onehot"][:, t-1])
        if self.args.obs_agent_id:
            inputs.append(th.eye(self.n_agents, device=batch.device).unsqueeze(0).expand(bs, -1, -1))

        inputs = th.cat([x.reshape(bs, self.n_agents, -1) for x in inputs], dim=-1)
        return inputs

    def _get_input_shape(self, scheme):
        input_shape = scheme["obs"]["vshape"]
        if self.args.obs_last_action:
            input_shape += scheme["actions_onehot"]["vshape"][0]
        if self.args.obs_agent_id:
            input_shape += self.n_agents

        return input_shape
