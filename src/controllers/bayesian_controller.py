from modules.agents import REGISTRY as agent_REGISTRY
from components.action_selectors import REGISTRY as action_REGISTRY
from .basic_controller import BasicMAC
import torch as th
from utils.rl_utils import RunningMeanStd
import numpy as np
from estimator.estimate import estimate
from copy import deepcopy

# This multi-agent controller shares parameters between agents
class BayesianMAC:
    def __init__(self, scheme, groups, args):
        self.args = args
        self.n_agents = args.n_agents

        input_shape = self._get_input_shape(scheme)
        self._build_agents(input_shape)
        self.agent_output_type = args.agent_output_type

        self.action_selector = action_REGISTRY[args.action_selector](args)
        self.save_probs = getattr(self.args, 'save_probs', False)

        self.hidden_states = None
        self.env_args = getattr(self.args,'env_args',{})
        self.grid_shape = self.env_args.get('grid_shape',[20,30])
        self.map_width = self.grid_shape[0]
        self.map_height = self.grid_shape[1]
        self.estimation_threshold = getattr(self.args,'estimation_threshold',50)

    def select_actions(self, step_number,env,ep_batch, t_ep, t_env, bs=slice(None), test_mode=False):
        avail_actions = ep_batch["avail_actions"][:, t_ep]
        utility_function = getattr(self.args,'utility','expected')
        estimation = getattr(self.args,'estimation','False')
        if utility_function=='random':
            avail_actions_shape = avail_actions.shape
            chosen_actions_shape = (avail_actions_shape[0], avail_actions_shape[1])  # assuming given shape
            return th.randint(0, avail_actions_shape[2], chosen_actions_shape).cpu()
        if estimation:
            chosen_actions =  self.select_action_with_est(step_number,env,ep_batch, t_ep, t_env, bs, test_mode).cpu()
            return chosen_actions
        qvals = self.forward(ep_batch, t_ep, test_mode=test_mode)
        chosen_actions = self.action_selector.select_action(qvals[bs], avail_actions[bs], t_env, test_mode=test_mode)
        return chosen_actions.cpu()

    def forward(self, ep_batch, t, test_mode=False):
        if test_mode:
            self.agent.eval()
            
        agent_inputs = self._build_inputs(ep_batch, t)
        avail_actions = ep_batch["avail_actions"][:, t]
        agent_outs, self.hidden_states = self.agent(agent_inputs, self.hidden_states)

        return agent_outs


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



    def select_action_with_est(self, step_number,env, ep_batch, t_ep, t_env, bs, test_mode):
        utility_function = getattr(self.args,'utility','expected')
        step_number_period = getattr(self.args,'estimation_step_size',90)
        avail_actions = ep_batch["avail_actions"][:, t_ep]
        x_loc_event, y_loc_event, x_loc_uav, y_loc_uav= self.get_event_and_uav_locations(ep_batch["obs"][:, t_ep],env.get_knowledge_map())

        if step_number < 1:
            self.estimated_intensity = None
        if x_loc_event == None:
            #print('aaa: ',(env.get_knowledge_map() == 1).sum())
            #print('.......Exploring.......')
            agent_outputs = self.forward(ep_batch, t_ep, test_mode=test_mode)
            chosen_actions = self.action_selector.select_action(agent_outputs[bs], avail_actions[bs], t_env, test_mode=test_mode)
            return chosen_actions
        if x_loc_event != None: 
            if  (step_number%step_number_period==0 or self.estimated_intensity is None):
                full_estimation_intensity, intensity_max, cov, std = estimate(5,x_loc_event, y_loc_event,self.args)
                #current_avg = full_estimation_intensity.mean()
                max_p = np.max(full_estimation_intensity)
                min_p = np.min(full_estimation_intensity)
                #target_avg = 0.5
                #scaling_factor = target_avg / current_avg
                self.estimated_intensity = full_estimation_intensity/max_p#*scaling_factor#/intensity_max
                #self.estimated_intensity = (full_estimation_intensity - min_p)/(max_p-min_p)#*scaling_factor#/intensity_max
                self.estimated_intensity,scaling_factor, iteration = self.bisection_scale_full_map(self.estimated_intensity)
                # print(((full_estimation_intensity - min_p)/(max_p-min_p)).mean(),self.estimated_intensity.mean(),scaling_factor, iteration)
                import sys
                #np.set_printoptions(threshold=sys.maxsize)
                #np.set_printoptions(formatter={'float': lambda x: f"{x:.2g}"}) 
                #print('Env Map full: ', env.environment_map)
                #print('Knowledge map: ',env.get_knowledge_map(), len(x_loc_event))
                # print('Estimation Map: ', full_estimation_intensity[0:10,0:10])
                # print('Estimation Map Scaled: ', self.estimated_intensity[0:10,0:10])
                #print('Estimation Map Full: ', self.estimated_intensity)

                #print('******************aaaa****************')
                print(step_number)
            # if utility_function == 'one':
            #     knowledge_map_shape = self.grid_shape
            #     knowledge_map_size = np.prod(knowledge_map_shape)
            #     binary_array_events = (self.estimated_intensity > 0.5).astype(int)
            #     observedState = deepcopy(env.observed_state[0,:,:])
            #     observedState[observedState == 2] = binary_array_events[observedState == 2]
            #     model_ep = deepcopy(ep_batch)
            #     binary_array_events_tensor = th.tensor(observedState.flatten(), dtype=th.float32).to(model_ep["obs"].device)
            #     for x in range(ep_batch["obs"][:, t_ep].shape[1]):
            #         model_ep["obs"][0, t_ep][x][:knowledge_map_size] = binary_array_events_tensor
            #     qvals = self.forward(model_ep, t_ep, test_mode=test_mode)
            #     chosen_actions = self.action_selector.select_action(qvals[bs], avail_actions[bs], t_env, test_mode=test_mode)
            #     return chosen_actions.cpu()
            sampled_maps = self.generate_binary_maps_with_probabilities(self.estimated_intensity,1)
            qvalsList = []
            knowledge_map_shape = self.grid_shape
            knowledge_map_size = np.prod(knowledge_map_shape)
            import sys
            #np.set_printoptions(threshold=sys.maxsize)
            np.set_printoptions(formatter={'float': lambda x: f"{x:.5g}"}) 
            for sampled_map,probability in sampled_maps:
                sampled_map = (self.estimated_intensity > 0.5).astype(int)
                probability = 1
                model_ep = deepcopy(ep_batch)
                observedState = deepcopy(env.observed_state[0,:,:])
                observedState[observedState == 2] = sampled_map[observedState == 2]
                #print(np.sum(np.logical_xor(observedState,env.observed_state[0,:,:])),'*',np.count_nonzero(env.observed_state[0,:,:] != 2),'*', np.sum(np.logical_xor(observedState,env.environment_map[0,:,:])))
                import sys
                np.set_printoptions(threshold=sys.maxsize)
                np.set_printoptions(formatter={'float': lambda x: f"{x:.2g}"}) 
                #np.set_printoptions(precision=2, suppress=True)
                #print('Env Map full: ', env.environment_map)
                #print('Estimation Map Full: ', self.estimated_intensity)                
                #print(observedState)
                #print('ccc: ',(sampled_map== 1).sum(),(is_sent_map== 0).sum(),(both_ones== 1).sum())
                binary_array_events_tensor = th.tensor(observedState.flatten(), dtype=th.float32).to(model_ep["obs"].device)
                for x in range(ep_batch["obs"][:, t_ep].shape[1]):
                    #print(np.count_nonzero(knowledge_map_size,model_ep["obs"][0, t_ep][x] == 2))
                    model_ep["obs"][0, t_ep][x][:knowledge_map_size] = binary_array_events_tensor
                    #print(np.count_nonzero(knowledge_map_size,model_ep["obs"][0, t_ep][x] == 2),'***')
                qvals = self.forward(model_ep, t_ep, test_mode=test_mode)
                qvalsList.append((qvals, probability))  
            qvals_tensor = th.stack([q[0] for q in qvalsList])  # Shape: (num_samples, 1, n_agents, n_actions)
            probabilities = th.tensor([q[1] for q in qvalsList], dtype=th.float32).to(qvals_tensor.device)
            expected_qvals = (qvals_tensor.squeeze(1) * probabilities.view(-1, 1, 1)).sum(dim=0)
            max_expected_actions = th.argmax(expected_qvals, dim=-1).view(qvals.shape[0], -1)  # Shape: (n_agents,)
            if utility_function=='expected':
                return max_expected_actions.cpu()
            adjusted_95th_percentile_qvals = self.probability_adjusted_percentile(qvals_tensor.squeeze(1), probabilities)
            max_percentile_actions = th.argmax(adjusted_95th_percentile_qvals, dim=-1).view(qvals.shape[0], -1)  # Shape: (n_agents,)
            return max_percentile_actions.cpu()


    def bisection_scale_full_map(self,intensity_map, target_mean=0.5, tolerance=1e-3, max_iterations=500):
        if target_mean/intensity_map.mean() > 1:
            low, high = 1, target_mean/(np.min(intensity_map)+1e-6)  # Reasonable bounds for scaling
        else:
            return intensity_map*target_mean/intensity_map.mean(), target_mean/intensity_map.mean(), 1
            
        iteration = 0
        scaled_map = intensity_map.copy()
        
        while iteration < max_iterations:
            # Midpoint of the current bounds
            scaling_factor = (low + high) / 2
            
            # Scale the map and clip to [0, 1]
            scaled_map = np.clip(intensity_map * scaling_factor, 0, 1)
            
            # Compute the mean of the scaled map
            current_mean = scaled_map.mean()
            
            # Check for convergence
            if abs(current_mean - target_mean) < tolerance:
                break
            
            # Update bounds
            if current_mean < target_mean:
                low = scaling_factor
            else:
                high = scaling_factor
            
            iteration += 1

        return scaled_map, scaling_factor, iteration

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


    def probability_adjusted_percentile(self,qvals_tensor, probabilities, percentile=0.95):
        """
        Calculate the probability-adjusted percentile for each agent separately.
        
        Args:
        - qvals_tensor (torch.Tensor): Shape (num_samples, n_agents, n_actions), Q-values.
        - probabilities (torch.Tensor): Shape (num_samples,), associated probabilities.
        - percentile (float): Percentile to calculate (default 95th percentile).
        
        Returns:
        - adjusted_percentile_qvals (torch.Tensor): Shape (n_agents, n_actions), adjusted percentile per agent.
        """
        num_samples, n_agents, n_actions = qvals_tensor.shape
        adjusted_percentile_qvals = th.zeros((n_agents, n_actions), device=qvals_tensor.device)
        
        for agent in range(n_agents):
            # Extract Q-values for the current agent across all samples
            agent_qvals = qvals_tensor[:, agent, :]  # Shape: (num_samples, n_actions)
            
            for action in range(n_actions):
                # Extract Q-values for this action across samples
                action_qvals = agent_qvals[:, action]  # Shape: (num_samples,)
                
                # Sort Q-values and associated probabilities
                sorted_indices = th.argsort(action_qvals)
                sorted_qvals = action_qvals[sorted_indices]
                sorted_probabilities = probabilities[sorted_indices]
                
                # Compute cumulative probabilities
                cumulative_probs = th.cumsum(sorted_probabilities, dim=0)
                
                # Find the index where the cumulative probability exceeds the target percentile
                target_cumulative_prob = percentile
                percentile_index = th.searchsorted(cumulative_probs, target_cumulative_prob)
                
                # Retrieve the Q-value corresponding to this index
                adjusted_percentile_qvals[agent, action] = sorted_qvals[min(percentile_index, len(sorted_qvals) - 1)]
        
        return adjusted_percentile_qvals





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
'''
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
        #print('aaa',.shape)
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
'''