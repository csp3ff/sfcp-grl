import copy
import numpy as np
import torch
from model.GCN_PPO import GCN_PPO
from controller.BaseController import BaseController

class GCN_PPO_Controller(BaseController):
    def __init__(self, config):
        super().__init__(config)
        self.agent = GCN_PPO(self.env.network.feature_dim, len(self.env.network.nodes), config, self.env.network.get_adjacency_matrix)

    def reset(self):
        self.env.reset()
        self.agent.reset()
        assert len(self.env.network.embedding_sfcr) == 0
        assert all(len(sn.hosting)==0 for sn in self.env.network.nodes)
        assert all(len(sl.hosting)==0 for sl in self.env.network.links)
    
    def solve(self, sfcr, timeslot):
        """
        Process the SFC request and return the result.
        """
        candidates_sn = dict()
        # First pass - decision phase
        for vn in sfcr.vnet.nodes:
            # Using modified choose_node which now returns action_prob and value
            mask = self.gen_mask(candidates_sn, vn)
            if torch.sum(mask) == 0:
                print(f"Node {vn.id} has no valid candidates, skipping SFC request.")
                return {'status': False, 'delay': 0, 'revenue': [], 'cost': [], 'revenue_cost_ratio': [], 'rewards': []}
            action, state, action_log_prob = self.agent.select_action(sfcr, candidates_sn, vn, self.env.network, mask)
            # Store all information for trajectory
            candidates_sn[vn.id] = action
            
        # Link selection
        status, candidates_sl = self.choose_link(sfcr, candidates_sn)
        result = self.env.step(status, sfcr, candidates_sn, candidates_sl, timeslot)
        return result
    
    def learn(self, sfcr, timeslot):
        """
        Process the SFC request and learn from the experience.
        """
        candidates_sn = dict()
        states = []
        actions = []
        action_probs = []
        masks = []
        
        for vn in sfcr.vnet.nodes:
            # Using modified choose_node which now returns action_prob and value
            mask = self.gen_mask(candidates_sn, vn)

            if torch.sum(mask) == 0:
                print(f"Node {vn.id} has no valid candidates, skipping SFC request.")
                return {'status': False, 'delay': 0, 'revenue': [], 'cost': [], 'revenue_cost_ratio': [], 'rewards': []}

            action, state, action_log_prob = self.agent.select_action(sfcr, candidates_sn, vn, self.env.network, mask)
            
            # Store all information for trajectory
            states.append(state)
            candidates_sn[vn.id] = action
            actions.append(action)
            action_probs.append(action_log_prob)
            masks.append(mask)

            sn = self.env.network.nodes[action]
            sn_remain_cpu, sn_remain_mem = sn.remain_sn()
            assert sn_remain_cpu >= vn.cpu and sn_remain_mem >= vn.mem, \
                f"Insufficient resources on SN {sn.id} for VN {vn.id}. Remaining CPU: {sn_remain_cpu}, Memory: {sn_remain_mem}"

        # Link selection
        status, candidates_sl = self.choose_link(sfcr, candidates_sn)
        # print(f"Link selection {'succeeded' if status else 'failed'} for SFCR {sfcr.id}.")

        result = self.env.step(status, sfcr, candidates_sn, candidates_sl, timeslot)
        rewards = result['rewards']
        # Store experiences in trajectory
        for i in range(len(states)):
            self.agent.remember(
                state=states[i],
                mask=masks[i],
                action=actions[i],
                reward=rewards[i],
                next_state=states[i+1] if i < len(states) - 1 else states[i],
                done=0 if i < len(states) - 1 else 1,
                action_prob=action_probs[i],
            )
        return result
