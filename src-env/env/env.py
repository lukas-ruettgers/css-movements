import functools
import random
from copy import copy

import numpy as np
from gymnasium.spaces import Discrete, MultiDiscrete

from pettingzoo import ParallelEnv

from .globals import SEED


class CSSMovements(ParallelEnv):
    metadata = {
        "name": "css_movements_v0",
    }

    def __init__(self, G, H, W, F_G, F_W, seed = SEED):
        self.H = H
        self.W = W 
        self.G = G
        self.F_G = F_G
        self.F_W = F_W
        self.time = None
        self.seed = seed 
        self.world_occupancy = None
        self.world_features  = None
        self.group_features  = None

    def reset(self, seed=SEED, options=None):
        self.time = 0
        self.world_occupancy = np.zeros((G, H, W), dtype=np.int32)
        self.world_features = np.zeros((F_W, H, W))
        self.group_features = np.zeros((G, F_G))

        """
        1. initialize the world_occupancy: for each group, randomly sample K=H*W/sparsity non-zero coordinates x,y and an occupancy value that is uniform from 0 to 100.  
        2. initialize the world_features: Just use some dummy normal initialization  
        3. initialize the group_features: Just use some dummy normal initialization  
        """

    def _get_group_strength(g,x,y):
        # Dummy value
        return self.world_occupancy[g][x][y]

    def _set_group_strength(g,x,y, val):
        self.world_occupancy[g][x][y] = val
    
    def _group_similarity(g1, g2):
        raise NotImplementedError()

    def _group_actions(g):
        # For every non-zero value self.world_occupancy[g][x][y], 
        # integral quantities 0 <= Q1,Q2,Q3,Q4 <= self.world_occupancy[g][x][y] can be moved up/right/down/left 
        pass

    def group_interaction(x,y):
        # Dummy interaction: Tournament mode
        # 1. Gather all non-zero group indices into an array a
        # 2. Randomly permute the array a
        # 3. Tournament mode: Contiguous groups fight each other, contiguous winners fight each other, etc.
        # - a[0] vs. a[1], a[2] vs. a[3]
        # 3a. For each fight, determine the group with the higher strength via _get_group_strength()
        #     The strengths of groups with strenghts g1,g2 are updated as max(0, g1 - g2) and max(0, g2 - g1)
        #     That is, the loser has strength 0 on this position afterwards.
        #     Use _set_group_strength for that. 
        # 3b. repeat this for the winners of each fight until there's only one left with non-zero strength
        pass

    def step(self, actions):
        """
        1. 
        For each group g, for each coord x,y,
        add integral quantities 0 <= Q1,Q2,Q3,Q4 <= self.world_occupancy[g][x][y]
        to their corresponding neighboring quantities.

        2. 
        For each coord x,y, check whether multiple groups occupy it.
        That is, self.world_occupancy[g][x][y] > 0 for multiple g
        In that case, invoke group_interaction(x,y) to resolve the collision.
        """
        self.time += 1

    def render(self):
        pass

    def observation_space(self, agent):
        # Return the entire occupancy grid, and world feature space
        return {
            "occupancy": self.world_occupancy,
            "features": self.world_features
        }

    def action_space(self, agent):
        return self._group_actions(agent)