

import numpy as np


class Abs_Reward(object):
   
    def __init__(self, config):
        return

    def get_reward(self):
        """:return: reward"""
        reward = 0
        return reward

    def __call__(self, *args, **kargs):
        return self.get_reward(*args, **kargs)

    def isinstant(self):
        
        raise NotImplementedError


class Instant_Reward(Abs_Reward):
    def __init__(self, config):
        self.ffr_ratio = config["ffr_ratio"]
        self.vvr_ratio = config["vvr_ratio"]

    def isinstant(self):
        return True


class EndEpisode_Reward(Abs_Reward):
    def __init__(self, config):
        self.ffr_ratio = config["ffr_ratio"]
        self.vvr_ratio = config["vvr_ratio"]

    def isinstant(self):
        return False




class VP_Penalty_small(Instant_Reward):
    
    def __init__(self, config):
        self.penalty = config["penalty"]

    def get_reward(self, performance_raise, v_t, target, *args):
        
        assert target > 0
        reward = performance_raise * v_t / target
        reward -= self.penalty * (v_t / target) ** 2
        assert not (np.isnan(reward) or np.isinf(reward)), f"{performance_raise}, {v_t}, {target}"
        return reward / 100


class VP_Penalty_small_vec(VP_Penalty_small):
    def get_reward(self, performance_raise, v_t, target, *args):
        """

        :param performance_raise: Abs(vv_ratio_t - 1) * 10000.
        :param target: Target volume
        :param v_t: The traded volume
        """
        
        assert target > 0
        
        reward = performance_raise * v_t.sum() / target
        
        # print(reward)
        reward -= self.penalty * ((v_t / target) ** 2).sum()
        # print( self.penalty * ((v_t / target) ** 2).sum())
        
        assert not (np.isnan(reward) or np.isinf(reward)), f"{performance_raise}, {v_t}, {target}"
        return reward / 100
    
class VP_Penalty_small_new(VP_Penalty_small):
    def get_reward(self, performance_raise, v_t, target,this_cash_before,this_cash_after, *args):
        """

        :param performance_raise: Abs(vv_ratio_t - 1) * 10000.
        :param target: Target volume
        :param v_t: The traded volume
        """
        assert target > 0
        reward = performance_raise * v_t.sum() / target 
        reward -= self.penalty * ((v_t / target) ** 2).sum()
        if this_cash_before>0 and this_cash_after==0:
            temp = -1
        else:
            temp=0
        reward+=temp
        assert not (np.isnan(reward) or np.isinf(reward)), f"{performance_raise}, {v_t}, {target}"
        return reward / 100
