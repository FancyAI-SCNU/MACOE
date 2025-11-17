import numpy as np
from gym.spaces import Discrete, Box, Tuple, MultiDiscrete

np.random.seed(11)

class Base_Action(object):
    """ """

    def __init__(self, config):
        return

    def __call__(self, *args, **kargs):
        return self.get_action(*args, **kargs)

    def get_action(self, action):
        """

        :param action:

        """
        return action



class Static_Action(Base_Action):
    """ """

    def __init__(self, config):
        self.action_num = config["action_num"]
        self.action_map = config["action_map"]

    def get_space(self):
        """ """
        return Discrete(self.action_num)

    def get_action(self, action, target, position, **kargs):
        """

        :param action:
        :param position:
        :param target:
        :param **kargs:

        """
        
        return min(target * self.action_map[action], position)
    
class ac_Static_Action(Base_Action):
    """ """

    def __init__(self, config):
        self.action_num = config["action_num"]
        self.action_map = config["action_map"]

    def get_space(self):
        """ """
        return Discrete(self.action_num)

    def get_action(self, action, target, position, **kargs):
        """

        :param action:
        :param position:
        :param target:
        :param **kargs:

        """
       
        # return min(target * self.action_map[action], position)
        return min(target * action, position)