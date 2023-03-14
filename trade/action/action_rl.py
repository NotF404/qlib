import numpy as np
from gym.spaces import Discrete, Box, Tuple, MultiDiscrete

from .base import Base_Action


class Static_Action(Base_Action):
    """ """

    def __init__(self, config):
        self.action_num = config["action_num"]
        self.action_map = config["action_map"]
        self.is_buy = config["is_buy"]

    def get_space(self):
        """ """
        return Discrete(self.action_num)

    def get_action(self, action, position, **kargs):
        """

        :param action:
        :param position:
        :param target:
        :param **kargs:

        """
        if self.is_buy:
            return min(self.action_map[action], 1. - position)
        else:
            return min(self.action_map[action], position)
