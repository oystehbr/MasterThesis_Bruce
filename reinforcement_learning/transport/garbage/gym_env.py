import gym
from gym import spaces
import pygame
import numpy as np
from Route_opt import RouteOptimization
import maps


class RouteEnv(gym.Env):

    def __init__(self, nodes, edges, dumpers):
        self.nodes = nodes
        self.edges = edges
        self.dumpers = dumpers
        self.world = RouteOptimization(maps.map3)

        # self.action_space = spaces.Discrete(self.nodes)  # Number of actions

    def step(self, action):

        # do your turn

        # run every other dumper, random at first - but use of model sooner

        # calculate reward and so one.
        pass

    def reset(self):
        pass
