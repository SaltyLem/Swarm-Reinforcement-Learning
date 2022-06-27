import numpy as np
import matplotlib.pyplot as plt
import random as rand

class Agent:
  def __init__(self, maze) -> None:
    self.maze = maze
    self.actions = [(0, 1), (1, 0), (0, -1), (-1, 0)]
    self.lastPosition = next(reversed(self.maze), None)
    self.lastState = (self.lastPosition[0] + 1) * (self.lastPosition[1] + 1) - 1
    self.policy = {}
    self.state = (0, 0)
    self.previousState = (0, 0)
    for m in maze:
      self.policy[m] = self.actions[np.random.randint(len(self.actions))]

    self.q = {}
    self.v = {} # 変化量. PSOで使用する
    for m in maze:
      for a in self.actions:
        self.q[(m, a)] = 0
        self.v[(m, a)] = 0

  def move(self, a):
    x, y = self.state
    dx, dy = a
    self.previousState = self.state
    if self.maze[self.state][self.actions.index((dx, dy))] == 1:
      return 0, self.previousState
    elif (x+dx, y+dy) == self.lastPosition:
      self.state = (x+dx, y+dy)
      return 100, self.previousState
    else:
      self.state = (x+dx, y+dy)
      return 0, self.previousState

  def policy_update(self, s):
    q_max = -10**10
    a_best = None
    for a in self.actions:
      if self.q[(s, a)] > q_max:
        q_max = self.q[(s, a)]
        a_best = a

    self.policy[s] = a_best

  def chooseAction(self, epsilon):
    if np.random.random() < epsilon:
      a = self.actions[np.random.randint(len(self.actions))]
    else:
      a = self.policy[self.state]

    return a

  def train(self, action, reward):
    self.q[(self.previousState, action)] += 0.2 * (reward + self.q[(self.state, self.policy[self.state])] - self.q[(self.previousState, action)])
    self.policy_update(self.previousState)
    return self.q[(self.previousState, action)]

  def moveStart(self):
    self.state = (0, 0)
    self.previousState = (0, 0)

  def setQ(self, q):
    for i in q:
      self.q[i] = q[i]

  def chengeGoal(self, xrange, yrange):
    mx, my = next(reversed(self.maze), None)
    x = mx - rand.randint(0,xrange)
    y = my - rand.randint(0,yrange)
    self.lastPosition = (x, y)

