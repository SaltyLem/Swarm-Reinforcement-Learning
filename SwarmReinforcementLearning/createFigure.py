from pickle import TRUE
import numpy as np
import matplotlib.pyplot as plt
import random as rand

class CreateFigure:
  def __init__(self, xMax, yMax) -> None:
    self.fig = plt.figure(figsize=(xMax, yMax))
    plt.subplots_adjust(left=0.05, right=0.95, bottom=0.05, top=0.95)

  def draw(self, maze: dict, drawWall: bool, plotCoordinate: bool) -> None:
    self.defaultDraw(maze, drawWall, plotCoordinate)

  def defaultDraw(self, maze: dict, drawWall: bool, plotCoordinate: bool = True) -> None:
    s = 0.2
    ax = plt.gca()
    lastPosition = next(reversed(maze), None)
    max = [0, 0]
    for m in maze:
      x,y = m
      x += 1
      y += 1
      if x > max[0]: max[0] = x
      if y > max[1]: max[1] = y

    for m in maze:
      x,y = m
      num = (x+1)+max[1]*y - 1

      # 十字架を作る部分
      plt.plot([x, x+s], [y, y], color='red', linewidth=2)
      plt.plot([x-s+1, x+1], [y, y], color='red', linewidth=2)
      plt.plot([x, x], [y-s+1, y+1], color='red', linewidth=2)
      plt.plot([x+1, x+1], [y-s+1, y+1], color='red', linewidth=2)
      plt.plot([x, x], [y+s, y], color='red', linewidth=2)
      plt.plot([x+1, x+1], [y+s, y], color='red', linewidth=2)

      # 状態の文字
      if num == 0:
        plt.text(x + 0.5, y + 0.4, 'START', ha='center')
      elif num == ((lastPosition[0] + 1) * (lastPosition[1] + 1) - 1):
        plt.text(x + 0.5, y + 0.4, 'GOAL', ha='center')
      elif plotCoordinate:
        plt.text(x + 0.5, y + 0.4, f'{m}', size=14, ha='center')

    if drawWall: self.drawWall(maze)

    # 描画範囲の設定と目盛りを消す設定
    ax.set_xlim(0, max[0])
    ax.set_ylim(0, max[1])
    plt.tick_params(labelbottom=False,
               labelleft=False,
               labelright=False,
               labeltop=False)

  def drawWall(self, maze: dict) -> None:
    for m in maze:
      x,y = m
      if maze[m][0] == 1:
        plt.plot([x, x+1], [y+1, y+1], color='red', linewidth=2)
      if maze[m][1] == 1:
        plt.plot([x+1, x+1], [y, y+1], color='red', linewidth=2)
      if maze[m][2] == 1:
        plt.plot([x, x+1], [y, y], color='red', linewidth=2)
      if maze[m][3] == 1:
        plt.plot([x, x], [y, y+1], color='red', linewidth=2)

  def result(self, episode) -> None:
    for (s, a) in episode:
      x, y = s
      if(s == (0, 0)):
        plt.plot([0.5], [0.5], marker="o", color='g', markersize=30)

      if a == (0, 1):
        plt.plot([x+0.5, x+0.5], [y+0.5, y+1.5], color='blue', linewidth=2)
      if a == (0, -1):
        plt.plot([x+0.5, x+0.5], [y+0.5, y-0.5], color='blue', linewidth=2)
      if a == (1, 0):
        plt.plot([x+0.5, x+1.5], [y+0.5, y+0.5], color='blue', linewidth=2)
      if a == (-1, 0):
        plt.plot([x+0.5, x-0.5], [y+0.5, y+0.5], color='blue', linewidth=2)
    self.fig.savefig("result_maze.png")

  def lineGraph(self, height, left, labels):
    colors = ['b', 'g', 'r', 'k']
    fig = plt.figure(figsize=(20, 20))
    for h in range(len(height)):
      label = labels[h] if isinstance(labels, list) else "f" + str(h)
      plt.plot(left[h], height[h], color=colors[h], label=label)
    plt.legend(loc="lower left", fontsize=18)
    fig.savefig("linegraph.png")

  def outputImage(self, name="img.png") -> None:
    self.fig.savefig(name)

  def showFig(self) -> None:
    self.fig.show()
