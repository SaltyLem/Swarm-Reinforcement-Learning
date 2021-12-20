import numpy as np
import matplotlib.pyplot as plt
import random as rand

class createFigure:
  def __init__(self) -> None:
    self.fig = plt.figure(figsize=(20, 20))

  def draw(self) -> None:
    self.defaultDraw()

  def defaultDraw(self) -> None:
    s = 0.2
    ax = plt.gca()
    plt.plot([0, 0], [0, 20], color='red', linewidth=2)
    plt.plot([20, 20], [0, 20], color='red', linewidth=2)
    plt.plot([0, 20], [0, 0], color='red', linewidth=2)
    plt.plot([0, 20], [20, 20], color='red', linewidth=2)
    for y in range(20):
      for x in range(20):
        num = (x+1)+20*y - 1
        if num == 0:
          plt.text(x + 0.5, y + 0.4, 'START', ha='center')
        elif num == 399:
          plt.text(x + 0.5, y + 0.4, 'GOAL', ha='center')
        else:
          plt.text(x + 0.5, y + 0.4, f'S{num}', size=14, ha='center')
        plt.plot([x, x+s], [y, y], color='red', linewidth=2)
        plt.plot([x-s+1, x+1], [y, y], color='red', linewidth=2)
        plt.plot([x, x], [y-s+1, y+1], color='red', linewidth=2)
        plt.plot([x+1, x+1], [y-s+1, y+1], color='red', linewidth=2)
        plt.plot([x, x], [y+s, y], color='red', linewidth=2)
        plt.plot([x+1, x+1], [y+s, y], color='red', linewidth=2)

    self.generateWall(0.1)

    # 描画範囲の設定と目盛りを消す設定
    ax.set_xlim(0, 20)
    ax.set_ylim(0, 20)
    plt.tick_params(labelbottom=False,
               labelleft=False,
               labelright=False,
               labeltop=False)

  def generateWall(self, freq: float) -> None:
    for y in range(20):
      for x in range(20):
        for n in range(4):
          r = rand.random()
          if r < freq:
            if n == 0:
              plt.plot([x, x], [y, y+1], color='red', linewidth=2)
            elif n == 1:
              plt.plot([x+1, x+1], [y, y+1], color='red', linewidth=2)
            elif n == 2:
              plt.plot([x, x+1], [y, y], color='red', linewidth=2)
            elif n == 3:
              plt.plot([x, x+1], [y+1, y+1], color='red', linewidth=2)


  def outputImage(self) -> None:
    self.fig.savefig("img.png")

  def showFig(self) -> None:
    self.fig.show()
