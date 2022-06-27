from pickle import TRUE
import numpy as np
import createFigure as cf
import agent as ag
import random as rand
import matplotlib.pyplot as plt
import time
import os
import sys
import datetime
import csv
class Main:
  def __init__(self) -> None:
    self.MAZEX = 40
    self.MAZEY = 40
    self.MAZEGOALXRANGE = 5 # ゴールが出現するX範囲
    self.MAZEGOALYRANGE = 5 # ゴールが出現するY範囲
    self.states = [(x, y) for x in range(self.MAZEX) for y in range(self.MAZEY)]
    self.maze = {}
    for s in self.states: self.maze[s] = [np.nan] * 4 #[上,右,下,左]
    self.normalEpisode = []
    self.bestEpisode = []
    self.psoEpisode = []
    self.L = 15000 # 各エピソードの行動上限数
    self.d = 0.999
    self.w = 0 # PSOで用いる変化量の重み
    self.c1 = 2.2 # PSOで用いるPbest-qの重み
    self.c2 = 2.2 # PSOで用いるGbest-qの重み
    self.r1 = rand.random()
    self.r2 = rand.random()
    self.outerWallCount = 0
    self.innerWallCount = 0
    self.pathname = ""

  def startTrain(self, epsilon, num=1, agentNum=1, type="Normal", train=True, epq=1): # epq qを取引する間のepisode数
    times = []
    agents = []
    actionCounts = []
    pbQ = [] # 自己最良Q値
    gbQ = {} # 全体最良Q値
    pbE = [] # 自己最良Q値評価値
    gbE = -10*10 # 全体最良Q値評価値
    for i in range(agentNum): agents.append(ag.Agent(self.maze)) # Agentのインスタンスを作成
    i = 0
    if type=="PSO":
      for l in range(agentNum): pbE.append(-10**10)
      for m in self.maze: # 自己最良Q値と全体最良Q値を-∞に初期化
        for a in agents[0].actions:
          gbQ[(m, a)] = 0
          for l in range(agentNum):
            pbQ.append({})
            pbQ[l][(m, a)] = 0
    while i < num+1: # 指定エピソードループ
      start = time.perf_counter()
      episodes = []
      pkActionCounts = [] # 各エージェントの行動回数を一時的に格納
      insQ = {} # Q値の一時的な保存
      bestQ = {} # best/averageで使用する1エピソードの最高のQ値を格納
      bestE = -10**100 # best/averageで使用する1エピソードの最高のQ値の評価値を格納
      if type=="PSO":
        for m in self.maze:
          for a in agents[0].actions:
            insQ[(m, a)] = 0
      if i == num: # 最後の一回に探索結果を出力する部分
        train = False
        epsilon = 0
      print("\r"+type+"の強化学習を実行中... episode: "+str(i),end="")
      for k in range(agentNum): # 各エージェントの処理
        for epqs in range(epq):
          ks = [] # best/average/PSOで使用.報酬を一時的に保存
          qs = {} # best/averageで使用.状態と行動に対してのqを一時的に保存する変数 (s, a) => q
          episodes.append([])
          agents[k].moveStart()
          agents[k].chengeGoal(self.MAZEGOALXRANGE, self.MAZEGOALYRANGE)
          for l in range(self.L): # 1エピソードの行動
            action = agents[k].chooseAction(epsilon)
            reward, state = agents[k].move(action)
            if l+1 == self.L: reward = -1
            episodes[k].append((state, action))
            if type=="Normal":
              if train: agents[k].train(action, reward)
              if reward == 100:
                pkActionCounts.append(l)
                break
            elif type=="Best" or type=="Average":
              if train:
                q = agents[k].train(action, reward)
                qs[(state, action)] = q
              ks.append(reward)
              if reward != 0:
                pkActionCounts.append(l)
                e = 0
                for iks in range(len(ks)):
                  e += self.d**(self.L-iks)*ks[iks]
                # step2-5
                if e > bestE:
                  bestE = e
                  for iqs in qs:
                    bestQ[iqs] = qs[iqs]
                break
            elif type=="PSO":
              if train:
                q = agents[k].train(action, reward)
                if q > insQ[(state, action)]:
                  insQ[(state, action)] = q
              ks.append(reward)
              if reward != 0:
                pkActionCounts.append(l)
                e = 0
                for iks in range(len(ks)):
                  e += (self.d**(self.L-iks))*ks[iks]
                if e > pbE[k]: # step2-5
                  for iq in insQ: # 自己最良Q値の更新
                    pbQ[k][iq] = insQ[iq]
                  pbE[k] = e # 自己最良Q値の評価値の更新
                if e > gbE: # step2-6
                  for iq in insQ: # 全体最良Q値の更新
                    gbQ[iq] = insQ[iq]
                  gbE = e # 全体最良Q値の評価値の更新
                break


      # step3
      for k in range(agentNum):
        if type=="Best":
          agents[k].setQ(bestQ)
        elif type=="Average":
          repQ = {}
          for iq in agents[k].q:
            bq = bestQ.get(iq)
            if bq != None:
              repQ[iq] = (agents[k].q[iq] + bq) / 2
          agents[k].setQ(repQ)
        elif type=="PSO":
          for s in agents[k].v:
            avs = self.w * agents[k].v[s] + self.c1 * self.r1 * (pbQ[k][s] - agents[k].q[s]) + self.c2 * self.r2 * (gbQ[s] - agents[k].q[s])
            agents[k].q[s] += avs
            if agents[k].q[s] > 100: print(agents[k].q[s])

      # アクション数の計測
      for _ in range(agentNum*epq):
        if len(pkActionCounts) > 0:
          actionCounts.append(min(pkActionCounts))
        else:
          actionCounts.append(self.L)

        if i != num: times.append((time.perf_counter() - start)/(agentNum*epq))
      i += agentNum*epq
    print("\r")
    return actionCounts, times, episodes

  def generateMaze(self, difficulty: float) -> dict:
    if self.MAZEX <= self.MAZEGOALXRANGE or self.MAZEY <= self.MAZEGOALYRANGE:
      print('Error: 迷路サイズはゴール範囲より大きくしてください。', file=sys.stderr)
      sys.exit(1)
    print('迷路を作成します。')
    while True:
      self.generateWall(difficulty)
      if self.checkGoal():
        break
      else:
        print('たどり着けないゴールになる可能性のある座標が存在したため再度迷路を生成します。')
        time.sleep(1)
    print('迷路を作成しました。\n')
    print('サイズ: ' + str(self.MAZEX) + ' ✖️ ' + str(self.MAZEY))
    print('壁生成難易度: ' + str(difficulty))
    print('内壁数: ' + str(self.innerWallCount) + ' 外壁数: ' + str(self.outerWallCount) + '\n')
    ink = ""
    while True:
      ink = input("迷路の環境を保存しますか？Yes(Y) or No(N): ")
      if ink == "Y" or ink == "N": break
    if ink == "Y":
      print('迷路の構造をdump中。')
      dt = datetime.datetime.now()
      d = dt.strftime('%Y%m%d%H%M%S')
      self.pathname = 'environment-'+ d + '/'
      os.makedirs(self.pathname)
      np.save(self.pathname +'maze.npy', self.maze)
      # 迷路の出力
      print('迷路の画像を出力中。')
      figure = cf.CreateFigure(obj.MAZEX, obj.MAZEY)
      figure.draw(obj.maze, drawWall = True, plotCoordinate = False)
      figure.outputImage(name='environment-'+ d +'/maze.png')
      print('迷路の環境を保存しました。')
  def generateWall(self, difficulty: float) -> dict:
    # 外枠を作る
    for s in self.states:
      x, y = s
      n = (x+1)+self.MAZEX*y
      self.outerWallCount += 1
      print("\r 外壁を作成中... progress: "+'{:.1f}'.format(n/len(self.states)*100)+"%",end="")
      if x == 0:
        self.maze[s][3] = 1
      if x == self.MAZEX - 1:
        self.maze[s][1] = 1
      if y == 0:
        self.maze[s][2] = 1
      if y == self.MAZEY - 1:
        self.maze[s][0] = 1
    print("\r")
    # 壁を作る
    for s in self.states:
      x, y = s
      num = ((x+1)+self.MAZEX*y - 1) * 2
      for n in range(2):
        print("\r 内壁を作成中... progress: "+'{:.1f}'.format(((num+n+1)/(len(self.states)*2)*100))+"%",end="")
        r = rand.random()
        if r < difficulty:
          self.innerWallCount += 1
          self.maze[s][n] = 1
          if n == 0 and s[1] != self.MAZEY - 1:
            self.maze[(s[0],s[1] + 1)][2] = 1
          if n == 1 and s[0] != self.MAZEX - 1:
            self.maze[(s[0] + 1,s[1])][3] = 1
    print("\r")
    # 全方位囲まれていた場合一部の壁を除去
    for s in self.states:
      if self.maze[s].count(1) == 4:
        self.innerWallCount -= 1
        x, y = s
        if y == self.MAZEY - 1:
          self.maze[s][2] = 0
          self.maze[(x, y - 1)][0] = 0
          print(' removed the wall between ' + str(s) + ' and ' + str((x, y - 1)))
        elif x == self.MAZEX - 1:
          self.maze[s][3] = 0
          self.maze[(x - 1, y)][1] = 0
          print(' removed the wall between ' + str(s) + ' and ' + str((x - 1, y)))
        elif y == 0:
          self.maze[s][0] = 0
          self.maze[(x, y + 1)][2] = 0
          print(' removed the wall between ' + str(s) + ' and ' + str((x, y + 1)))
        elif x == 0:
          self.maze[s][1] = 0
          self.maze[(x + 1, y)][3] = 0
          print(' removed the wall between ' + str(s) + ' and ' + str((x + 1, y)))
        else:
          self.maze[s][0] = 0
          self.maze[(x, y + 1)][2] = 0
          print(' removed the wall between ' + str(s) + ' and ' + str((x, y + 1)))
  def checkGoal(self) -> bool:
    return True

  def loadMaze(self):
    file_name = input("環境名を入力: ")
    try:
      maze = np.load(file_name + "/maze.npy", allow_pickle=True)
      insm = dict(enumerate(maze.flatten(),1))[1]
      if type(insm) == dict:
        self.maze = insm
        self.pathname = file_name
      else:
        print('Error: 不正な迷路構造です。', file=sys.stderr)
        sys.exit(1)
    except:
      print('Error: 不正な環境名です。', file=sys.stderr)
      sys.exit(1)


# maze: {(0, 0): [nan, nan, 1, 1] ...}という形。1は壁を表す
obj = Main()
inf = ""
while True:
  inf = input("迷路の環境を読み込みますか？Yes(Y) or No(N): ")
  if inf == "Y" or inf == "N": break
if inf == "Y":
  obj.loadMaze()
else:
  obj.generateMaze(0)

figure = cf.CreateFigure(obj.MAZEX, obj.MAZEY)


for k in range(5):
  print(str(k+1)+"回目の強化学習を実行中...")
  actions3, times3, episode3 = obj.startTrain(epsilon=0.2, num=5000, agentNum=4, epq=5, type="PSO")
  with open(obj.pathname + '/pso-step.csv', 'a') as f:
    writer = csv.writer(f)
    p = [actions3]
    for sk in range(len(p)):
      result = []
      minl = 10**10
      for i in range(len(p[sk])):
        if minl > p[sk][i]:
          minl = p[sk][i]
          result.append(p[sk][i])
        else:
          result.append(minl)
      writer.writerow(result)

  with open(obj.pathname + '/pso-time.csv', 'a') as f:
    writer = csv.writer(f)
    for sk in range(len(p)):
      p = [times3]
      result = []
      minl = 10**10
      for i in range(len(p[sk])):
        if minl > p[sk][i]:
          minl = p[sk][i]
          result.append(p[sk][i])
        else:
          result.append(minl)
      writer.writerow(result)
'''
figure.lineGraph(
  [times0, times1, times2, times3],
  [np.arange(2500), np.arange(2500), np.arange(2500), np.arange(2500)],
  ["Normal", "Best", "Average", "PSO"]
)
'''
