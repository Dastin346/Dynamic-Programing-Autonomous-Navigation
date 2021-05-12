from os import kill
import gym_minigrid
import numpy as np
from collections import deque
from copy import deepcopy
from numpy.core.defchararray import islower
from numpy.lib.arraysetops import isin
from utils import step_cost, step, load_random_env

MF = 0 # Move Forward
TL = 1 # Turn Left
TR = 2 # Turn Right
PK = 3 # Pickup Key
UD = 4 # Unlock Door


class DPSolver:
    def __init__(self, env, info) -> None:
        super().__init__()
        self.env = env
        self.info = info
        self.init_state = self.InitState(env, info)
        # col, row, direction, hasKey, is_locked
        self.costMap = np.full((env.grid.width, env.grid.height, 4, 2, 2), np.inf)
        self.policyMap = np.full((env.grid.width, env.grid.height, 4, 2, 2), -1, dtype=np.int)

    def InitState(self, env, info):
        vec = info['init_agent_dir']
        door = env.grid.get(info['door_pos'][0], info['door_pos'][1])
        is_locked = door.is_locked
        if vec[0] == 0:
            dir = 1 + vec[1]
        else:
            dir = 2 - vec[0]
        return (info['init_agent_pos'][0], info['init_agent_pos'][1], dir, 1 if env.carrying else 0, 1 if is_locked else 0)

    def State2Vec(self, dir):
        vectors = {
            0: (0, -1),
            1: (1, 0),
            2: (0, 1),
            3: (-1, 0),
            }
        return vectors[dir]

    def IsMovable(self, i, j, hasKey, is_locked):
        if self.env.grid.get(i, j) is None:
            return True
        if isinstance(self.env.grid.get(i, j), gym_minigrid.minigrid.Key) and hasKey==1:
            return True
        if isinstance(self.env.grid.get(i, j), gym_minigrid.minigrid.Door) and is_locked==0:
            return True
        return False
    
    def IsInitState(self, state):
        return state==self.init_state

    def GetCost(self, i, j, dir, hasKey, is_locked):
        return self.costMap[i][j][dir][hasKey][is_locked]
    
    def SetCost(self, i, j, dir, hasKey, is_locked, value):
        self.costMap[i][j][dir][hasKey][is_locked] = value
    
    def SetPolicy(self, i, j, dir, hasKey, is_locked, value):
        self.policyMap[i][j][dir][hasKey][is_locked] = value

    def ExpandOpenNodes(self, frontier):
        start_reached = False
        for _ in range(len(frontier)):
            i, j, dir, hasKey, is_locked = frontier.popleft()
            cost = self.GetCost(i, j, dir, hasKey, is_locked)
            
            # turn left
            if self.GetCost(i, j, (dir+1)%4, hasKey, is_locked) > cost + step_cost(TL):
                self.SetCost(i, j, (dir+1)%4, hasKey, is_locked, cost + step_cost(TL))
                self.SetPolicy(i, j, (dir+1)%4, hasKey, is_locked, TL)
                frontier.append((i, j, (dir+1)%4, hasKey, is_locked))
                if self.IsInitState(frontier[-1]):
                    return True
            
            # turn right
            if self.GetCost(i, j, (dir-1)%4, hasKey, is_locked) > cost + step_cost(TR):
                self.SetCost(i, j, (dir-1)%4, hasKey, is_locked, cost + step_cost(TR))
                self.SetPolicy(i, j, (dir-1)%4, hasKey, is_locked, TR)
                frontier.append((i, j, (dir-1)%4, hasKey, is_locked))
                if self.IsInitState(frontier[-1]):
                    return True
            
            # move forward
            deltai, deltaj = self.State2Vec(dir)
            if self.IsMovable(i-deltai, j-deltaj, hasKey, is_locked) and self.GetCost(i-deltai, j-deltaj, dir, hasKey, is_locked)> cost + step_cost(MF):
                self.SetCost(i-deltai, j-deltaj, dir, hasKey, is_locked, cost + step_cost(MF))
                self.SetPolicy(i-deltai, j-deltaj, dir, hasKey, is_locked, MF)
                frontier.append((i-deltai, j-deltaj, dir, hasKey, is_locked))
                if self.IsInitState(frontier[-1]):
                    return True
            
            # unlock door
            if hasKey==1 and is_locked==0 and isinstance(self.env.grid.get(i+deltai, j+deltaj), gym_minigrid.minigrid.Door):
                if self.GetCost(i, j, dir, hasKey, 1) > cost + step_cost(UD):
                    self.SetCost(i, j, dir, hasKey, 1, cost + step_cost(UD))
                    self.SetPolicy(i, j, dir, hasKey, 1, UD)
                    frontier.append((i, j, dir, hasKey, 1))
                    if self.IsInitState(frontier[-1]):
                        return True

            # pickup key
            if hasKey==1 and isinstance(self.env.grid.get(i+deltai, j+deltaj), gym_minigrid.minigrid.Key):
                if self.GetCost(i, j, dir, 0, is_locked) > cost + step_cost(PK):
                    self.SetCost(i, j, dir, 0, is_locked, cost + step_cost(PK))
                    self.SetPolicy(i, j, dir, 0, is_locked, PK)
                    frontier.append((i, j, dir, 0, is_locked))
                    if self.IsInitState(frontier[-1]):
                        return True

        print('current cost: {}'.format(cost))            
        return start_reached

    def ComputePolicy(self):
        ei, ej = self.info['goal_pos']
        frontier = deque([])
        for dir in range(4):
            for hasKey in range(2):
                for is_locked in range(2):
                    self.costMap[ei][ej][dir][hasKey][is_locked] = 0
                    frontier.append((ei, ej, dir, hasKey, is_locked))
                    
        start_reached = False
        while not start_reached:
            start_reached = self.ExpandOpenNodes(frontier)

    def GetPath(self):
        path = []
        env = deepcopy(self.env)
        i, j, dir, hasKey, is_locked = self.init_state
        done = False
        while not done:
            action = self.policyMap[i][j][dir][hasKey][is_locked]
            path.append(action)
            _, done = step(env, action)
            i, j = env.agent_pos
            vec = env.dir_vec
            if vec[0] == 0:
                dir = 1 + vec[1]
            else:
                dir = 2 - vec[0]
            hasKey = 1 if env.carrying else 0
            is_locked = 1 if env.grid.get(self.info['door_pos'][0], self.info['door_pos'][1]).is_locked else 0
        return path
    
    def Solve(self):
        self.ComputePolicy()
        return self.GetPath()
        

class DPSolverRandomEnv(DPSolver):
    def __init__(self, env_folder) -> None:
        self.env_folder = env_folder
        # col, row, direction, hasKey, is_locked
        self.costMap = np.full((8, 8, 4, 2, 2, 2, 3, 3), np.inf)
        self.policyMap = np.full((8, 8, 4, 2, 2, 2, 3, 3), -1, dtype=np.int)

    def InitState(self, env, info):
        locked_1 = 0 if info['door_open'][0] else 1
        locked_2 = 0 if info['door_open'][1] else 1
        hasKey = 1 if env.carrying else 0
        key_idx = self.GetKeyLocIdx(info['key_pos'])
        goal_idx = self.GetGoalLocIdx(info['goal_pos'])
        return (3, 5, 0, hasKey, locked_1, locked_2, key_idx, goal_idx)

    def GetKeyLoc(self, key_idx):
        key_loc = {
            0: (1, 1),
            1: (2, 3),
            2: (1, 6)
        }
        return key_loc[key_idx]
    
    def GetKeyLocIdx(self, key_loc):
        key_loc = tuple(key_loc)
        if key_loc == (1, 1):
            return 0
        if key_loc == (2, 3):
            return 1
        if key_loc == (1, 6):
            return 2

    def GetGoalLoc(self, goal_idx):
        goal_loc = {
            0: (5, 1),
            1: (6, 3),
            2: (5, 6)
        }
        return goal_loc[goal_idx]

    def GetGoalLocIdx(self, goal_loc):
        goal_loc = tuple(goal_loc)
        if goal_loc == (5, 1):
            return 0
        if goal_loc == (6, 3):
            return 1
        if goal_loc == (5, 6):
            return 2
    
    def IsMovable(self, i, j, hasKey, locked_1, locked_2):
        if self.env.grid.get(i, j) is None:
            return True
        if isinstance(self.env.grid.get(i, j), gym_minigrid.minigrid.Key) and hasKey==1:
            return True
        if (i, j)==tuple(self.info['door_pos'][0]) and locked_1==0:
            return True
        if (i, j)==tuple(self.info['door_pos'][1]) and locked_2==0:
            return True
        return False

    def IsDPTerminal(self, key_idx, goal_idx):
        return (self.policyMap[3, 5, 0, :, :, :, key_idx, goal_idx] >= 0).all()

    def GetCost(self, i, j, dir, hasKey, locked_1, locked_2, key_idx, goal_idx):
        return self.costMap[i][j][dir][hasKey][locked_1][locked_2][key_idx][goal_idx]
    
    def SetCost(self, i, j, dir, hasKey, locked_1, locked_2, key_idx, goal_idx, value):
        self.costMap[i][j][dir][hasKey][locked_1][locked_2][key_idx][goal_idx] = value
    
    def SetPolicy(self, i, j, dir, hasKey, locked_1, locked_2, key_idx, goal_idx, value):
        self.policyMap[i][j][dir][hasKey][locked_1][locked_2][key_idx][goal_idx] = value

    def ExpandOpenNodes(self, frontier):
        start_reached = False
        for _ in range(len(frontier)):
            i, j, dir, hasKey, locked_1, locked_2, key_idx, goal_idx = frontier.popleft()
            cost = self.GetCost(i, j, dir, hasKey, locked_1, locked_2, key_idx, goal_idx)
            
            # turn left
            if self.GetCost(i, j, (dir+1)%4, hasKey, locked_1, locked_2, key_idx, goal_idx) > cost + step_cost(TL):
                self.SetCost(i, j, (dir+1)%4, hasKey, locked_1, locked_2, key_idx, goal_idx, cost + step_cost(TL))
                self.SetPolicy(i, j, (dir+1)%4, hasKey, locked_1, locked_2, key_idx, goal_idx, TL)
                frontier.append((i, j, (dir+1)%4, hasKey, locked_1, locked_2, key_idx, goal_idx))
            
            # turn right
            if self.GetCost(i, j, (dir-1)%4, hasKey, locked_1, locked_2, key_idx, goal_idx) > cost + step_cost(TR):
                self.SetCost(i, j, (dir-1)%4, hasKey, locked_1, locked_2, key_idx, goal_idx, cost + step_cost(TR))
                self.SetPolicy(i, j, (dir-1)%4, hasKey, locked_1, locked_2, key_idx, goal_idx, TR)
                frontier.append((i, j, (dir-1)%4, hasKey, locked_1, locked_2, key_idx, goal_idx))
            
            # move forward
            deltai, deltaj = self.State2Vec(dir)
            if self.IsMovable(i-deltai, j-deltaj, hasKey, locked_1, locked_2) and self.GetCost(i-deltai, j-deltaj, dir, hasKey, locked_1, locked_2, key_idx, goal_idx)> cost + step_cost(MF):
                self.SetCost(i-deltai, j-deltaj, dir, hasKey, locked_1, locked_2, key_idx, goal_idx, cost + step_cost(MF))
                self.SetPolicy(i-deltai, j-deltaj, dir, hasKey, locked_1, locked_2, key_idx, goal_idx, MF)
                frontier.append((i-deltai, j-deltaj, dir, hasKey, locked_1, locked_2, key_idx, goal_idx))
            
            # unlock door
            if hasKey==1 and (i+deltai, j+deltaj)==tuple(self.info['door_pos'][0]) and locked_1==0:
                if self.GetCost(i, j, dir, 1, 1, locked_2, key_idx, goal_idx) > cost + step_cost(UD):
                    self.SetCost(i, j, dir, 1, 1, locked_2, key_idx, goal_idx, cost + step_cost(UD))
                    self.SetPolicy(i, j, dir, 1, 1, locked_2, key_idx, goal_idx, UD)
                    frontier.append((i, j, dir, 1, 1, locked_2, key_idx, goal_idx))
            
            if hasKey==1 and (i+deltai, j+deltaj)==tuple(self.info['door_pos'][1]) and locked_2==0:
                if self.GetCost(i, j, dir, 1, locked_1, 1, key_idx, goal_idx) > cost + step_cost(UD):
                    self.SetCost(i, j, dir, 1, locked_1, 1, key_idx, goal_idx, cost + step_cost(UD))
                    self.SetPolicy(i, j, dir, 1, locked_1, 1, key_idx, goal_idx, UD)
                    frontier.append((i, j, dir, 1, locked_1, 1, key_idx, goal_idx))

            # pickup key
            if hasKey==1 and isinstance(self.env.grid.get(i+deltai, j+deltaj), gym_minigrid.minigrid.Key):
                if self.GetCost(i, j, dir, 0, locked_1, locked_2, key_idx, goal_idx) > cost + step_cost(PK):
                    self.SetCost(i, j, dir, 0, locked_1, locked_2, key_idx, goal_idx, cost + step_cost(PK))
                    self.SetPolicy(i, j, dir, 0, locked_1, locked_2, key_idx, goal_idx, PK)
                    frontier.append((i, j, dir, 0, locked_1, locked_2, key_idx, goal_idx))
        
        if self.IsDPTerminal(key_idx, goal_idx):
            return True
        # print('current cost: {}'.format(cost))            
        return start_reached

    def ComputePolicy(self, env_idx):
        frontier = deque([])
        env, info, _ = load_random_env(self.env_folder, env_idx)
        self.env = env
        self.info = info
        self.init_state = self.InitState(env, info)
        key_idx = self.init_state[-2]
        goal_idx = self.init_state[-1]
        ei, ej = self.info['goal_pos']
        for dir in range(4):
            for hasKey in range(2):
                for locked_1 in range(2):
                    for locked_2 in range(2):
                        self.costMap[ei][ej][dir][hasKey][locked_1][locked_2][key_idx][goal_idx] = 0
                        frontier.append((ei, ej, dir, hasKey, locked_1, locked_2, key_idx, goal_idx))
                    
        start_reached = False
        while not start_reached:
            start_reached = self.ExpandOpenNodes(frontier)

    def GetPath(self, env, info):
        path = []
        init_state = self.InitState(env, info)
        i, j, dir, hasKey, locked_1, locked_2, key_idx, goal_idx = init_state
        done = False
        while not done:
            action = self.policyMap[i][j][dir][hasKey][locked_1][locked_2][key_idx][goal_idx]
            path.append(action)
            _, done = step(env, action)
            i, j = env.agent_pos
            vec = env.dir_vec
            if vec[0] == 0:
                dir = 1 + vec[1]
            else:
                dir = 2 - vec[0]
            hasKey = 1 if env.carrying else 0
            locked_1 = 1 if env.grid.get(self.info['door_pos'][0][0], self.info['door_pos'][0][1]).is_locked else 0
            locked_2 = 1 if env.grid.get(self.info['door_pos'][1][0], self.info['door_pos'][1][1]).is_locked else 0
        return path
    
    def Solve(self):
        for i in range(36):
            self.ComputePolicy(i)
            print('computing {}-th env'.format(i))