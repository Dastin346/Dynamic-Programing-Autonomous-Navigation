import numpy as np
import gym
import glob
from utils import *
from DP import DPSolver, DPSolverRandomEnv

MF = 0 # Move Forward
TL = 1 # Turn Left
TR = 2 # Turn Right
PK = 3 # Pickup Key
UD = 4 # Unlock Door

def doorkey_problem(env):
    '''
    You are required to find the optimal path in
        doorkey-5x5-normal.env
        doorkey-6x6-normal.env
        doorkey-8x8-normal.env
        
        doorkey-6x6-direct.env
        doorkey-8x8-direct.env
        
        doorkey-6x6-shortcut.env
        doorkey-8x8-shortcut.env
        
    Feel Free to modify this fuction
    '''
    optim_act_seq = [TL, MF, PK, TL, UD, MF, MF, MF, MF, TR, MF]
    return optim_act_seq


def partA():
    paths = glob.glob('./envs/*.env')
    for env_path in paths:
        env, info = load_env(env_path) # load an environment
        solver = DPSolver(env, info)
        seq = solver.Solve()
        draw_gif_from_seq(seq, load_env(env_path)[0], './gif/{}.gif'.format(env_path.split('/')[-1].split('.')[0])) # draw a GIF & save
    
    
def partB():
    env_folder = './envs/random_envs'
    solver = DPSolverRandomEnv(env_folder)
    for i in range(36):
        solver.ComputePolicy(i)
        print('computing {}-th env'.format(i))
    # generate visualizations 
    for i in range(36):
        env, info, env_path = load_random_env(env_folder, i)
        seq = solver.GetPath(env, info)
        draw_gif_from_seq(seq, load_random_env(env_folder, i)[0], './gif/{}.gif'.format(env_path.split('/')[-1].split('.')[0])) # draw a GIF & save
    

if __name__ == '__main__':
    print('Known Environment')
    partA()
    print('Random Environment')
    partB()

    print('DONE')


        
        
    
