import gym
from environment import TSCEnv
from world import World
from generator import LaneVehicleGenerator
from agent.pdqn_agent import PDQNAgent
from metric import TravelTimeMetric
import argparse

import numpy as np
from Pareto.Pareto import Pareto

from Pareto.ReplayMemory import ReplayMemory
import gym
from gym import wrappers

import torch
from Pareto.metrics import metrics
import copy


import matplotlib.pyplot as plt

# parse args
parser = argparse.ArgumentParser(description='Run Example')
parser.add_argument('config_file', type=str, help='path of config file')
parser.add_argument('--thread', type=int, default=1, help='number of threads')
parser.add_argument('--steps', type=int, default=100, help='number of steps')
args = parser.parse_args()

# create world
world = World(args.config_file, thread_num=args.thread)

# create agents
agents = []
for i in world.intersections:
    action_space = gym.spaces.Discrete(len(i.phases))
    agents.append(PDQNAgent(
        action_space,
        LaneVehicleGenerator(world, i, ["lane_count"], in_only=True, average="road"),
        LaneVehicleGenerator(world, i, ["lane_delay"], in_only=True, average="all", negative=True),
        LaneVehicleGenerator(world, i, ["lane_waiting_count"], in_only=True, average="all", negative=True),

    ))

# create metric
metric = TravelTimeMetric(world)

# create env
env = TSCEnv(world, agents, metric)
"""
# simulate
obs = env.reset()
for i in range(args.steps):
    if i % 5 == 0:
        actions = env.action_space.sample()
    obs, rewards, dones, info = env.step(actions)
    #print(obs[0].shape)
    print(rewards)
    #print(info["metric"])

print("Final Travel Time is %.4f" % env.metric.update(done=True))
"""



################################################################################################################################################################
################################################################################
################################################################################
################################################################################
################################################################################


env = TSCEnv(world, agents, metric)
memory_capacity = 100
metr = metrics()
number_of_episodes = 20000
starting_learn = 50
number_of_states = agents[0].number_of_states

number_of_p_points = 5
number_of_actions = len(i.phases)
D = ReplayMemory((number_of_states,),size= memory_capacity, nO=2)
Pareto = Pareto(env=env,metrs=metr,number_of_states=number_of_states,number_of_actions=number_of_actions,step_start_learning = starting_learn,numberofeps = number_of_episodes,ReplayMem = D,
    number_of_p_points = number_of_p_points, epsilon_start = 1.,epsilon_decay = 0.99997,epsilon_min = 0.01,gamma = 1.,copy_every=50,ref_point = [-1,-2] )



number_of_objectives = 2
e=0




minibatch_size = 16
from collections import namedtuple


Transition = namedtuple('Transition',
                        ['state',
                         'action',
                         'reward',
                         'next_state',
                         'terminal'])
number_of_objectives = 2
MAX_STEPS = 2000

polIndex = 0
qtable = np.zeros((number_of_states, number_of_actions, 2),dtype=object)
starting_learn = 50

polDict = np.zeros((number_of_episodes, number_of_actions,number_of_p_points, number_of_objectives))
total_steps = 0
state = env.reset()
while e < number_of_episodes:
    
    step = 0
    
    terminal = False
    acumulatedRewards = [0,0]
    
    qcopy =[]
      #convert the state to a number
    #### to do: need to do this for multiples arrays of states (more than 1 crossroad), check other config file
    
    while terminal is False or step == MAX_STEPS:
        #one_hot_state[state] = 1
        #env.render()
        step += 1
        if total_steps > 2*starting_learn:
            #action selection step
            
            q_front = Pareto.q_front(state, n=number_of_p_points, use_target_network=False)
            
            hv = Pareto.compute_hypervolume(q_front[0],number_of_actions,np.array([-1,-2]))
            action = Pareto.e_greedy_action(hv)

        else:    
            action =  env.action_space.sample()
        
        #take the action
        print(f'action: {action}')
        if isinstance(action, np.int64):
            action = int(action)  # Convert numpy.int64 to integer
            action = np.array([action])
        next_state, reward, terminal, _ = env.step(action)
        reward = reward[0]
        
        
        

        acumulatedRewards[0] += reward[0]
        acumulatedRewards[1] += reward[1]
        
        
        
        
        

        #add transition to memory
        
        
        t = Transition(state=state[0],
                       action=action,
                       reward=reward,
                       next_state=next_state[0],
                       terminal=terminal)
        D.add(t)
        

        
        if total_steps > starting_learn:
            #sample from memory if is learning
            minibatch = D.sample(minibatch_size)
            
            minibatch_non_dominated = []
            minibatch_states = []
            minibatch_actions = []
           
            minibatch_rew_normalized = minibatch.reward

            #Calculate q front for the minibatch    
            batch_q_front_next = Pareto.q_front(minibatch.next_state, n=number_of_p_points, use_target_network=True)
            
            #take the sample point of maximum value of the q_front
            b_max = np.argmax(batch_q_front_next[:, :, :, 1], axis=2)
            b_indices, s_indices = np.indices(batch_q_front_next.shape[:2])
            batch_q_front_next = batch_q_front_next[b_indices, s_indices, b_max]
            
            for batch_index, approximations in enumerate(batch_q_front_next):
                
                
                if minibatch.terminal[batch_index] is True:
                    #find the index of the reward that is closest to the reward of the terminal state
                    
                    rew_index = np.abs(approximations[:, 0] - minibatch_rew_normalized[batch_index][0]).argmin()
                    
                    non_dominated = approximations
                    # update all rows, execpt the one at rew_index, with the second reward with -1
                    non_dominated[:rew_index, -1] = minibatch_rew_normalized[batch_index][-1]

                    # Set the row at min_ to the new row
                    non_dominated[rew_index, :] = minibatch_rew_normalized[batch_index]

                    # Update rows starting from min_+1 (inclusive) in the last column with -1
                    non_dominated[rew_index+1:, -1] = -1
                 
                else:
                    non_dominated = approximations
                
                    
                minibatch_non_dominated.append(non_dominated)
                
                
                #repeat 4 times the ohe state vector
                states = np.tile(minibatch.state[batch_index], (number_of_actions, *([1]*1)))
                
                minibatch_states.append(states)
                
                #repeat the action 4 times for training
                actions = np.repeat(minibatch.action[batch_index], number_of_actions)
                minibatch_actions.append(actions)
            
            minibatch_actions = np.concatenate(minibatch_actions)
            minibatch_states = np.concatenate(minibatch_states)
            minibatch_non_dominated = np.concatenate(minibatch_non_dominated)
            
            #shape = (number of objectives, number_of_actions * minibatch_size)

            minibatch_non_dominated_1st_objective = minibatch_non_dominated[:, -1:]    #target_values
            minibatch_non_dominated_2nd_objective = minibatch_non_dominated[:, :-1] 
            e_loss = Pareto.nd_estimator.update(minibatch_non_dominated_1st_objective.astype(np.float32),
                                                    minibatch_states.astype(np.float32),
                                                    minibatch_non_dominated_2nd_objective.astype(np.float32),
                                                    minibatch_actions.astype(np.float32),
                                                    step=total_steps)
            

            Pareto.rew_estim.update(minibatch.reward.astype(np.float32),
                                        minibatch.state,
                                        minibatch.action.astype(np.long),
                                        step=total_steps)
        #state = next_state
            
            
          
            
        
        
    metr.rewards1.append(acumulatedRewards[0])
    metr.rewards2.append(acumulatedRewards[1])
    metr.episodes.append(e)

    try:
        episodeqtable = copy.deepcopy(q_front)
        try:
            polDict[e] = episodeqtable
        except ValueError:
            pass
    except NameError:
        pass
        

    
    

    
    
    print("episode = " + str(e) +  " | rewards = [" + str(acumulatedRewards[0]) + "," + str(acumulatedRewards[1]) + "]")
        
    Pareto.epsilon_decrease()
    
    e+=1
    total_steps = total_steps + step


