from __future__ import print_function
from collections import defaultdict
from itertools import count
import numpy as np
import math
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.distributions
from torch.autograd import Variable

import copy
import pickle 
import os

class Environment(object):
    """
    The Tic-Tac-Toe Environment
    """
    # possible ways to win
    win_set = frozenset([(0,1,2), (3,4,5), (6,7,8), # horizontal
                         (0,3,6), (1,4,7), (2,5,8), # vertical
                         (0,4,8), (2,4,6)])         # diagonal
    # statuses
    STATUS_VALID_MOVE = 'valid'
    STATUS_INVALID_MOVE = 'inv'
    STATUS_WIN = 'win'
    STATUS_TIE = 'tie'
    STATUS_LOSE = 'lose'
    STATUS_DONE = 'done'

    def __init__(self):
        self.reset()

    def reset(self):
        """Reset the game to an empty board."""
        self.grid = np.array([0] * 9) # grid
        self.turn = 1                 # whose turn it is
        self.done = False             # whether game is done
        return self.grid

    def render(self):
        """Print what is on the board."""
        map = {0:'.', 1:'x', 2:'o'} # grid label vs how to plot
        print(''.join(map[i] for i in self.grid[0:3]))
        print(''.join(map[i] for i in self.grid[3:6]))
        print(''.join(map[i] for i in self.grid[6:9]))
        print('====')

    def check_win(self):
        """Check if someone has won the game."""
        for pos in self.win_set:
            s = set([self.grid[p] for p in pos])
            if len(s) == 1 and (0 not in s):
                return True
        return False

    def step(self, action):
        """Mark a point on position action."""
        assert type(action) == int and action >= 0 and action < 9
        # done = already finished the game
        if self.done:
            return self.grid, self.STATUS_DONE, self.done
        # action already have something on it
        if self.grid[action] != 0:
            return self.grid, self.STATUS_INVALID_MOVE, self.done
        # play move
        self.grid[action] = self.turn
        if self.turn == 1:
            self.turn = 2
        else:
            self.turn = 1
        # check win
        if self.check_win():
            self.done = True
            return self.grid, self.STATUS_WIN, self.done
        # check tie
        if all([p != 0 for p in self.grid]):
            self.done = True
            return self.grid, self.STATUS_TIE, self.done
        return self.grid, self.STATUS_VALID_MOVE, self.done

    def random_step(self):
        """Choose a random, unoccupied move on the board to play."""
        pos = [i for i in range(9) if self.grid[i] == 0]
        move = random.choice(pos)
        return self.step(move)

    def play_against_random(self, action):
        """Play a move, and then have a random agent play the next move."""
        # state, status, done = self.step(action)
        # if not done and self.turn == 2:
        #     state, s2, done = self.random_step()
        #     if done:
        #         if s2 == self.STATUS_WIN:
        #             status = self.STATUS_LOSE
        #         elif s2 == self.STATUS_TIE:
        #             status = self.STATUS_TIE
        #         else:
        #             raise ValueError("???")
        state, status, done = self.step(action)
        if not done:
            state, s2, done = self.random_step()
            if done:
                if s2 == self.STATUS_WIN:
                    status = self.STATUS_LOSE
                elif s2 == self.STATUS_TIE:
                    status = self.STATUS_TIE
                else:
                    raise ValueError("???")

        return state, status, done

    def play_against_self(self, policy, action):
        state, status, done = self.step(action)
        if not done:
            action_comp, logprob_comp = select_action(policy, state)
            state, s2, done = self.step(action_comp)
            if done:
                if s2 == self.STATUS_WIN:
                    status = self.STATUS_LOSE
                elif s2 == self.STATUS_TIE:
                    status = self.STATUS_TIE
                else:
                    raise ValueError("???")

        return state, status, done

class Policy(nn.Module):
    """
    The Tic-Tac-Toe Policy
    """
    def __init__(self, input_size=27, hidden_size=64, output_size=9):
        super(Policy, self).__init__()
        # TODO
        self.hidden_size = hidden_size
        self.linear1 = torch.nn.Linear(input_size, hidden_size)
        self.linear2 = torch.nn.Linear(hidden_size, output_size)

    def forward(self, x):
        # TODO
        h_relu = F.relu(self.linear1(x))
        y_pred = self.linear2(h_relu)
        return F.softmax(y_pred, dim=-1)

def select_action(policy, state):
    """Samples an action from the policy at the state."""
    state = torch.from_numpy(state).long().unsqueeze(0)
    state = torch.zeros(3,9).scatter_(0,state,1).view(1,27)
    pr = policy(Variable(state))
    m = torch.distributions.Categorical(pr) 
    action = m.sample()
    log_prob = torch.sum(m.log_prob(action))
    return action.data[0], log_prob

def compute_returns(rewards, gamma=1.0):
    """
    Compute returns for each time step, given the rewards
      @param rewards: list of floats, where rewards[t] is the reward
                      obtained at time step t
      @param gamma: the discount factor
      @returns list of floats representing the episode's returns
          G_t = r_t + \gamma r_{t+1} + \gamma^2 r_{t+2} + ... 

    >>> compute_returns([0,0,0,1], 1.0)
    [1.0, 1.0, 1.0, 1.0]
    >>> compute_returns([0,0,0,1], 0.9)
    [0.7290000000000001, 0.81, 0.9, 1.0]
    >>> compute_returns([0,-0.5,5,0.5,-10], 0.9)
    [-2.5965000000000003, -2.8850000000000002, -2.6500000000000004, -8.5, -10.0]
    """
    # TODO
    # r = copy.deepcopy(rewards)
    # res = [float(i) for i in r]
    # for i in reversed(range(len(rewards))):
    #     j = i - 1
    #     counter = 1
    #     while j >= 0:
    #         res[j] += rewards[i] * np.power(gamma, counter)
    #         counter += 1
    #         j -= 1
    # return res

    G_t=[]
    for i in range(len(rewards)):
        j=0
        curr_return=0
        for reward in rewards[i:]:
            curr_return+=reward*gamma**j
            j+=1
        G_t.append(curr_return)
    return G_t

def finish_episode(saved_rewards, saved_logprobs, gamma=1.0):
    """Samples an action from the policy at the state."""
    policy_loss = []
    returns = compute_returns(saved_rewards, gamma)
    returns = torch.Tensor(returns)
    # subtract mean and std for faster training
    returns = (returns - returns.mean()) / (returns.std() +
                                            np.finfo(np.float32).eps)
    for log_prob, reward in zip(saved_logprobs, returns):
        policy_loss.append(-log_prob * reward)
    policy_loss = torch.cat(policy_loss).sum()
    policy_loss.backward(retain_graph=True)
    # note: retain_graph=True allows for multiple calls to .backward()
    # in a single step

def get_reward(status):
    """Returns a numeric given an environment status."""
    return {
            Environment.STATUS_VALID_MOVE  : 0.0, # TODO
            Environment.STATUS_INVALID_MOVE: -3, #change to 3??
            Environment.STATUS_WIN         : 1.5,
            Environment.STATUS_TIE         : 0.5,
            Environment.STATUS_LOSE        : -1.5
    }[status]


def first_move_distr(policy, env):
    """Display the distribution of first moves."""
    state = env.reset()
    state = torch.from_numpy(state).long().unsqueeze(0)
    state = torch.zeros(3,9).scatter_(0,state,1).view(1,27)
    pr = policy(Variable(state))
    return pr.data


def load_weights(policy, episode):
    """Load saved weights"""
    weights = torch.load("ttt/policy-%d.pkl" % episode)
    policy.load_state_dict(weights)


def load_weights_improved(policy, episode, part):
    """Load saved weights"""
    part_file = ''
    if part == 1:
        part_file = 'p1/'
    else:
        part_file = 'p2/'

    weights = torch.load("policy/" + part_file + str(policy.hidden_size)+"_hidden/policy-%d.pkl" % episode)
    policy.load_state_dict(weights)


def train_improved(policy, env, gamma=1.0, log_interval=1000, num_iter=60000):
    """Train policy gradient."""
    optimizer = optim.Adam(policy.parameters(), lr=0.001)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10000, gamma=0.9)
    running_reward = 0

    #generate directories
    policy_dir = "policy/p1/"+str(policy.hidden_size)+"_hidden/"
    average_return_dir = "average_return/p1/"+str(policy.hidden_size)+"_hidden/"
    perf_dir =  "perf/p1/"+str(policy.hidden_size)+"_hidden/"
    if not os.path.isdir(average_return_dir):
        os.makedirs(average_return_dir)
    if not os.path.isdir(policy_dir):
        os.makedirs(policy_dir)
    if not os.path.isdir(perf_dir):
        os.makedirs(perf_dir)

    #required variable
    invalid_move_count = 0
    win_count = 0
    lose_count = 0 
    tie_count = 0 


    for i_episode in range(1, num_iter):
        saved_rewards = []
        saved_logprobs = []
        state = env.reset()
        done = False

        #Turn = 1, agent starts second
        turn = np.random.randint(2, size=1)[0]
        if turn:
            env.random_step()
            

        while not done:
            action, logprob = select_action(policy, state)
            state, status, done = env.play_against_random(action)
            reward = get_reward(status)
            saved_logprobs.append(logprob)
            saved_rewards.append(reward)

        #count number of invalid moves
        #same as checking for Environment.STATUS_INVALID_MOVE
        if (-3 in saved_rewards[:-1]):
            invalid_move_count += 1

        if status == Environment.STATUS_WIN:
            win_count += 1
        if status == Environment.STATUS_TIE:
            tie_count += 1
        if status == Environment.STATUS_LOSE:
            lose_count += 1

        R = compute_returns(saved_rewards)[0]
        running_reward += R

        finish_episode(saved_rewards, saved_logprobs, gamma)

        if i_episode % log_interval == 0:
            average_return = running_reward / log_interval

            print('Episode {}\tAverage return: {:.2f}'.format(
                i_episode, average_return))
            print('Total invalid move count since last log_interval: '+ str(invalid_move_count))
            
            total = float(win_count + lose_count + tie_count)
            print('Win: {}, Loss: {}, Tie: {}'.format(win_count/total, lose_count/total, tie_count/total))
            print('===='*6)

            # Store returns (serialize)
            with open(average_return_dir+"average_return-%d.pkl" % i_episode, 'wb') as handle:
                pickle.dump(average_return, handle,  protocol=pickle.HIGHEST_PROTOCOL)
            
            #Store state_dict of network
            torch.save(policy.state_dict(), policy_dir+"policy-%d.pkl" % i_episode)

            # Store win loss dictionary
            perf_dict = {'win': win_count/total,
                        'loss': lose_count/total, 
                        'tie': tie_count/total,
                        'inv': invalid_move_count,
                        'turn': turn}
            with open(perf_dir+"perf-%d.pkl" % i_episode, 'wb') as handle:
                pickle.dump(perf_dict, handle,  protocol=pickle.HIGHEST_PROTOCOL)

            #refresh reward and counts
            running_reward = 0
            win_count = 0
            lose_count = 0 
            tie_count = 0 
            invalid_move_count = 0
            
        if i_episode % 1 == 0: # batch_size
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()


def result_against_random(policy, env, turn):
    win_count = 0
    lose_count = 0 
    tie_count = 0 
    invalid_move_count = 0 

    for i_episode in range(100):
        saved_rewards = []
        saved_logprobs = []
        state = env.reset()
        done = False

        if turn:
            action, logprob = select_action(policy, state)
            env.step(action)

        while not done:
            action, logprob = select_action(policy, state)
            state, status, done = env.play_against_random(action)
            reward = get_reward(status)
            saved_logprobs.append(logprob)
            saved_rewards.append(reward)

        if (-3 in saved_rewards[:-1]):
            invalid_move_count += 1
        if status == Environment.STATUS_WIN:
            win_count += 1
        if status == Environment.STATUS_TIE:
            tie_count += 1
        if status == Environment.STATUS_LOSE:
            lose_count += 1

    #print results
    total = float(win_count + tie_count + lose_count)
    print('Win: {}, Loss: {}, Tie: {}'.format(win_count/total, lose_count/total, tie_count/total))

    perf_dict = {'win': win_count/total,
                'loss': lose_count/total, 
                'tie': tie_count/total,
                'inv': invalid_move_count,
                'turn': turn}

    return perf_dict


def train_against_self(policy, env, gamma=1.0, log_interval=1000, num_iter=6000):
    """Train policy gradient."""
    optimizer = optim.Adam(policy.parameters(), lr=0.001)
    scheduler = torch.optim.lr_scheduler.StepLR(
            optimizer, step_size=10000, gamma=0.9)
    running_reward = 0

    #generate directories
    policy_dir = "policy/p2/"+str(policy.hidden_size)+"_hidden/"
    average_return_dir = "average_return/p2/"+str(policy.hidden_size)+"_hidden/"
    perf_dir =  "perf/p2/"+str(policy.hidden_size)+"_hidden/"
    if not os.path.isdir(average_return_dir):
        os.makedirs(average_return_dir)
    if not os.path.isdir(policy_dir):
        os.makedirs(policy_dir)
    if not os.path.isdir(perf_dir):
        os.makedirs(perf_dir)


    for i_episode in range(num_iter):
        saved_rewards = []
        saved_logprobs = []
        state = env.reset()
        done = False

        #Turn = 1, agent starts second
        turn = np.random.randint(2, size=1)[0]
        if turn:
            action, logprob = select_action(policy, state)
            env.step(action)

        while not done:
            action, logprob = select_action(policy, state)
            state, status, done = env.play_against_self(policy, action)
            reward = get_reward(status)
            saved_logprobs.append(logprob)
            saved_rewards.append(reward)

        R = compute_returns(saved_rewards)[0]
        running_reward += R

        finish_episode(saved_rewards, saved_logprobs, gamma)

        if i_episode % log_interval == 0:
            print('Episode {}\tAverage return: {:.2f}'.format(
                i_episode,
                running_reward / log_interval))
            running_reward = 0

            #save average return 
            average_return = running_reward / log_interval
            with open(average_return_dir+"average_return-%d.pkl" % i_episode, 'wb') as handle:
                pickle.dump(average_return, handle,  protocol=pickle.HIGHEST_PROTOCOL)
            

            #save policy
            torch.save(policy.state_dict(), policy_dir+"policy-%d.pkl" % i_episode)


            #save performances
            perf_dict1 = result_against_random(policy, env, turn=1)
            perf_dict0 = result_against_random(policy, env, turn=0)

            with open(perf_dir+"perf1-%d.pkl" % i_episode, 'wb') as handle:
                pickle.dump(perf_dict1, handle,  protocol=pickle.HIGHEST_PROTOCOL)
            with open(perf_dir+"perf0-%d.pkl" % i_episode, 'wb') as handle:
                pickle.dump(perf_dict0, handle,  protocol=pickle.HIGHEST_PROTOCOL)


        if i_episode % 1 == 0: # batch_size
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()




if __name__ == '__main__':
    import sys
    policy = Policy()
    env = Environment()

    if len(sys.argv) == 1:
        # `python tictactoe.py` to train the agent

        train_against_self(policy, env)
    else:
        # `python tictactoe.py <ep>` to print the first move distribution
        # using weightt checkpoint at episode int(<ep>)
        ep = int(sys.argv[1])
        load_weights(policy, ep)
        print(first_move_distr(policy, env))





