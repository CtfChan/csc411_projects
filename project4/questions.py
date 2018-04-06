from tictactoe import *

import glob
import matplotlib.pyplot as plt


def part1():
    #Part 1
    env = Environment()
    print('Initiating lonely game of tic-tac-toe ')
    env.render()
    while True:
        move = raw_input('Player 1: Make a move: ')
        env.step(int(move))
        env.render()
        if env.check_win():
            break

        move = raw_input('Player 2: Make a move: ')
        env.step(int(move))
        env.render()
        if env.check_win():
            break

def part2():
	pass

def part3a():
	print(compute_returns([0,0,0,1], 1.0))
	print([1.0, 1.0, 1.0, 1.0])

	print(compute_returns([0,0,0,1], 0.9))
	print([0.7290000000000001, 0.81, 0.9, 1.0])

	print(compute_returns([0,-0.5, 5,0.5,-10], 0.9))
	print([-2.5965000000000003, -2.8850000000000002, -2.6500000000000004, -8.5, -10.0])

def plot_average_returns(policies):
    #loop through
    for num in policies:
        files = sorted(glob.glob('average_return/' + str(num) +'_hidden/*'))  
        episode_list = []
        average_return_list = []

        for f in files: 
            with open(f, 'rb') as handle:
                average_return = pickle.load(handle)
            average_return_list.append(average_return)
            episode = f.replace('-',' ').replace('.',' ').split()[1]
            episode_list.append(int(episode))
        
        sizes = [2 for n in range(len(episode_list))]
        plt.scatter(episode_list, average_return_list, label=str(num)+" hidden units",s=sizes)

    plt.title("Episodes vs. Average Return")
    plt.xlabel("Episodes")
    plt.ylabel("Average Return")
    plt.legend(loc="best")
    plt.savefig("figures/average_return_vs_episodes" + str(policies)+ ".png")
    plt.close('all')
    #plt.show()


def plot_inv_move(policies):
    #loop through
    for num in policies:
        files = sorted(glob.glob('perf/' + str(num) +'_hidden/*'))  
        inv_move_list = []
        episode_list = []

        for f in files: 
            with open(f, 'rb') as handle:
                perf_dict = pickle.load(handle)
    
            episode = f.replace('-',' ').replace('.',' ').split()[1]
            
            episode_list.append(int(episode))
            inv_move_list.append(perf_dict['inv'])

        
        sizes = [2 for n in range(len(episode_list))]
        plt.scatter(episode_list, inv_move_list, label="invalid moves " + str(num),s=sizes)

    plt.title("Invalid Moves vs. Episodes")
    plt.xlabel("Episodes")
    plt.ylabel("Invalid Moves")
    
    art = []
    lgd = plt.legend(loc=9, bbox_to_anchor=(0.5, -0.1), ncol=4)
    art.append(lgd)

    plt.savefig("figures/inv_move_vs_episode_" + str(policies)+ ".png",additional_artists=art,
    bbox_inches="tight")

    plt.close('all')
    #plt.show()


def plot_perf(policies):
    #loop through
    for num in policies:
        files = sorted(glob.glob('perf/' + str(num) +'_hidden/*'))  
        wins= []
        losses = []
        ties = []
        episode_list = []

        for f in files: 
            with open(f, 'rb') as handle:
                perf_dict = pickle.load(handle)
    
            episode = f.replace('-',' ').replace('.',' ').split()[1]
            
            episode_list.append(int(episode))
            wins.append(perf_dict['win']) 
            losses.append(perf_dict['loss']) 
            ties.append(perf_dict['tie'])

        
        sizes = [2 for n in range(len(episode_list))]
        plt.scatter(episode_list, wins, label="wins " + str(num),s=sizes)
        plt.scatter(episode_list, losses, label="losses " + str(num),s=sizes)
        plt.scatter(episode_list, ties, label="ties " + str(num),s=sizes)

    plt.title("Win/Loss/Tie Rate vs. Episodes")
    plt.xlabel("Episodes")
    plt.ylabel("Performance")
    
    art = []
    lgd = plt.legend(loc=9, bbox_to_anchor=(0.5, -0.1), ncol=len(policies))
    art.append(lgd)

    plt.savefig("figures/perf_vs_episode_" + str(policies)+ ".png",additional_artists=art,
    bbox_inches="tight")

    plt.close('all')
    #plt.show()


#Do 5b before doing a
def part5b():
    policies = [32, 64, 128, 256, 512]
    for num in policies:
        policy = Policy(input_size=27, hidden_size=num, output_size=9)
        env = Environment()
        train_improved(policy, env, num_iter=90000)
    plot_average_returns(policies)

    #debugging
    plot_perf(policies)

def part5a():
    policy = Policy(input_size=27, hidden_size=240, output_size=9)
    env = Environment()
    train_improved(policy, env,num_iter=130000)
    plot_average_returns([256])
    plot_perf([256])

def part5c():
    '''
    Figure out when it learns invalid moves
    '''
    policies = [256]
    plot_inv_move(policies)

def part5d(): 
    '''Load weigths and play 100 games
    '''
    #load weights
    policy = Policy(input_size=27, hidden_size=256, output_size=9)
    env = Environment()
    load_weights_improved(policy, 50000)

    #count wins, losses and ties
    win_count = 0
    lose_count = 0 
    tie_count = 0 

    #play 100 games
    for i_episode in range(100):
        saved_rewards = []
        saved_logprobs = []
        state = env.reset()
        done = False
        while not done:
            action, logprob = select_action(policy, state)
            state, status, done = env.play_against_random(action)
            reward = get_reward(status)
            saved_logprobs.append(logprob)
            saved_rewards.append(reward)

            #Playback for first 5 moves
            if i_episode < 5:
                env.render()
                if done == True:
                    print('===' * 6)

        if status == Environment.STATUS_WIN:
            win_count += 1
        if status == Environment.STATUS_TIE:
            tie_count += 1
        if status == Environment.STATUS_LOSE:
            lose_count += 1

    #print results
    total = float(win_count + tie_count + lose_count)
    print('Win: {}, Loss: {}, Tie: {}'.format(win_count/total, lose_count/total, tie_count/total))


def part6():
    policies = [256]
    plot_perf(policies)

def part7():
    policy = Policy(input_size=27, hidden_size=256, output_size=9)
    env = Environment()

    distribution = first_move_distr(policy, env).numpy()
    to_print = '0 & '
    for num in distribution[0]:
        to_print += str(num) + ' & ' 
    to_print = to_print[:-2] + ' \\' + '\\'
    print(to_print)
    print('\hline')
 

    episode_list = [1000, 10000, 20000, 30000, 50000, 60000]
    for ep in episode_list:
        load_weights_improved(policy, ep)
        distribution = first_move_distr(policy, env).numpy()

        to_print = str(ep) + ' & '
        for num in distribution[0]:
            to_print += str(num) + ' & '
        to_print = to_print[:-2] + ' \\' + '\\'
        print(to_print)
        print('\hline')

def part8():
    '''Load weigths and play 100 games
    '''
    #load weights
    policy = Policy(input_size=27, hidden_size=256, output_size=9)
    env = Environment()
    load_weights_improved(policy, 50000)

    #count wins, losses and ties
    win_count = 0
    lose_count = 0 
    tie_count = 0 

    temp_env = Environment()
    
    #play 100 games
    for i_episode in range(200):
        saved_rewards = []
        saved_logprobs = []
        state = env.reset()
        done = False
        while not done:
            #distr = first_move_distr(policy, env)
            action, logprob = select_action(policy, state)
            temp_env = copy.deepcopy(env)


            state, status, done = env.play_against_random(action)
            reward = get_reward(status)
            saved_logprobs.append(logprob)
            saved_rewards.append(reward)

        #Playback for losing games
        if status == Environment.STATUS_LOSE:
            temp_env.render()
            #print(first_move_distr(policy, temp_env).numpy().reshape(3,3))
            env.render()
            print('====' * 6)

        if status == Environment.STATUS_WIN:
            win_count += 1
        if status == Environment.STATUS_TIE:
            tie_count += 1
        if status == Environment.STATUS_LOSE:
            lose_count += 1

    #print results
    total = float(win_count + tie_count + lose_count)
    print('Win: {}, Loss: {}, Tie: {}'.format(win_count/total, lose_count/total, tie_count/total))



if __name__ == '__main__':
    #part1()
    #part3a()
    
    #part5b()
    #part5a()
    #part5c()
    #part5d()

    #part6()

    #part7()

    part8()

