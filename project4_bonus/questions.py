from tictactoe_bonus import *

import glob
import matplotlib.pyplot as plt


def plot_average_returns(policies, part):
    part_file = ''
    if part == 1:
        part_file = 'p1/'
    else:
        part_file = 'p2/'


    #loop through
    for num in policies:
        files = sorted(glob.glob('average_return/' + part_file + str(num) +'_hidden/*'))  
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
    plt.savefig("figures/" + part_file +"average_return_vs_episodes" + str(policies)+ ".png")
    plt.close('all')
    #plt.show()




def plot_perf(policies, part):
    part_file = ''
    if part == 1:
        part_file = 'p1/'
    else:
        part_file = 'p2/'

    #loop through
    for num in policies:
        files = sorted(glob.glob('perf/' + part_file +  str(num) +'_hidden/*'))  
        #Lists to store result when agent plays first or second
        wins_first = []
        losses_first = []
        ties_first = []

        wins_second = []
        losses_second = []
        ties_second = []

        episode_list_first = []
        episode_list_second = []

        for f in files: 
            with open(f, 'rb') as handle:
                perf_dict = pickle.load(handle)
    
            episode = f.replace('-',' ').replace('.',' ').split()[1]
            
            #if perf_dict['turn'] == 1 then agent started second
            if perf_dict['turn']:
                wins_second.append(perf_dict['win']) 
                losses_second.append(perf_dict['loss']) 
                ties_second.append(perf_dict['tie'])
                episode_list_second.append(int(episode))
            else:
                wins_first.append(perf_dict['win']) 
                losses_first.append(perf_dict['loss']) 
                ties_first.append(perf_dict['tie'])
                episode_list_first.append(int(episode))


        sizes = [2 for n in range(len(wins_first))]
        plt.scatter(episode_list_first, wins_first, label="wins (agent first)",s=sizes)
        plt.scatter(episode_list_first, losses_first, label="losses (agent first)",s=sizes)
        plt.scatter(episode_list_first, ties_first, label="ties (agent first)" ,s=sizes)

        sizes = [2 for n in range(len(wins_second))]
        plt.scatter(episode_list_second, wins_second, label="wins (agent second)",s=sizes)
        plt.scatter(episode_list_second, losses_second, label="losses (agent second)",s=sizes)
        plt.scatter(episode_list_second, ties_second, label="ties (agent second)" ,s=sizes)

    plt.title("Win/Loss/Tie Rate vs. Episodes")
    plt.xlabel("Episodes")
    plt.ylabel("Performance")
    
    art = []
    lgd = plt.legend(loc=9, bbox_to_anchor=(0.5, -0.1), ncol=len(policies)*2)
    art.append(lgd)

    plt.savefig("figures/" + part_file + "perf_vs_episode_" + str(policies)+ ".png",additional_artists=art,
    bbox_inches="tight")

    plt.close('all')
    #plt.show()



def plot_perf_part2(policies, part, turn):
    part_file = ''
    if part == 1:
        part_file = 'p1/'
    else:
        part_file = 'p2/'

    #loop through
    for num in policies:
        files = sorted(glob.glob('perf/' + part_file +  str(num) 
                +'_hidden/perf' + str(turn) + '*'))

        #Lists to store result when agent plays first or second
        wins_first = []
        losses_first = []
        ties_first = []

        wins_second = []
        losses_second = []
        ties_second = []

        episode_list_first = []
        episode_list_second = []

        for f in files: 
            with open(f, 'rb') as handle:
                perf_dict = pickle.load(handle)
    
            episode = f.replace('-',' ').replace('.',' ').split()[1]
            
            #if perf_dict['turn'] == 1 then agent started second
            if perf_dict['turn']:
                wins_second.append(perf_dict['win']) 
                losses_second.append(perf_dict['loss']) 
                ties_second.append(perf_dict['tie'])
                episode_list_second.append(int(episode))
            else:
                wins_first.append(perf_dict['win']) 
                losses_first.append(perf_dict['loss']) 
                ties_first.append(perf_dict['tie'])
                episode_list_first.append(int(episode))

        if not turn:
            sizes = [2 for n in range(len(wins_first))]
            plt.scatter(episode_list_first, wins_first, label="wins (agent first)",s=sizes)
            plt.scatter(episode_list_first, losses_first, label="losses (agent first)",s=sizes)
            plt.scatter(episode_list_first, ties_first, label="ties (agent first)" ,s=sizes)
        else:
            sizes = [2 for n in range(len(wins_second))]
            plt.scatter(episode_list_second, wins_second, label="wins (agent second)",s=sizes)
            plt.scatter(episode_list_second, losses_second, label="losses (agent second)",s=sizes)
            plt.scatter(episode_list_second, ties_second, label="ties (agent second)" ,s=sizes)

    extra = "(Agent Second)" if turn else "(Agent First)"
    plt.title("Win/Loss/Tie Rate vs. Episodes" + extra)
    plt.xlabel("Episodes")
    plt.ylabel("Performance")
    
    art = []
    lgd = plt.legend(loc=9, bbox_to_anchor=(0.5, -0.1), ncol=len(policies)*2)
    art.append(lgd)

    plt.savefig("figures/" + part_file + "perf_vs_episode_" + 
                    str(policies)+ extra + ".png",additional_artists=art,
    bbox_inches="tight")

    plt.close('all')
    #plt.show()


def play_against_random_print(policy, env, games_to_play,to_print=True):
    win_count = 0
    lose_count = 0 
    tie_count = 0 
    invalid_move_count = 0

    for i_episode in range(games_to_play):
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

        #Playback all finished games
        
        if to_print:
            print("Started Second" if turn == 1 else "Started First")
            env.render()

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


def play_against_self_print(policy, env, games_to_play):
    win_count = 0
    lose_count = 0 
    tie_count = 0 
    invalid_move_count = 0

    for i_episode in range(games_to_play):
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

        #Playback all finished games
        print("Started Second" if turn == 1 else "Started First")
        env.render()

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


def bonus1():
    policy = Policy(input_size=27, hidden_size=256, output_size=9)
    env = Environment()

    train_improved(policy, env,num_iter=60000)

    plot_average_returns([256], 1)
    plot_perf([256], 1)

def bonus1_play():
    policy = Policy(input_size=27, hidden_size=256, output_size=9)
    env = Environment()

    load_weights_improved(policy, 50000, 1)
    play_against_random_print(policy, env, 100, False)


def bonus2():
    policy = Policy(input_size=27, hidden_size=256, output_size=9)
    env = Environment()

    #load from part 1
    load_weights_improved(policy, 4000, 1)
    train_against_self(policy, env, num_iter=60000)

    plot_perf_part2([256], 2, 0)
    plot_perf_part2([256], 2, 1)

    plot_average_returns([256], 2)


def bonus2_p2():
    policy = Policy(input_size=27, hidden_size=256, output_size=9)
    env = Environment()
    load_weights_improved(policy, 25000, 2)

    print("5 Games Against Random")
    play_against_random_print(policy, env, 5)
    print('====' * 6)
    print("5 Games Against Self")
    play_against_self_print(policy, env, 5)

if __name__ == '__main__':
    #bonus1()
    bonus1_play()

    #bonus2()
    #bonus2_p2()

