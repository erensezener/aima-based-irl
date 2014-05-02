"""
Author: Eren Sezener (erensezener@gmail.com)
Date: April 4, 2014

Description: Runs the BIRL algorithm multiple times.

Status: Works correctly.

Dependencies:

Known bugs: -

"""

# from birl import *
from functools import partial
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.gridspec as gridspec
from modified_birl import *


def main():
    number_of_iterations = 10

    # expert_mdp = GridMDP([[-10, -5, 0, 0, 10],
    #         [-5, -3, 0, 0, 0],
    #         [0, 0, 0, 0, 0],
    #         [0, 0, 0, 0, 0]],
    #         terminals=[(4,3)])

    # expert_mdp = GridMDP([[-10, -5, -3, -1, 0, 0, 0, 0, 0, 10],
    #         [-8, -5, -3, 0, 0, 0, 0, 0, 0, 0],
    #         [-5, -2, -1, 0, 0, 0, 0, 0, 0, 0],
    #         [-3, -1, 0, 0, 0, 0, 0, 0, 0, 0],
    #         [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]],
    #         terminals=[(9,4)])
    #
    # expert_mdp = GridMDP([[0, 0, 0, 0, -1, -1, 0, 0, 0, 10],
    #                     [0, 0, 0, -3, -3, -3, -3, 0, 0, 0],
    #                     [0, 0, 0, -3, -5, -5, -3, 0, 0, 0],
    #                     [0, 0, 0, -3, -3, -3, -3, 0, 0, 0],
    #                     [0, 0, 0, 0, 0, -1, -1, 0, 0, 0]],
    #                     terminals=[(9,4)])
    #
    # rewards = [[0, 0, 0, 0, -1, -1, 0, 0, 0, 10],
    #            [0, 0, 0, -3, -3, -3, -3, 0, 0, 0],
    #            [0, 0, 0, -3, -5, -5, -3, 0, 0, 0],
    #            [0, 0, 0, -3, -3, -3, -3, 0, 0, 0],
    #            [0, 0, 0, 0, 0, -1, -1, 0, 0, 0]]
    #

    rewards = [[0, 0, 0, 0, -8, -8, 0, 0, 0, 10],
               [0, 0, 0, -8, -10, -10, -8, 0, 0, 0],
               [0, 0, 0, -8, -10, -10, -8, 0, 0, 0],
               [0, 0, 0, -8, -10, -10, -8, 0, 0, 0],
               [0, 0, 0, 0, 0, -8, -8, 0, 0, 0]]

    # rewards = [[-6, -3, -1, 0, 0, 0, 0, 0, 0, 10],
    #             [-3, -3, -1, 0, 0, 0, 0, 0, 0, 0],
    #             [-1, -1, -1, 0, 0, 0, 0, -1, -1, -1],
    #             [0, 0, 0, 0, 0, 0, 0, -1, -3, -3],
    #             [0, 0, 0, 0, 0, 0, 0, -1, -3, -6]]
    #
    # rewards = [[0, 0, 0, 0, 0, 0, 0, 0, 0, -3, -3, -3, -3, 0, 0, 0, 0, 0, 10],
    #         [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, -3, -3, -3, -3, 0, 0, 0, 0, 0],
    #         [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, -3, -3, -3, -3, 0, 0, 0, 0, 0],
    #         [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, -3, -3, -3, -3, 0, 0, 0, 0, 0],
    #         [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, -3, -3, -3, -3, 0, 0, 0, 0, 0],
    #         [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, -3, -3, -3, -3, 0, 0, 0, 0, 0],
    #         [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    #         [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, -3, -3, -3, 0, 0, 0, 0, 0, 0],
    #         [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, -3, -3, -3, 0, 0, 0, 0, 0, 0],
    #         [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, -3, -3, -3, 0, 0, 0, 0, 0, 0]]



    expert_mdp = GridMDP(rewards,
                         terminals=[(9, 4)])

    expert_trace = best_policy(expert_mdp, value_iteration(expert_mdp, 0.001))
    print "Expert rewards:"
    expert_mdp.print_rewards()
    print "Expert policy:"
    print_table(expert_mdp.to_arrows(expert_trace))
    print "---------------"

    expert_trace.pop((0,1))
    expert_trace.pop((0,2))
    expert_trace.pop((0,3))

    birl = ModifiedBIRL(expert_trace, expert_mdp.get_grid_size(), expert_mdp.terminals,
                partial(calculate_error_sum, expert_mdp), birl_iteration=2, step_size=1.0)
    run_multiple_birl(birl, expert_mdp, expert_trace, number_of_iterations)


def plot_errors(policy_error, reward_error, directory_name, birl, i, expert_mdp, mdp):
    gs = gridspec.GridSpec(3, 2)
    ax0 = plt.subplot(gs[0, :-1])
    ax1 = plt.subplot(gs[0, -1])
    ax2 = plt.subplot(gs[1, :])
    ax3 = plt.subplot(gs[2, :])

    expert_data = np.array(expert_mdp.get_grid())
    ax0.pcolor(expert_data, cmap=plt.cm.RdYlGn)
    ax0.set_title("Expert's Rewards")
    ax0.invert_yaxis()

    data = np.array(mdp.get_grid())
    ax1.pcolor(data, cmap=plt.cm.RdYlGn)
    ax1.set_title("Reward Estimations")
    ax1.invert_yaxis()

    ax2.plot(range(birl.birl_iteration), policy_error, 'ro')
    ax2.set_title('Policy change')
    ax3.plot(range(birl.birl_iteration), reward_error, 'bo')
    ax3.set_title('Reward change')

    plt.tight_layout()
    plt.savefig(directory_name + "/run" + str(i) + ".png")


def run_multiple_birl(birl, expert_mdp, expert_trace, number_of_iteration):
    """Run BIRL algorithm number_of_iteration times.
    """
    directory_name = initialize_output_directory(birl)

    for i in range(number_of_iteration):
        pi, mdp, policy_error, reward_error = birl.run_birl()
        plot_errors(policy_error, reward_error, directory_name, birl, i, expert_mdp, mdp)
        print("Run :" + str(i))
        print_reward_comparison(mdp, pi, expert_mdp, expert_trace)
        print_error_sum(mdp, birl, expert_mdp)


def initialize_output_directory(birl):
    directory_name = 'outputs/iter' + str(birl.birl_iteration) + \
                     '_stepsize' + str(birl.step_size) + '_no' + str(randint(0, 2 ** 30))
    if not os.path.exists(directory_name):
        os.makedirs(directory_name)
    return directory_name


def print_reward_comparison(mdp, pi, expert_mdp, expert_trace):
    print_table(mdp.to_arrows(pi))
    print "vs"
    print_table(mdp.to_arrows(expert_trace))
    print("Policy difference is " + str(get_policy_difference(pi, expert_trace)))
    mdp.print_rewards()
    print "vs"
    expert_mdp.print_rewards()


def print_error_sum(mdp, birl, expert_mdp):
    print ("Total Error: " + str(normalize_by_max_reward(calculate_error_sum(mdp, expert_mdp), birl)))
    print "---------------"


def print_sse(mdp, expert_trace):
    print ("Reward SSE: " + str(calculate_sse(mdp, expert_trace)))
    print "---------------"


def normalize_by_max_reward(value, birl):
    if birl.r_max != abs(birl.r_min):
        raise Exception("Normalization cannot be done. r_min and r_max values have different abs sums!")
    return value / float(birl.r_max)


def calculate_sse(mdp1, mdp2):
    "Returns the sum of the squared errors between two reward functions"
    sse = 0
    if not (mdp1.cols == mdp2.cols and mdp1.rows == mdp2.rows):
        raise Exception("Mismatch between # of rows and columns of reward vectors")

    for x in range(mdp1.cols):
        for y in range(mdp1.rows):
            sse += (mdp1.reward[x, y] - mdp2.reward[x, y]) ** 2
    return sse


def calculate_error_sum(mdp1, mdp2):
    """Returns the sum of errors between two reward functions
    Sum is normalized with respect to the number of states
    """
    sum = 0
    if not (mdp1.cols == mdp2.cols and mdp1.rows == mdp2.rows):
        raise Exception("Mismatch between # of rows and columns of reward vectors")

    for x in range(mdp1.cols):
        for y in range(mdp1.rows):
            sum += abs(mdp1.reward[x, y] - mdp2.reward[x, y])
    return sum / (float(mdp1.cols * mdp1.rows))


def get_policy_difference(new_pi, ex_pi):
    shared_items = set(new_pi.items()) & set(ex_pi.items())
    return len(new_pi.items()) - len(shared_items)


if __name__ == "__main__":
    main()