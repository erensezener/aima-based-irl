__author__ = 'Eren Sezener'
__maintainer__ = 'Eren Sezener'
__email__ = "erensezener@gmail.com"

"""
Author: Eren Sezener (erensezener@gmail.com)
Date: April 3, 2014

Status: Works correctly.

This module finds the reward function via Bayesian Inverse Reinforcement Learning.
See "Bayesian Inverse Reinforcement Learning" by Deepak Ramachandran and Eyal Amir (2007) for algorithm details.

This module is compatible with Python 2.7.5.

"""


from mdp import *
from utils import *
from copy import deepcopy
from math import exp


def run_birl():
    iteration_limit = 30

    term = (4, 3) #Coordinate of the terminal in (column_index, row_index) format. bottom-left is 0, 0
    expert_mdp = GridMDP([[-10, -5, 0, 0, 10],
                [-5, -3, 0, 0, 0],
                [0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0]],
               terminals=[term])

    print "Expert rewards:"
    expert_mdp.print_rewards()
    expert_pi = best_policy(expert_mdp, value_iteration(expert_mdp, 0.1))
    print "Expert policy:"
    print_table(expert_mdp.to_arrows(expert_pi))
    print "---------------"

    diff_max = BIG_NUMBER
    best_pi = None
    best_mdp = None

    for i in range(iteration_limit):
        pi, mdp, diff = iterate_birl(expert_mdp.get_grid_size(), term,  expert_pi)
        print("Run :" + str(i))
        print_table(mdp.to_arrows(pi))
        print "vs"
        print_table(expert_mdp.to_arrows(expert_pi))
        print("Policy difference is " + str(get_difference(pi, expert_pi)))
        mdp.print_rewards()
        print "vs"
        expert_mdp.print_rewards()
        print ("Reward SSE: " + str(calculate_sse(mdp, expert_mdp)))
        print "---------------"

        if diff < diff_max:
            diff_max = diff
            best_pi = pi
            best_mdp = mdp

    print "---------------"
    print"Best results:"
    print_table(best_mdp.to_arrows(best_pi))
    print "vs"
    print_table(expert_mdp.to_arrows(expert_pi))
    print("Policy difference is " + str(get_difference(best_pi, expert_pi)))
    best_mdp.print_rewards()
    print "vs"
    expert_mdp.print_rewards()
    print ("Reward SSE: " + str(calculate_sse(best_mdp, expert_mdp)))
    print "---------------"



def iterate_birl(grid_size, term_tuple, expert_pi, iteration_limit = 1000, step_size = 2):
    n_rows, n_columns = grid_size
    mdp = create_random_rewards(n_rows, n_columns, term_tuple)
    # mdp = create_similar_rewards()

    pi, U = policy_iteration(mdp)
    Q = get_q_values(mdp, U)
    posterior = calculate_posterior(mdp, Q, expert_pi)

    best_posterior = NEGATIVE_SMALL_NUMBER
    best_mdp = None
    best_pi = None



    for iter in range(iteration_limit):
        new_mdp = deepcopy(mdp) #creates a new reward function that is very similar to the original one
        new_mdp.modify_rewards_randomly(step_size)
        new_U = policy_evaluation(pi, U, new_mdp, 1)

        if pi != best_policy(new_mdp, new_U):
            new_pi, new_U = policy_iteration(new_mdp)
            new_Q = get_q_values(new_mdp, new_U)
            new_posterior = calculate_posterior(new_mdp, new_Q, expert_pi)

            if probability(min(1, exp(new_posterior - posterior))): # with min{1, P(R',pi') / P(R,pi)}
                pi, U = new_pi, new_U
                mdp = deepcopy(new_mdp)
                posterior = new_posterior
        else:
            new_Q = get_q_values(new_mdp, new_U)
            new_posterior = calculate_posterior(new_mdp, new_Q, expert_pi)

            if probability(min(1, exp(new_posterior - posterior))): # with min{1, P(R',pi) / P(R,pi)}
                mdp = deepcopy(new_mdp)
                posterior = new_posterior


        # sse = calculate_sse(mdp, expert_mdp);
        if posterior > best_posterior:
            best_posterior = posterior
            best_mdp = deepcopy(mdp)
            best_pi = pi

        # print("Difference is " + str(get_difference(pi, expert_pi)))
        # print str(calculate_sse(mdp, expert_mdp))

    return best_pi, best_mdp, get_difference(best_pi, expert_pi)

def get_difference(new_pi, ex_pi):
    shared_items = set(new_pi.items()) & set(ex_pi.items())
    return len(new_pi.items()) - len(shared_items)

# def create_similar_rewards():
#     return GridMDP([[-0.02, -2, -4, +7],
#                           [-0.06, -0.01,  -0.1, -0.04],
#                           [-2, -0.04, -0.04, -0.08],
#                           [1, -0.04, -2, -0.02]],
#                          terminals=[(3, 3)])


# def create_similar_rewards():
#     return GridMDP([[-0.02, -0.04, -0.04, +10],
#                           [-0.06, -0.01,  -0.1, -0.04],
#                           [-0.10, -0.04, -0.04, -0.08],
#                           [-0.04, -0.04, -0.2, -0.02]],
#                          terminals=[(3, 3)])

def create_random_rewards(m_tuple, n_tuple, term_tuple):
    return GridMDP([[0 for _ in range(n_tuple)] for _ in range(m_tuple)] #create 4-by-4 matrix with random doubles
                   ,terminals=[term_tuple])

# def create_random_rewards():
#     return GridMDP([[random.uniform(-9.9,+9.9) for _ in range(4)] for _ in range(4)] #create 4-by-4 matrix with random doubles
#                    ,terminals=[(3, 3)])

# def create_random_rewards():
#     return GridMDP([[random.gauss(0,1) for _ in range(4)] for _ in range(4)] #create 4-by-4 matrix with random doubles
#                    ,terminals=[(3, 3)])

def main():
    run_birl()

if __name__=="__main__":
    main()