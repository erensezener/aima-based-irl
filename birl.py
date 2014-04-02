__author__ = 'Eren Sezener'
__maintainer__ = 'Eren Sezener'
__email__ = "erensezener@gmail.com"

"""
This module finds the reward function via Bayesian Inverse Reinforcement Learning.
See "Bayesian Inverse Reinforcement Learning" by Deepak Ramachandran and Eyal Amir (2007) for algorithm details.

This module is compatible with Python 2.7.5.

"""


from mdp import *
from utils import *
from copy import deepcopy
import random
from math import exp


def run_birl():
    iteration_limit = 1

    # expert_mdp = GridMDP([[1, 0, 0, 2],
    #                       [-0.4, -0.4, -0.4, -0.4],
    #                       [-0.4, -0.4, -0.4, -0.4],
    #                       [-0.4, -0.4, -0.4, -0.4]],
    #                      terminals=[(3, 3)])

    expert_mdp = GridMDP([[0, 0, 0, 10],
                [0, 0, 0, 0],
                [0, 0, 0, 0],
                [0, 0, 0, 0]],
               terminals=[(3, 3)])

    expert_mdp.print_rewards()
    expert_pi = best_policy(expert_mdp, value_iteration(expert_mdp, 0.1))
    print "Expert pi:"
    print_table(expert_mdp.to_arrows(expert_pi))

    errors = []

    for i in range(iteration_limit):
        pi, mdp, diff = iterate_birl(expert_pi, expert_mdp)
        errors.append(calculate_sse(mdp, expert_mdp))



    # print_table(best_mdp.to_arrows(best_pi))
    # print "vs"
    # print_table(expert_mdp.to_arrows(expert_pi))
    # print("Difference is " + str(get_difference(best_pi, expert_pi)))
    mdp.print_rewards()
    print "vs"
    expert_mdp.print_rewards()
    print(str(max(errors)))




def iterate_birl(expert_pi, expert_mdp, iteration_limit = 1000, step_size = 0.5):
    mdp = create_random_rewards()
    U = value_iteration(mdp)
    pi = best_policy(mdp, U)
    Q = get_q_values(mdp, U)

    min_error = 2**10
    min_mdp = None

    for iter in range(iteration_limit):
        new_mdp = deepcopy(mdp) #creates a new reward function that is very similar to the original one
        new_mdp.modify_rewards_randomly(step_size)
        # new_U = value_iteration(new_mdp)
        # new_pi = best_policy(new_mdp, new_U)
        # new_U = policy_evaluation(pi, U, new_mdp)
        # new_Q = get_q_values(new_mdp, new_U)
        # new_pi = best_policy(new_mdp, new_U)

        if True:
        # if new_pi != best_policy(new_mdp, new_U):
            new_U = value_iteration(new_mdp)
            # new_pi = policy_iteration(new_mdp)
            # new_U = policy_evaluation(new_pi, new_U, new_mdp)
            new_pi = best_policy(new_mdp, new_U)
            new_Q = get_q_values(new_mdp, new_U)
            posterior = calculate_posterior(mdp, Q, expert_pi, uniform_prior)
            new_posterior = calculate_posterior(new_mdp, new_Q, expert_pi, uniform_prior)


            if probability(min(1, exp(new_posterior - posterior))): # with min{1, P(R',pi') / P(R,pi)}
                pi, U, Q = new_pi, new_U, new_Q
                mdp = deepcopy(new_mdp)
        else:
            posterior = calculate_posterior(mdp, Q, expert_pi, uniform_prior)
            new_posterior = calculate_posterior(new_mdp, new_Q, expert_pi, uniform_prior)

            if probability(min(1, exp(new_posterior - posterior))): # with min{1, P(R',pi) / P(R,pi)}
                mdp, U, Q = new_mdp, new_U, new_Q

        sse = calculate_sse(mdp, expert_mdp);
        if sse < min_error:
            min_error = sse
            min_mdp = deepcopy(mdp)

        print("Difference is " + str(get_difference(pi, expert_pi)))
        print str(calculate_sse(mdp, expert_mdp))

    return pi, min_mdp, get_difference(pi, expert_pi)

def get_difference(new_pi, ex_pi):
    shared_items = set(new_pi.items()) & set(ex_pi.items())
    return len(new_pi.items()) - len(shared_items)


def create_similar_rewards():
    return GridMDP([[-0.02, -0.04, -0.04, +1],
                          [-0.06, -0.01,  -0.1, -0.04],
                          [-0.10, -0.04, -0.04, -0.08],
                          [-0.04, -0.04, -0.2, -0.02]],
                         terminals=[(3, 3)])

def create_random_rewards():
    return GridMDP([[random.uniform(-1,1) for _ in range(4)] for _ in range(4)] #create 4-by-4 matrix with random doubles
                   ,terminals=[(3, 3)])


def main():
    run_birl()

if __name__=="__main__":
    main()