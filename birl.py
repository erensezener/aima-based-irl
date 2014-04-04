"""
Author: Eren Sezener (erensezener@gmail.com)
Date: April 3, 2014

Description: This module finds the reward function via Bayesian Inverse Reinforcement Learning.
See "Bayesian Inverse Reinforcement Learning" by Deepak Ramachandran and Eyal Amir (2007) for algorithm details.

Status: Works correctly.

Dependencies: This module is compatible with Python 2.7.5.

Known bugs: -


"""


from mdp import *
from utils import *
from copy import deepcopy
from math import exp

class BIRL():

    def __init__(self, expert_mdp, iteration_limit = 30, r_min = -10, r_max = 10):
        self.expert_mdp = expert_mdp
        self.iteration_limit = iteration_limit
        self.n_rows, self.n_columns = expert_mdp.get_grid_size()
        self.r_min, self.r_max = r_min, r_max


    def run_multiple_birl(self):
        print "Expert rewards:"
        self.expert_mdp.print_rewards()
        expert_pi = best_policy(self.expert_mdp, value_iteration(self.expert_mdp, 0.1))
        print "Expert policy:"
        print_table(self.expert_mdp.to_arrows(expert_pi))
        print "---------------"

        diff_max = BIG_NUMBER
        best_pi = None
        best_mdp = None

        for i in range(self.iteration_limit):
            pi, mdp, diff = self.run_birl(expert_pi)
            print("Run :" + str(i))
            print_table(mdp.to_arrows(pi))
            print "vs"
            print_table(self.expert_mdp.to_arrows(expert_pi))
            print("Policy difference is " + str(get_difference(pi, expert_pi)))
            mdp.print_rewards()
            print "vs"
            self.expert_mdp.print_rewards()
            print ("Reward SSE: " + str(calculate_sse(mdp, self.expert_mdp)))
            print "---------------"

            if diff < diff_max:
                diff_max = diff
                best_pi = pi
                best_mdp = mdp

        print "---------------"
        print"Best results:"
        print_table(best_mdp.to_arrows(best_pi))
        print "vs"
        print_table(self.expert_mdp.to_arrows(expert_pi))
        print("Policy difference is " + str(get_difference(best_pi, expert_pi)))
        best_mdp.print_rewards()
        print "vs"
        self.expert_mdp.print_rewards()
        print ("Reward SSE: " + str(calculate_sse(best_mdp, self.expert_mdp)))
        print "---------------"



    def run_birl(self, expert_pi, iteration_limit = 1000, step_size = 2):
        mdp = self.create_rewards(self.create_gaussian_rewards)
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

    def create_rewards(self, reward_function_to_call = None):
        # If no reward function is specified, sets all rewards as 0
        if reward_function_to_call is None:
            return self.create_zero_rewards()
        return reward_function_to_call()

    def create_zero_rewards(self):
        return GridMDP([[0 for _ in range(self.n_columns)] for _ in range(self.n_rows)]
                       ,terminals=deepcopy(self.expert_mdp.terminals))

    def create_random_rewards(self):
        return GridMDP(
            [[random.uniform(self.r_min, self.r_max) for _ in range(self.n_columns)] for _ in range(self.n_rows)]
                       ,terminals=deepcopy(self.expert_mdp.terminals))

    def create_gaussian_rewards(self):
        mean, stdev = 0, self.r_max/3
        return GridMDP(
            [[self.bound_rewards(random.gauss(mean, stdev)) for _ in range(self.n_columns)] for _ in range(self.n_rows)]
                       ,terminals=deepcopy(self.expert_mdp.terminals))

    def bound_rewards(self, reward):
        if reward > self.r_max:
            reward = self.r_max
        elif reward < self.r_min:
            reward = self.r_min
        return reward


def get_difference(new_pi, ex_pi):
    shared_items = set(new_pi.items()) & set(ex_pi.items())
    return len(new_pi.items()) - len(shared_items)



