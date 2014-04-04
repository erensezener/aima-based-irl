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
    def __init__(self, expert_mdp, iteration_limit=30, birl_iteration=1000, step_size=2, r_min=-10, r_max=10):
        self.expert_mdp = expert_mdp
        self.iteration_limit = iteration_limit
        self.n_rows, self.n_columns = expert_mdp.get_grid_size()
        self.r_min, self.r_max = r_min, r_max
        self.step_size = step_size
        self.expert_pi = best_policy(self.expert_mdp, value_iteration(self.expert_mdp, 0.01))
        self.birl_iteration = birl_iteration

    def run_multiple_birl(self):
        """Run BIRL algorithm iteration_limit times.
        Pick the result with the highest posterior probability
        """

        print "Expert rewards:"
        self.expert_mdp.print_rewards()
        print "Expert policy:"
        print_table(self.expert_mdp.to_arrows(self.expert_pi))
        print "---------------"

        max_policy_difference = BIG_NUMBER
        best_pi = None
        best_mdp = None

        for i in range(self.iteration_limit):
            pi, mdp, policy_difference = self.run_birl()
            print("Run :" + str(i))

            self.print_reward_comparison(mdp, pi)
            self.print_sse(mdp)

            if policy_difference < max_policy_difference:
                max_policy_difference = policy_difference
                best_pi = pi
                best_mdp = mdp

        print "---------------"
        print"Best results:"

        self.print_reward_comparison(best_mdp, best_pi)
        self.print_sse(best_mdp)

    def run_birl(self):
        #This is the core BIRL algorithm
        mdp = self.create_rewards(self.create_zero_rewards)
        pi, u = policy_iteration(mdp)
        q = get_q_values(mdp, u)
        posterior = calculate_posterior(mdp, q, self.expert_pi)
        best_posterior, best_mdp, best_pi = NEGATIVE_SMALL_NUMBER, None, None

        for _ in range(self.birl_iteration):
            new_mdp = deepcopy(mdp)
            new_mdp.modify_rewards_randomly(self.step_size)
            new_u = policy_evaluation(pi, u, new_mdp, 1)

            if pi != best_policy(new_mdp, new_u):
                new_pi, new_u = policy_iteration(new_mdp)
                new_q = get_q_values(new_mdp, new_u)
                new_posterior = calculate_posterior(new_mdp, new_q, self.expert_pi)

                if probability(min(1, exp(new_posterior - posterior))):
                    pi, u, mdp, posterior = new_pi, new_u, deepcopy(new_mdp), new_posterior

            else:
                new_q = get_q_values(new_mdp, new_u)
                new_posterior = calculate_posterior(new_mdp, new_q, self.expert_pi)

                if probability(min(1, exp(new_posterior - posterior))):
                    mdp, posterior = deepcopy(new_mdp), new_posterior

            if posterior > best_posterior:  # Pick the mdp with the best posterior
                best_posterior, best_mdp, best_pi = posterior, deepcopy(mdp), pi

        return best_pi, best_mdp, get_difference(best_pi, self.expert_pi)

    def print_reward_comparison(self, mdp, pi):
        print_table(mdp.to_arrows(pi))
        print "vs"
        print_table(self.expert_mdp.to_arrows(self.expert_pi))
        print("Policy difference is " + str(get_difference(pi, self.expert_pi)))
        mdp.print_rewards()
        print "vs"
        self.expert_mdp.print_rewards()

    def print_sse(self, mdp):
        print ("Reward SSE: " + str(calculate_sse(mdp, self.expert_mdp)))
        print "---------------"

#------------- Reward functions ------------
    def create_rewards(self, reward_function_to_call=None):
        # If no reward function is specified, sets all rewards as 0
        if reward_function_to_call is None:
            return self.create_zero_rewards()
        return reward_function_to_call()

    def create_zero_rewards(self):
        return GridMDP([[0 for _ in range(self.n_columns)] for _ in range(self.n_rows)]
                       , terminals=deepcopy(self.expert_mdp.terminals))

    def create_random_rewards(self):
        return GridMDP(
            [[random.uniform(self.r_min, self.r_max) for _ in range(self.n_columns)] for _ in range(self.n_rows)]
            , terminals=deepcopy(self.expert_mdp.terminals))

    def create_gaussian_rewards(self):
        mean, stdev = 0, self.r_max / 3
        return GridMDP(
            [[self.bound_rewards(random.gauss(mean, stdev)) for _ in range(self.n_columns)] for _ in range(self.n_rows)]
            , terminals=deepcopy(self.expert_mdp.terminals))

    def bound_rewards(self, reward):
        if reward > self.r_max:
            reward = self.r_max
        elif reward < self.r_min:
            reward = self.r_min
        return reward

#---------------------------------------
def get_difference(new_pi, ex_pi):
    shared_items = set(new_pi.items()) & set(ex_pi.items())
    return len(new_pi.items()) - len(shared_items)