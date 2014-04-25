"""
Author: Eren Sezener (erensezener@gmail.com)
Date: April 6, 2014

Description: This implementation is based on "Improving the Efficiency of Bayesian Inverse Reinforcement Learning" by
Michini and How.

Status: Work In Progress

Dependencies: This module is compatible with Python 2.7.5.

Known bugs: -


"""

from birl import *


def cooling_function(i):
    return 25 + float(i) / 50


def pick_random_state(mdp):
    m,n = mdp.get_grid_size()
    return random.randint(0, m-1), random.randint(0, n-1)


class ModifiedBIRL(BIRL):
    def __init__(self, expert_trace, grid_size, terminals, error_func, birl_iteration=1000, step_size=1.0):
        BIRL.__init__(self, expert_trace, grid_size, terminals, error_func, birl_iteration, step_size)


    @property
    def run_birl(self):
        policy_error, reward_error = [], []
        #This is the core BIRL algorithm
        mdp = self.create_rewards()
        pi, u = policy_iteration(mdp)
        q = get_q_values(mdp, u)
        posterior = calculate_posterior(mdp, q, self.expert_trace)

        for i in range(self.birl_iteration):
            state_index = pick_random_state(mdp)
            if not probability(self.state_relevance_function(state_index, mdp)):
                continue

            new_mdp = deepcopy(mdp)
            new_mdp.modify_state(state_index, self.step_size)
            new_u = policy_evaluation(pi, u, new_mdp, 1)

            if pi != best_policy(new_mdp, new_u):
                new_pi, new_u = policy_iteration(new_mdp)
                new_q = get_q_values(new_mdp, new_u)
                new_posterior = calculate_posterior(new_mdp, new_q, self.expert_trace)

                if probability(min(1, (exp(new_posterior - posterior)) ** (cooling_function(i)))):
                    pi, u, mdp, posterior = new_pi, new_u, deepcopy(new_mdp), new_posterior

            else:
                new_q = get_q_values(new_mdp, new_u)
                new_posterior = calculate_posterior(new_mdp, new_q, self.expert_trace)

                if probability(min(1, (exp(new_posterior - posterior)) ** (cooling_function(i)))):
                    mdp, posterior = deepcopy(new_mdp), new_posterior

            policy_error.append(runner.get_policy_difference(pi, self.expert_trace))
            reward_error.append(runner.normalize_by_max_reward(self.error_func(mdp), self))
        return pi, mdp, policy_error, reward_error

    def state_relevance_function(self, s, mdp):
        sum = 0
        for sp in self.expert_trace:
            sum += kernel(s, sp)
        return sum / self.get_normalizing_constant(s, mdp)

    def get_normalizing_constant(self, s, mdp):
        # return max([self.state_relevance_function(s) for s in self.expert_trace])
        return sum ([kernel(s, sp) for sp in mdp.reward.keys()])


def kernel(s, sp, sigma=0.01):
    distance = (euclidean_distance(s, sp)) ** 2
    return math.exp((-distance ** 2) / 2 * (sigma ** 2))


def euclidean_distance(s, sp):
    x0, y0 = s
    x1, y1 = sp
    return math.sqrt(abs(x0 - x1) ** 2 + abs(y0 - y1) ** 2)