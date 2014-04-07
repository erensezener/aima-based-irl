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


class ModifiedBIRL(BIRL):
    def __init__(self, expert_trace, grid_size, terminals, error_func):
        BIRL.__init__(self, expert_trace, grid_size, terminals, error_func)


    @property
    def run_birl(self):
        errors_per_iteration = []
        #This is the core BIRL algorithm
        mdp = self.create_rewards(self.create_rewards)
        pi, u = policy_iteration(mdp)
        q = get_q_values(mdp, u)
        posterior = calculate_posterior(mdp, q, self.expert_trace)
        best_posterior, best_mdp, best_pi = NEGATIVE_SMALL_NUMBER, None, None

        for _ in range(self.birl_iteration):
            new_mdp = deepcopy(mdp)
            new_mdp.modify_rewards_randomly(self.step_size)
            new_u = policy_evaluation(pi, u, new_mdp, 1)

            if pi != best_policy(new_mdp, new_u):
                new_pi, new_u = policy_iteration(new_mdp)
                new_q = get_q_values(new_mdp, new_u)
                new_posterior = calculate_posterior(new_mdp, new_q, self.expert_trace)

                if probability(min(1, exp(new_posterior - posterior))):
                    pi, u, mdp, posterior = new_pi, new_u, deepcopy(new_mdp), new_posterior

            else:
                new_q = get_q_values(new_mdp, new_u)
                new_posterior = calculate_posterior(new_mdp, new_q, self.expert_trace)

                if probability(min(1, exp(new_posterior - posterior))):
                    mdp, posterior = deepcopy(new_mdp), new_posterior

            if posterior > best_posterior:  # Pick the mdp with the best posterior
                best_posterior, best_mdp, best_pi = posterior, deepcopy(mdp), pi

            errors_per_iteration.append(self.error_func(mdp))
        return best_pi, best_mdp, errors_per_iteration

    def state_relevance_function(self, s):
        sum = 0
        for sp in self.expert_trace:
            sum += kernel(s, sp)
        return sum / self.get_normalizing_constant()

    def get_normalizing_constant(self):
        return max([self.state_relevance_function(s) for s in self.expert_trace])

def kernel(s, sp, sigma = 0.01):
    distance = (euclidean_distance(s, sp))**2
    return math.exp((-distance**2)/2*(sigma**2))


def euclidean_distance(s, sp):
    x0, y0 = s
    x1, y1 = sp
    return math.sqrt( abs(x0 - x1)**2 + abs(y0 - y1)**2 )