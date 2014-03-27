__author__ = 'erensezener'

from mdp import *
from utils import *
from copy import copy

def run_irl():
    expert_mdp = GridMDP([[-0.04, -0.04, -0.04, +1],
                          [-0.04, -0.04,  -0.04, -0.04],
                          [-0.04, -0.04, -0.04, -0.04],
                          [-0.04, -0.04, -0.04, -0.04]],
                         terminals=[(3, 3)])
    expert_pi = policy_iteration(expert_mdp)

    pi, mdp = iterate_irl(expert_mdp, expert_pi)

    print_table(mdp.to_arrows(pi))

    print "vs"

    print_table(expert_mdp.to_arrows(expert_pi))

def iterate_irl(expert_mdp, expert_pi):
    iteration_limit = 1000
    epsilon = 0.1
    mdp = create_random_rewards()
    pi = policy_iteration(mdp)

    for iter in range(iteration_limit):
        new_mdp = copy(mdp) #creates a new reward function that is very similar to the original one
        new_mdp.modify_rewards_randomly(epsilon)
        new_pi = policy_iteration(mdp)
        if get_difference(new_pi, expert_pi) == 0:
            return new_pi, new_mdp
        elif get_difference(new_pi, expert_pi) < get_difference(pi, expert_pi): #if policy is more similar to the exper policy
            mdp = new_mdp
            pi = new_pi
            print ("Improvement in " + str(iter))


    return pi, mdp


def get_difference(new_pi, ex_pi):
    shared_items = set(new_pi.items()) & set(ex_pi.items())
    return len(shared_items)


def create_random_rewards():
    return GridMDP([
                       [random.uniform(-1,1) for i in range(4)] for j in range(4)] #create 4-by-4 matrix with random doubles
                   ,terminals=[(3, 3)])

def main():
    run_irl()

if __name__=="__main__":
    main()