__author__ = 'erensezener'

from mdp import *
from utils import *
from copy import copy


def run_irl():
    iteration_limit = 100

    expert_mdp = GridMDP([[-0.04, -0.04, -0.04, +1],
                          [-0.04, -0.04,  -0.04, -0.04],
                          [-0.04, -0.04, -0.04, -0.04],
                          [-0.04, -0.04, -0.04, -0.04]],
                         terminals=[(3, 3)])
    # expert_pi = policy_iteration(expert_mdp)
    expert_pi = best_policy(expert_mdp,value_iteration(expert_mdp, 0.1))
    # print_table(expert_mdp.to_arrows(expert_pi))
    # print "vs"
    # print_table(expert_mdp.to_arrows(expert_pi2))
    # return None

    best_difference = 16
    best_pi = None
    best_mdp = None

    for i in range(iteration_limit):
        pi, mdp, diff = iterate_irl(expert_pi)
        if get_difference(pi, expert_pi) < best_difference:
            best_difference = get_difference(pi, expert_pi)
            best_pi = pi
            best_mdp = mdp
            print("Difference is " + str(get_difference(best_pi, expert_pi)))


    print_table(best_mdp.to_arrows(best_pi))
    print "vs"
    print_table(expert_mdp.to_arrows(expert_pi))
    print("Difference is " + str(get_difference(best_pi, expert_pi)))
    mdp.print_rewards()
    print "vs"
    expert_mdp.print_rewards()




def iterate_irl(expert_pi, iteration_limit = 500, epsilon = 0.3):
    mdp = create_random_rewards()
    pi = policy_iteration(mdp)

    for iter in range(iteration_limit):
        new_mdp = copy(mdp) #creates a new reward function that is very similar to the original one
        new_mdp.modify_rewards_randomly(epsilon)
        new_pi = policy_iteration(mdp)
        if get_difference(new_pi, expert_pi) == 0:
            return new_pi, new_mdp, get_difference(new_pi, expert_pi)
        elif get_difference(new_pi, expert_pi) < get_difference(pi, expert_pi): #if policy is more similar to the exper policy
            mdp = new_mdp
            pi = new_pi
            # print ("Improvement in " + str(iter))
            # print("Difference is " + str(get_difference(new_pi, expert_pi)))
    return pi, mdp, get_difference(pi, expert_pi)


def get_difference(new_pi, ex_pi):
    shared_items = set(new_pi.items()) & set(ex_pi.items())
    return len(new_pi.items()) - len(shared_items)


def create_random_rewards():
    return GridMDP([[random.uniform(-1,1) for i in range(4)] for j in range(4)] #create 4-by-4 matrix with random doubles
                   ,terminals=[(3, 3)])


def main():
    run_irl()

if __name__=="__main__":
    main()