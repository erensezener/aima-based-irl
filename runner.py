"""
Author: Eren Sezener (erensezener@gmail.com)
Date: April 4, 2014

Description:

Status: Works correctly.

Dependencies:

Known bugs: -

"""

from birl import *

def main():
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

    expert_mdp = GridMDP([[0, 0, 0, 0, -1, -1, 0, 0, 0, 10],
                        [0, 0, 0, -3, -3, -3, -3, 0, 0, 0],
                        [0, 0, 0, -3, -5, -5, -3, 0, 0, 0],
                        [0, 0, 0, -3, -3, -3, -3, 0, 0, 0],
                        [0, 0, 0, 0, 0, -1, -1, 0, 0, 0]],
                        terminals=[(9,4)])


    birl = BIRL(expert_mdp)
    birl.run_multiple_birl()

if __name__=="__main__":
    main()