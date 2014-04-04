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
    expert_mdp = GridMDP([[-10, -5, 0, 0, 10],
            [-5, -3, 0, 0, 0],
            [0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0]],
            terminals=[(4,3)])
    birl = BIRL(expert_mdp)
    birl.run_birl()

if __name__=="__main__":
    main()