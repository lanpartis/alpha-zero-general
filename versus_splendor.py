from splendorgame.splendorgame import SplendorGame as Game
from splendorgame.pytorch.fcnet import FCNet as nn
import Arena
from MCTS import MCTS
from splendor.players.h5 import H5Player
import numpy as np
from utils import *
args = dotdict({
    'numMCTSSims': 1600,          # Number of games moves for MCTS to simulate.
    'max_round':35,
    'checkpoint': './splendor_chkpoints/',
    'cpuct':1,
    'load_model': True,
    'load_folder_file': ('./splendor_chkpoints','best.pth.tar'),

})

def main():
    g = Game(max_round=args.max_round)
    nnet = nn(g)

    if args.load_model:
        print('Loading checkpoint "%s/%s"...', *args.load_folder_file)
        nnet.load_checkpoint(args.load_folder_file[0], args.load_folder_file[1])
    else:
        print('Not loading a checkpoint!')
    
    opponent = MCTS(g, nnet, args)

    player = H5Player()
    board = g.getInitBoard()

    while(True):
        while (board.current_player==0):
            
            act = opponent.getActionProb(board,temp=0)
            act = np.argmax(act)
            # print(opponent.Qsa.get((g.stringRepresentation(board),act),0)
            board, _ = g.getNextState(board,1,act)
        while (board.current_player==1):
            act = player(g.env.env.splendor.observation())
            g.env.env.splendor.move(act)
            board = g.env.env.splendor.state
        if board.winner is not None:
            break


if __name__ == '__main__':
    main()