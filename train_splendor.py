import logging

import coloredlogs

from Coach_mp import Coach
#from othello.OthelloGame import OthelloGame as Game
#from othello.pytorch.NNet import NNetWrapper as nn
from splendorgame.splendorgame import SplendorGame as Game
from splendorgame.pytorch.fcnet import FCNet as nn
from utils import *

log = logging.getLogger(__name__)

coloredlogs.install(level='INFO')  # Change this to DEBUG to see more info.

args = dotdict({
    'numIters': 1000,
    'numEps': 100,              # Number of complete self-play games to simulate during a new iteration.
    'tempThreshold': 20,        #
    'updateThreshold': 0.6,     # During arena playoff, new neural net will be accepted if threshold or more of games are won.
    'maxlenOfQueue': 200000,    # Number of game examples to train the neural networks.
    'numMCTSSims': 1600,          # Number of games moves for MCTS to simulate.
    'arenaCompare': 8,         # Number of games to play during arena play to determine if new net will be accepted.
    'cpuct': 4,
    'workers': 8,
    'max_round': 36,
    'checkpoint': '/home/jiao/rltest/jdata/azsplendor/splendor_chkpoints',
    'load_model': True,
    'load_folder_file': ('./splendor_chkpoints','best.pth.tar'),
    'load_examples': True,
    'load_example_folder_file':('./splendor_chkpoints','checkpoint_4.pth.tar'),
    'numItersForTrainExamplesHistory': 20,

})


def main():
    log.info('Loading %s...', Game.__name__)
    g = Game(max_round=args.max_round)

    log.info('Loading %s...', nn.__name__)
    nnet = nn(g)

    if args.load_model:
        log.info('Loading checkpoint "%s/%s"...', *args.load_folder_file)
        nnet.load_checkpoint(args.load_folder_file[0], args.load_folder_file[1])
    else:
        log.warning('Not loading a checkpoint!')

    log.info('Loading the Coach...')
    c = Coach(g, nnet, args)

    if args.load_examples:
        log.info("Loading 'trainExamples' from file...")
        c.loadTrainExamples()

    log.info('Starting the learning process 🎉')
    c.learn()


if __name__ == "__main__":
    main()
