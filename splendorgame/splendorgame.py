from __future__ import print_function
import sys

from splendor.core.core import Player
sys.path.append('..')
from Game import Game
import numpy as np
from .splendor_base_wrapper import SplendorBaseWrapper, ActionTransTool, ObservationTransTool
from splendor.envs.env import SplendorEnv
from splendor.core.obs import Obs
import random

class SplendorGame(Game):

    def get_player_game_id(self, MCTS_id):
        return self.player if MCTS_id == 1 else (1 - self.player)

    def __init__(self, player = 0, max_round = 40):
        self.env = SplendorBaseWrapper(SplendorEnv(player_num=2))
        self.player = player
        self.max_round = max_round
        
    def seed(self, seed=None):
        self.env.seed(seed)

    def getInitBoard(self):
        """
        Returns:
            startBoard: a representation of the board (ideally this is the form
                        that will be the input to your neural network)
        """
        self.env.reset()
        return self.env.splendor.state

    def getBoardSize(self):
        """
        Returns:
            (x,y): a tuple of board dimensions
        """
        return self.env.observation_space['obs'].shape[0]

    def getActionSize(self):
        """
        Returns:
            actionSize: number of all possible actions
        """
        return self.env.action_space.n

    def getNextState(self, board, player, action, shuffle=False):
        """
        Input:
            board: current board
            player: current player (1 or -1)
            action: action taken by current player

        Returns:
            nextBoard: board after applying action
            nextPlayer: player who plays in the next turn (should be -player)
        """
        self.env.load(board)

        if shuffle:
            #randomize game
            lst = self.env.env.splendor.state.cards.tier1[:-4]
            random.shuffle(lst)
            self.env.env.splendor.state.cards.tier1[:-4] = lst
            lst = self.env.env.splendor.state.cards.tier2[:-4]
            random.shuffle(lst)
            self.env.env.splendor.state.cards.tier2[:-4] = lst
            lst = self.env.env.splendor.state.cards.tier3[:-4]
            random.shuffle(lst)
            self.env.env.splendor.state.cards.tier3[:-4] = lst
        # self.env.splendor
        obs, reward, done, info = self.env.step(action)
        state = self.env.splendor.state
        return state, self.currentPlayer(state)


    def getValidMoves(self, board, player):
        """
        Input:
            board: current board
            player: current player

        Returns:
            validMoves: a binary vector of length self.getActionSize(), 1 for
                        moves that are valid from the current board and player,
                        0 for invalid moves
        """
        obs = Obs.from_state(board)
        action_mask = ActionTransTool.valid_action(obs)
        return action_mask

    def getGameEnded(self, board, player):
        """
        Input:
            board: current board
            player: current player (1 or -1)

        Returns:
            r: 0 if game has not ended. 1 if player won, -1 if player lost,
               small non-zero value for draw.
               
        """
        obs = Obs.from_state(board)
        done = 0 
        player_game_id = self.get_player_game_id(player)
        if obs.winner is not None:
            done = 1 if obs.winner == player_game_id else -1
        if obs.winner is None and board.play_round > self.max_round:
            done = 0.1 * (board.players[player_game_id].score - board.players[1-player_game_id].score)
            if board.players[player_game_id].score == board.players[1-player_game_id].score:
                done = 0.1 if sum(board.players[player_game_id].cards.values()) > sum(board.players[1-player_game_id].cards.values()) else -0.1
        
        return done

    def getCanonicalForm(self, board, player=1):
        """
        Input:
            board: current board
            player: current player (1 or -1)

        Returns:
            canonicalBoard: returns canonical form of board. The canonical form
                            should be independent of player. For e.g. in chess,
                            the canonical form can be chosen to be from the pov
                            of white. When the player is white, we can return
                            board as is. When the player is black, we can invert
                            the colors and return the board.
        """
        raw_obs = Obs.from_state(board)
        obs = ObservationTransTool.obs_to_state(raw_obs)
        return obs

    def getSymmetries(self, board, pi):
        """
        Input:
            board: current board
            pi: policy vector of size self.getActionSize()

        Returns:
            symmForms: a list of [(board,pi)] where each tuple is a symmetrical
                       form of the board and the corresponding pi vector. This
                       is used when training the neural network from examples.
        """
        # raw_obs_from_state = Obs.from_state(board)
        # obs = ObservationTransTool.obs_to_state(raw_obs_from_state)
        #todo positional symmetries
        return [(board, pi)]

    def stringRepresentation(self, board):
        """
        Input:
            board: current board

        Returns:
            boardString: a quick conversion of board to a string format.
                         Required by MCTS for hashing.
        """
        raw_obs = Obs.from_state(board)
        obs = ObservationTransTool.obs_to_state(raw_obs)
        reprsentation = ''
        for o in obs:
            reprsentation+=str(int(o))
        return reprsentation

    def currentPlayer(self, board):
        """return current player

        Args:
            board (splendor.core.state.State): current state

        Returns:
            1 or -1: current_player
        """        
        return 1 if self.player == board.current_player else -1