from sample_players import DataPlayer
import pickle
import random
import copy
import math


class CustomPlayer(DataPlayer):
    """ Implement your own agent to play knight's Isolation

    The get_action() method is the only required method for this project.
    You can modify the interface for get_action by adding named parameters
    with default values, but the function MUST remain compatible with the
    default interface.

    **********************************************************************
    NOTES:
    - The test cases will NOT be run on a machine with GPU access, nor be
      suitable for using any other machine learning techniques.

    - You can pass state forward to your agent on the next turn by assigning
      any pickleable object to the self.context attribute.
    **********************************************************************
    """
    def get_action(self, state):
        """ Employ an adversarial search technique to choose an action
        available in the current state calls self.queue.put(ACTION) at least

        This method must call self.queue.put(ACTION) at least once, and may
        call it as many times as you want; the caller will be responsible
        for cutting off the function after the search time limit has expired.

        See RandomPlayer and GreedyPlayer in sample_players for more examples.

        **********************************************************************
        NOTE: 
        - The caller is responsible for cutting off search, so calling
          get_action() from your own code will create an infinite loop!
          Refer to (and use!) the Isolation.play() function to run games.
        **********************************************************************
        """
        # TODO: Replace the example implementation below with your own search
        #       method by combining techniques from lecture
        #
        # EXAMPLE: choose a random move without any search--this function MUST
        #          call self.queue.put(ACTION) at least once before time expires
        #          (the timer is automatically managed for you)

        if state.ply_count < 2:
          self.queue.put(random.choice(state.actions()))
        else:
          self.queue.put(self.alpha_beta_pruning(state, float("-inf"), float("inf"), 4))
        
    def alpha_beta_pruning(self, state, alpha, beta, depth = 4):
        
        def min_value(state, alpha, beta, depth = 4):
            if state.terminal_test(): 
              return state.utility(self.player_id)
            if depth <= 0:
              return self.offensive_to_defensive_score(state, depth)
            value = float("inf")
            for action in state.actions():
                value = min(value, max_value(state.result(action), alpha, beta, depth - 1))
                if alpha >= value:
                  return alpha
                else:
                  beta = value
            return value

        def max_value(state, alpha, beta, depth = 4):
            if state.terminal_test(): 
              return state.utility(self.player_id)
            if depth <= 0:
              return self.offensive_to_defensive_score(state, depth)
            value = float("-inf")
            for action in state.actions():
              value = max(value, min_value(state.result(action), alpha, beta, depth - 1))
              if beta <= value:
                return beta
              else:
                alpha = value
            return value

        return max(state.actions(), key = lambda x: min_value(state.result(x), alpha, beta, depth))

    def baseline_score(self, state, depth = 1):
        own_loc = state.locs[self.player_id]
        opp_loc = state.locs[1 - self.player_id]
        own_liberties = state.liberties(own_loc)
        opp_liberties = state.liberties(opp_loc)
        return len(own_liberties) - len(opp_liberties)
    
    def defensive_score(self, state, depth = 1):
        own_loc = state.locs[self.player_id]
        opp_loc = state.locs[1 - self.player_id]
        own_liberties = state.liberties(own_loc)
        opp_liberties = state.liberties(opp_loc)
        return 2 * len(own_liberties) - len(opp_liberties)
    
    def offensive_score(self, state, depth = 1):
        own_loc = state.locs[self.player_id]
        opp_loc = state.locs[1 - self.player_id]
        own_liberties = state.liberties(own_loc)
        opp_liberties = state.liberties(opp_loc)
        return len(own_liberties) - 2 * len(opp_liberties)
    
    def defensive_to_offensive_score(self, state, depth = 1):
        own_loc = state.locs[self.player_id]
        opp_loc = state.locs[1 - self.player_id]
        own_liberties = state.liberties(own_loc)
        opp_liberties = state.liberties(opp_loc)
        return len(own_liberties) - max(1, 4 - depth) * len(opp_liberties)
    
    def offensive_to_defensive_score(self, state, depth = 1):
        own_loc = state.locs[self.player_id]
        opp_loc = state.locs[1 - self.player_id]
        own_liberties = state.liberties(own_loc)
        opp_liberties = state.liberties(opp_loc)
        return len(own_liberties) - max(1, depth) * len(opp_liberties)
