import numpy as np
import matplotlib.pyplot as plt
from dataclasses import dataclass
from enum import Enum

@dataclass
class Card:
    suit: str
    rank: str
    value: int = 0


class Action(Enum):
    HIT = 0
    STICK = 1


class Deck():
    def __init__(self):
        suits = ['♠', '♢', '♡', '♣']
        ranks = ['A', '2', '3', '4', '5', '6', '7', '8', '9', '10', 'J', 'Q', 'K']
        values = [11, 2, 3, 4, 5, 6, 7, 8, 9, 10, 10, 10, 10]
        self.cards = [ Card(s,r,v) for s in suits for (r, v) in zip(ranks, values)]
        self.shuffle()

    def shuffle(self):
        np.random.shuffle(self.cards)

    def __len__(self):
        return len(self.cards)

    def draw(self):
        if len(self) > 0:
            return self.cards.pop()
        else:
            raise ValueError("Deck empty!")

    def reset(self):
        print("Resetting deck")
        self.cards = [ f'{p}{s}' for p in self.pips for s in self.suits]


@dataclass
class GameObs:
    """State as observed by the agent."""
    dealer_visible: str
    player_total: int
    player_ace: bool
    player_reward: int = 0
    game_over: bool = False




class Agent():
    def __init__(self):
        pass

    def policy(self, g_obs: GameObs) -> Action:
        """Decide on an action based on a game observation."""
        pass


class RiskyAgent(Agent):
    def policy(self, g_obs: GameObs) -> Action:
        if g_obs.player_total >= 20:
            return Action.STICK
        else:
            return Action.HIT


class Dealer():
    """This is the environment for the blackjack example 5.1."""
    def __init__(self):
        self.reset_game()

    def draw_card(self):
        return self.deck.draw()

    def reset_game(self):
        self.deck = Deck()
        self.dealer_total = 0
        self.dealer_aces = 0
        self.dealer_visible = ""
        self.player_total = 0
        self.player_aces = 0
        self.game_over = False

    def dealer_move(self) -> None:
        """Dealer: hit or stick according to a fixed strategy."""

        while self.dealer_total < 17: # dealer hits
            c = self.deck.draw()
            if c.rank == 'A':
                # if new card is an ace
                if self.dealer_total + 11 <= 21:
                    # dealer uses the new ace as a an 11
                    self.dealer_aces += 1
                    self.dealer_total += 11
                else:
                    # dealer uses the new ace as a 1
                    self.dealer_total += 1
            else:
                if self.dealer_total + c.value > 21 and self.dealer_aces > 0:
                    # dealer would bust but has a usable ace
                    self.dealer_aces -= 1
                    self.dealer_total -= 10 + c.value
                else:
                    # dealer has no option but to add the total
                    self.dealer_total += c.value
        return


    def player_move(self, agent: Agent) -> None:
        # get a game observation
        g_obs = self.get_game_obs()

        # pass it to the agent
        agent_action = agent.policy(g_obs)
        print(agent_action)
        
        if agent_action == Action.HIT:
            while True:
                c = self.draw_card()
                # do update logic
                if c.rank == 'A':
                    # if new card is an ace
                    if self.player_total + 11 <= 21:
                        # player uses the new ace as a an 11
                        self.player_aces += 1
                        self.player_total += 11
                    else:
                        # player uses the new ace as a 1
                        self.player_total += 1
                else:
                    if self.player_total + c.value > 21 and self.player_aces > 0:
                        # player would bust but has a usable ace
                        self.player_aces -= 1
                        self.player_total -= 10 + c.value
                    else:
                        # player has no option but to add the total
                        self.player_total += c.value

                # select next action
                g_obs = self.get_game_obs()
                agent_action = agent.policy(g_obs)
                if agent_action == Action.STICK:
                    break

        return

    def game_loop(self, agent: Agent):
        if self.game_over:
            print("Error: game is already over!")
            return

        # init game
        self.start_game()

        # check if it's over already
        g_obs = self.get_game_obs()
        if g_obs.game_over:
            self.print_gameover(g_obs, natural=True)
            return g_obs.player_reward

        # player's turn
        self.player_move(agent)
        g_obs = self.get_game_obs()
        if g_obs.game_over:
            self.print_gameover(g_obs)
            return g_obs.player_reward

        # dealer's turn
        self.dealer_move()
        g_obs = self.get_game_obs()
        if g_obs.game_over:
            self.print_gameover(g_obs)
            return g_obs.player_reward
        
        # neither party busts
        self.game_over = True
        g_obs = self.check_winner()
        return g_obs.player_reward


    def check_winner(self):
        """If neither party busts, check to see who won."""
        print("No one busted. Checking totals")
        if self.player_total > self.dealer_total:
            reward = 1
        elif self.player_total == self.dealer_total:
            reward = 0
        else:
            reward = -1

        return GameObs(
            dealer_visible=self.dealer_visible,
            player_total=self.player_total,
            player_ace=self.player_aces,
            player_reward=reward,
            game_over=True,
        )



    def print_gameover(self, g_obs: GameObs, natural: bool = False):
        print("printing message")
        natural = "natural " if natural else ""
        if g_obs.player_reward == 0:
            print(f"Game ends in a {natural}draw!")
        elif g_obs.player_reward > 0:
            print(f"Game ends with a {natural}player victory!")
        else:
            print(f"Game ends with a {natural}dealer victory.")


    def start_game(self) -> None:
        """Initialize a game with two cards for each of the dealer and the player."""
        # initialize data
        self.deck = Deck()
        self.dealer_total = 0
        self.dealer_aces = 0
        self.player_total = 0
        self.player_aces = 0
        self.game_over = False

        # draw two cards for the dealer
        for _ in range(2):
            c = self.deck.draw()
            self.dealer_total += c.value
            if c.rank == 'A':
                self.dealer_aces += 1
            self.dealer_visible = c.value

        # draw two cards for the player
        for _ in range(2):
            c = self.deck.draw()
            self.player_total += c.value
            if c.rank == 'A':
                self.player_aces = True
        return


    def get_game_obs(self) -> GameObs:
        """Contains logic for checking game state and returning an observation."""
        if self.player_total == 21 and self.dealer_total == 21:
                # draw
                reward = 0
                self.game_over = True
        elif self.dealer_total > 21:
            # player wins (dealer busts)
            reward = +1
            self.game_over = True
        elif self.player_total > 21:
            # dealer wins (player busts)
            reward = -1
            self.game_over = True
        else:
            # game continues
            reward = 0
            self.game_over = False


        g_obs = GameObs(
            dealer_visible=self.dealer_visible,
            player_total=self.player_total,
            player_ace=self.player_aces,
            player_reward=reward,
            game_over=self.game_over,
        )
        return g_obs