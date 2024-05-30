
import collections

State = collections.namedtuple(
    'State',
    ['player_total', 'dealer_card_shown', 'has_usable_ace']
)

MAP_STATE_TO_IDXS = { v: k for k, v in enumerate([
    State(pt, dvs, hua) for pt in range(12, 22) for dvs in range(1, 11) for hua in (False, True)
])}



class Q:
    def __init__(self):
        self.values = np.ones((10,10,2,2)) # s1, s2, s3, a

    def __getitem(self, key):
        if len(key) == 2 and isinstance(key[0], State) and isinstance(key[1], int):
            state, action = key
            s_idx = MAP_STATE_TO_IDXS[state]
            return self.values[s_idx,]




Card = collections.namedtuple(
    'Card',
    ['rank', 'value'],
)

class BlackjackDeck:
    """Simplified deck for generating card draws."""
    def __init__(self):
        ranks = [str(n) for n in range(2, 11)] + list('JQKA')
        values = list(range(2, 11)) + [10]*3 + [11]
        self._cards = [Card(r,v) for r,v in zip(ranks, values)]

    def draw(self):
        return random.choice(self._cards)

DECK = BlackjackDeck()


class BlackjackPlayer:
    def __init__(self):
        pass

    def reset(self):
        """Start a new game."""
        self.hand = [DECK.draw(), DECK.draw()]

    def draw(self):
        self.hand.append(DECK.draw())
    
    @property
    def has_usable_ace(self):
        return 'A' in [h.rank for h in self.hand]

    @property
    def bust(self):
        return self.total > 21

    @property
    def total(self):
        total_ = sum([h.value for h in self.hand])
        if total_ > 21 and self.has_usable_ace:
            total_ -= 10
        return total_

    @property
    def has_natural(self):
        if len(self.hand) == 2:
            ranks = [h.rank for h in self.hand]
            vals = [h.value for h in self.hand]
            if 'A' in ranks and 10 in vals:
                return True
        return False
    

class Player(BlackjackPlayer):

    def __init__(self):
        self.Q = Q()
        self.returns = collections.defaultdict(list)
        self.reset()

    def policy(self, state: State):
        """Act greedily wrt current action-value function."""
        return np.argmax(self.Q[state,:])

    
class Dealer(BlackjackPlayer):
    def __init__(self):
        self.reset()

    @property
    def shown(self) -> int:
        if self.hand[0].rank == 'A':
            return 1
        else:
            return self.hand[0].value

    def policy(self):
        if self.total < 17:
            return 1 # hit
        return 0 # stay