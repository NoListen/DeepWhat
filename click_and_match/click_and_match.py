import numpy as np
import random

# There are two challenges in this small game.
# One is partial observation at each time.
# Antoher is the bad representation of each one.
# One hot ??? OK, it is included in the later processing.

click_reward = -0.1
shuffle_time = 10
random_seed = 1234567
random.seed(random_seed)

# I need a MASK and dropout in later processing.

class Click_And_Match(object):
    # By Default, size is one number.
    # But it can be a tuple
    def __init__(self, size, n, match_num=2):
        n_positions = 0
        if isinstance(size, int):
            dim = 2
            self.shape = (size, size)
            n_positions = size * size
        elif isinstance(size, tuple):
            dim = len(size)
            self.shape = size
            # can be one-dim or several-dim
            n_position = np.product(size)
        else:
            raise ValueError('invalid size')

        self.map = np.zeros(self.shape)
        self.visual_map = np.zeros(self.shape)

        # nothing left.
        # TODO allow things are left but to achieve the maxium reward. Closer to the real world.
        assert(n_positions % match_num == 0 and n_positions >= match_num*n)
        self.n = n
        self.match_num = match_num
        self.n_positions = n_positions
        self.click_times = 0
        self.click_pos_history = []
        self.dim = dim
        self.n_positions_left = self.n_positions


    def reset(self):
        self.generate_map()
        self.visual_map = np.zeros(self.shape)
        self.clean_click_history()
        self.n_positions_left = self.n_positions
        self.visual_map = np.zeros(self.shape)

    def generate_map(self):
        # from 1 -> n
        s = range(1, self.n+1) * (self.n_positions // ( self.n * self.match_num ))
        remain = self.n_positions % ( self.n * self.match_num) // self.match_num
        add_up = []
        for _ in range(remain):
            add_up.append(random.randint(1, self.n))
        s += add_up
        s *= self.match_num
        for _ in range(shuffle_time):
            random.shuffle(s)
        self.map = np.array(s).reshape(self.shape)


    def clean_click_history(self):
        self.click_times = 0
        self.click_pos_history = []

    def reset_visual_map(self):
        his = self.click_pos_history
        for p in his:
            self.visual_map[p] = 0

    def vanish_or_not(self):
        assert(len(self.click_pos_history) == self.match_num)
        his = self.click_pos_history
        v = self.map[his[0]]
        for i in range(1, self.match_num):
            if v != self.map[his[i]]:
                self.reset_visual_map()
                return False
        for p in his:
            self.map[p] = self.visual_map[p] = -1
        self.n_positions_left -= self.match_num
        return True

    # NULL is expressed as -1
    # pos is a tuple but expressed as different action in each axis.
    # TODO use softmax to decide the action.
    # I mean the spatial action.
    # TODO HOW to understand no action. ( IF I THINK MY EACH ACTION HAVE INFLUENCE ON THE ENCIROMENT, IT'LL BE CONFUSING.)
    # there is a research about action related environment change.

    # IF it's a mapping, thinking too much means overfitting.


    def click(self, pos):

        if pos is None:
            reward = 0
        else:
            reward = click_reward

        # ACTION HAS NO INFLUENCE THIS TIME.
        if self.click_times == self.match_num:
            v = self.vanish_or_not()
            self.clean_click_history()

            if v:
                reward += 1
            if self.n_positions_left != 0:
                return reward, False
            else:
                return reward, True

        if pos is None:
            return 0, False
        assert (len(pos) == self.dim and self.map[pos] != -1)
        assert (len(self.click_pos_history) == self.click_times)
        self.click_times += 1
        self.click_pos_history.append(pos)
        # whatever, it should be seen as a result.
        # but will be turned back immediately.
        self.visual_map[pos] = self.map[pos]
        return reward, False


def test():
    env = Click_And_Match(2, 2)
    env.reset()
    done = False
    print env.visual_map
    print env.map
    while not done:
        x = int(raw_input("enter x:"))
        y = int(raw_input("enter y:"))
        r, done = env.click((x,y))
        print env.n_positions_left
        print r,done
        print env.visual_map
        print env.map
test()


        
        


