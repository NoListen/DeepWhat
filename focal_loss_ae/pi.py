class GymRandomPolicy:
    def __init__(self, action_space, seed=None):
        self.action_space = action_space
        if seed is not None:
            action_space.seed(None)

    def act(self, *args):
        return self.action_space.sample()
