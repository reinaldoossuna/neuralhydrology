class DummyLRS:
    def __init__(self, optmizer):
        self.optimizer = optmizer

    def step(self):
        pass

    def get_last_lr(self):
        return [self.optimizer.param_groups[0]["lr"]]
