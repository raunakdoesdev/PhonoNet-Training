class Averager:
    def __init__(self):
        self.sum = 0.
        self.count = 0.

    def __call__(self, x=None):
        if x is None:
            if self.count == 0:
                return -1
            return self.sum / self.count
        else:
            self.count += 1.
            self.sum += x
