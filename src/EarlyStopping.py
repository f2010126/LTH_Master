class EarlyStopping:
    def __init__(self, patience=5, min_delta=.09):
        """

        :param patience: how long to wait w/o loss improving
        :param min_delta: difference b/w current loss and best loss
        """
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_loss = None
        self.early_stop = False

    def __call__(self, val_loss):
        if self.best_loss is None:
            self.best_loss = val_loss
        elif self.best_loss - val_loss < self.min_delta:
            # no improvement found
            self.counter += 1
            # print(f"counter {self.counter} of {self.patience}")
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            # loss is reducing
            self.best_loss = val_loss
            self.counter = 0
