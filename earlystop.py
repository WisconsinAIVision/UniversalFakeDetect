class EarlyStopping:
    def __init__(self, patience=7, verbose=False, delta=0):
        self.patience = patience
        self.verbose = verbose
        self.delta = delta
        self.counter = 0
        self.best_score = None
        self.early_stop = False

    def __call__(self, score, model):
        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(score, model)
            return

        if score < self.best_score + self.delta:
            self.counter += 1
            if self.verbose:
                print("EarlyStopping counter: %d out of %d" % (self.counter, self.patience))
            if self.counter >= self.patience:
                self.early_stop = True
            return

        self.best_score = score
        self.save_checkpoint(score, model)
        self.counter = 0

    def save_checkpoint(self, score, model):
        if self.verbose:
            print("Validation score improved to %.6f. Saving model_epoch_best.pth" % score)
        model.save_networks("model_epoch_best.pth")
