# early stop class

class EarlyStopper:
    """
    A class that implements early stopping based on validation loss.

    Attributes:
    ----------
        patience (int):
          The number of epochs to wait for improvement in validation loss before stopping.
        min_delta (float):
          The minimum change in validation loss required to be considered as improvement.
        counter (int):
          The number of epochs without improvement in validation loss.
        min_validation_loss (float):
          The minimum validation loss achieved so far.

    Methods:
    ----------
        early_stop(validation_loss):
          Checks if early stopping criteria is met based on the given validation loss.

    """

    def __init__(self, patience=10, min_delta=0):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.min_validation_loss = float('inf')

    def early_stop(self, validation_loss):
        """
        Checks if early stopping criteria is met based on the given validation loss.

        Args:
        ----------
            validation_loss (float):
              The current validation loss.

        Returns:
        ----------
            bool:
              True if early stopping criteria is met, False otherwise.

        """
        if validation_loss < self.min_validation_loss:
            self.min_validation_loss = validation_loss
            self.counter = 0
        elif validation_loss > (self.min_validation_loss + self.min_delta):
            self.counter += 1
            if self.counter >= self.patience:
                return True
        return False