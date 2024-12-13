import numpy as np
import torch


class EarlyStopping:
    def __init__(self, path_to_save, delta=0.01, patience=3, restore_best_weights=True):
        """
        delta là tham số kiểm soát độ dao động của loss, loss ở các epoch đầu được phép dao dộng lớn hơn
        loss ở các epoch về sau patience là số lượng epoch mà loss không giảm thì sẽ dừng train
        restore_best_weights là lựa chọn có restore lại weight tốt nhất hay không
        """
        self.best_val_loss_average = np.inf
        self.best_val_loss = np.inf
        self.path_to_save = path_to_save
        self.early_stop = False
        self.delta = delta
        self.patience = patience
        self.counter = 0
        self.restore_best_weights = restore_best_weights
        self.best_weights = None

    def __call__(self, current_average_loss_val, current_loss_val, model):
        if current_average_loss_val < self.best_val_loss_average:
            print(
                "loss validation average decrease from {0} to {1}, Saving checkpoint...".format(
                    self.best_val_loss_average, current_average_loss_val
                )
            )
            self.best_val_loss_average = current_average_loss_val
            self.best_val_loss = current_loss_val
            self.counter = 0
            self.best_weights = model.state_dict()
            torch.save(model.state_dict(), self.path_to_save)
        elif current_loss_val < self.best_val_loss:
            print(
                "loss validation decrease from {0} to {1}, Saving checkpoint...".format(
                    self.best_val_loss, current_loss_val
                )
            )
            self.best_val_loss = current_loss_val
            self.counter = 0
            self.best_weights = model.state_dict()
            torch.save(model.state_dict(), self.path_to_save)
        elif (
            current_average_loss_val
            < self.best_val_loss_average + self.delta * self.best_val_loss_average
        ):
            print(
                "loss validation average slightly increased from {0} to {1}, No save! ".format(
                    self.best_val_loss_average, current_average_loss_val
                )
            )
            self.counter += 1
        else:
            self.counter += 1
            print(
                "No improvement. Early stopping counter: {0}/{1}".format(
                    self.counter, self.patience
                )
            )

        # Nếu counter >= patience thì dừng train
        if self.counter >= self.patience:
            print("Early stopping")
            self.early_stop = True
            if self.restore_best_weights and self.best_weights is not None:
                model.load_state_dict(self.best_weights)
                print("Restore best weights")
