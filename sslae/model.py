import warnings

import numpy as np
import pytorch_lightning as pl
import torch
import torch.nn as nn
from pytorch_lightning.trainer.connectors.data_connector import PossibleUserWarning
from torch import optim, isnan, tensor

warnings.filterwarnings("ignore", category=PossibleUserWarning)



class SemiSupervisedAutoEncoder(pl.LightningModule):
    def __init__(
        self,
        n_input: int,
        n_intrinsic: int,
        n_classes: int,
        ssl_w: float,
        random_state: int,
        mode: str = "MCC",
        lr: float = 1e-3,
        weight_decay: float = 0.0,
        dropout: float = 0.0,
    ):
        super().__init__()
        self._set_seeds(random_state)
        self.mode = mode.upper()
        n_hidden = round((n_input + n_intrinsic) // 2)
        n_hidden_cls = n_intrinsic + n_classes
        self.weight = ssl_w
        self.weight_decay = weight_decay
        self.dropout = dropout

        self.encoder = nn.Sequential(
            nn.Linear(n_input, n_hidden), nn.ReLU(), nn.Linear(n_hidden, n_intrinsic)
        )
        self.classifier = nn.Sequential(
            nn.Linear(n_intrinsic, n_hidden_cls),
            nn.ReLU(),
            nn.Dropout(self.dropout),
            nn.Linear(n_hidden_cls, n_classes),
        )
        self.decoder = nn.Sequential(
            nn.Linear(n_intrinsic, n_hidden), nn.ReLU(), nn.Linear(n_hidden, n_input)
        )

        self.r_loss = nn.SmoothL1Loss(reduction="mean")
        self.c_loss = nn.CrossEntropyLoss(reduction="none")
        self.lr = lr

        self.y_proba_train = np.empty(0)
        self.y_prediction_train = np.empty(0)
        self.y_proba_validation = np.empty(0)
        self.y_prediction_validation = np.empty(0)
        self.y_proba_test = np.empty(0)
        self.y_prediction_test = np.empty(0)

    def _set_seeds(self, seed):
        """Set random seeds for reproducibility."""
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)
        np.random.seed(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    def forward(self, x):
        z = self.encoder(x)
        return self.classifier(z)

    def configure_optimizers(self):
        return optim.Adam(self.parameters(), lr=self.lr, weight_decay=self.weight_decay)

    def compute_classification_loss(self, y_hat: tensor, y: tensor) -> tensor:
        null_mask = isnan(y).float()
        y[isnan(y)] = 0
        y_cl = y.view(-1).long()
        c_loss = self.c_loss(y_hat, y_cl)
        masked_loss = c_loss * (1 - null_mask.squeeze())
        denominator = null_mask.numel() - null_mask.sum()
        classification_loss = masked_loss.sum() / denominator
        if isnan(classification_loss):
            classification_loss = tensor(0.1, requires_grad=True)
        return classification_loss

    def get_prediction_probability_arrays(
        self, y_hat: tensor, batch_idx: int, step
    ) -> tensor:
        prediction_array_batch = (
            nn.functional.softmax(y_hat, dim=1).detach().cpu().float().numpy()
        )
        predicted_label_batch = np.argmax(prediction_array_batch, axis=1)

        if batch_idx == 0:
            self.reset_prediction_arrays(step, prediction_array_batch.shape[1])

        return prediction_array_batch, predicted_label_batch

    def reset_prediction_arrays(self, step, n_classes):
        if step == "train":
            self.y_proba_train = self.y_proba_train.reshape(-1, n_classes)
        elif step == "val":
            self.y_proba_validation = self.y_proba_validation.reshape(-1, n_classes)
        else:
            self.y_prediction_test = self.y_prediction_test.reshape(-1, 1)
            self.y_proba_test = self.y_proba_test.reshape(-1, n_classes)

    def shared_step(self, batch: tuple[tensor, tensor], batch_idx: int, step) -> tensor:
        x, y = batch
        x = x.view(x.size(0), -1)

        z = self.encoder(x)
        y_hat = self.classifier(z)
        x_hat = self.decoder(z)

        classification_loss = self.compute_classification_loss(y_hat, y)
        reconstruction_loss = self.r_loss(x_hat, x)

        (
            prediction_array_batch,
            predicted_label_batch,
        ) = self.get_prediction_probability_arrays(y_hat, batch_idx, step)

        return (
            (1 - self.weight) * classification_loss + self.weight * reconstruction_loss,
            prediction_array_batch,
            predicted_label_batch,
        )

    def training_step(
        self, train_batch: tuple[tensor, tensor], batch_idx: int
    ) -> tensor:
        loss, prediction_array_batch, predicted_label_batch = self.shared_step(
            train_batch, batch_idx, step="train"
        )
        self.log("train_loss", loss, prog_bar=True, on_epoch=True, on_step=False)

        self.y_proba_train = np.concatenate(
            (self.y_proba_train, prediction_array_batch)
        )

        self.y_prediction_train = np.append(
            self.y_prediction_train, predicted_label_batch
        )

        return loss

    def validation_step(self, val_batch: tuple[tensor, tensor], batch_idx: int):
        loss, prediction_array_batch, predicted_label_batch = self.shared_step(
            val_batch, batch_idx, step="val"
        )
        self.log("val_loss", loss, prog_bar=True, on_epoch=True, on_step=False)

        self.y_proba_validation = np.concatenate(
            (self.y_proba_validation, prediction_array_batch)
        )

        self.y_prediction_validation = np.append(
            self.y_prediction_validation, predicted_label_batch
        )

    def test_step(self, test_batch: tuple[tensor, tensor], batch_idx: int):
        loss, prediction_array_batch, predicted_label_batch = self.shared_step(
            test_batch, batch_idx, step="test"
        )

        self.log("test_loss", loss, prog_bar=True, on_epoch=True, on_step=False)

        self.y_prediction_test = np.append(
            self.y_prediction_test, predicted_label_batch
        )

        self.y_proba_test = np.concatenate((self.y_proba_test, prediction_array_batch))
