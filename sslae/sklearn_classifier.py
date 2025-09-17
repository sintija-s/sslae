import random
import time

import numpy as np
import pytorch_lightning as pl
import torch
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.utils.validation import check_X_y, check_array, check_is_fitted

import sslae.dataset as dataset
from sslae.model import SemiSupervisedAutoEncoder


class SSLAEClassifier(BaseEstimator, ClassifierMixin):
    """
    Semi-Supervised Learning Auto-Encoder Classifier

    A scikit-learn compatible wrapper for the Semi-Supervised AutoEncoder that
    handles both labeled and unlabeled data.

    Parameters
    ----------
    n_intrinsic : int or str, default="auto"
        The dimension of the latent space. If "auto", it will be estimated using
        the TwoNN algorithm.

    ssl_weight : float, default=0.5
        The weight given to the reconstruction loss (0.0 for supervised only,
        1.0 for unsupervised only)

    batch_size : int, default=32
        Batch size for training

    learning_rate : float, default=1e-3
        Learning rate for the optimizer

    weight_decay : float, default=0.0
        Weight decay for the optimizer

    num_epochs : int, default=100
        Maximum number of training epochs

    stopping_crit : int, default=10
        Number of epochs with no improvement after which training will be stopped

    dropout : float, default=0.0
        Dropout rate for the classifier

    mode : str, default="MCC"
        "MCC" for multi-class classification or "MLC" for multi-label classification

    random_state : int, default=None
        Seed for random number generators

    verbose : bool, default=False
        Whether to print progress during training
    """

    def __init__(
        self,
        n_intrinsic="auto",
        ssl_weight=0.5,
        batch_size=32,
        learning_rate=1e-3,
        weight_decay=0.0,
        num_epochs=100,
        stopping_crit=10,
        dropout=0.0,
        mode="MCC",
        random_state=42,
        verbose=False,
    ):
        self.n_intrinsic = n_intrinsic
        self.ssl_weight = ssl_weight
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.num_epochs = num_epochs
        self.stopping_crit = stopping_crit
        self.dropout = dropout
        self.mode = mode
        self.random_state = random_state
        self.verbose = verbose
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.scaler = StandardScaler()

    def _set_seeds(self, seed):
        """Set random seeds for reproducibility."""
        if seed is not None:
            torch.manual_seed(seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed(seed)
                torch.cuda.manual_seed_all(seed)
            np.random.seed(seed)
            random.seed(seed)
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
            pl.seed_everything(seed)

    def _find_intrinsic_dimension(self, X):
        """Estimate the intrinsic dimension of data using TwoNN algorithm."""
        d = dataset.twonn_dimension(X)
        # If the estimated dimension is larger than input dimension, cap it
        if d > X.shape[1]:
            d = round(X.shape[1] / 3)
        return round(d)

    def fit(self, X_labeled, X_unlabeled, y_labeled, X_val=None, y_val=None):
        """
        Fit the semi-supervised auto-encoder model with labeled and unlabeled data.

        Parameters
        ----------
        X_labeled : array-like of shape (n_labeled_samples, n_features)
            Labeled data features

        X_unlabeled : array-like of shape (n_unlabeled_samples, n_features)
            Unlabeled data features

        y_labeled : array-like of shape (n_labeled_samples,)
            Labels for X_labeled

        X_val : array-like of shape (n_val_samples, n_features), optional
            Validation data features. If None, a portion of X_labeled will be used.

        y_val : array-like of shape (n_val_samples,), optional
            Labels for X_val. Must be provided if X_val is provided.

        Returns
        -------
        self : object
            Returns self.
        """
        self._set_seeds(self.random_state)

        # Validation checks
        X_labeled, y_labeled = check_X_y(X_labeled, y_labeled, accept_sparse=False)
        X_unlabeled = check_array(X_unlabeled, accept_sparse=False)

        # Store training data properties
        self.n_features_in_ = X_labeled.shape[1]
        self.classes_ = np.unique(y_labeled)
        self.n_classes_ = len(self.classes_)

        # If validation data is not provided, create it from the labeled data
        if X_val is None or y_val is None:
            # Try to do a stratified split if possible, otherwise do a regular split
            try:
                X_train_labeled, X_val, y_train_labeled, y_val = train_test_split(
                    X_labeled,
                    y_labeled,
                    test_size=0.15,
                    random_state=self.random_state,
                    stratify=y_labeled,
                )
            except:
                X_train_labeled, X_val, y_train_labeled, y_val = train_test_split(
                    X_labeled, y_labeled, test_size=0.15, random_state=self.random_state
                )
        else:
            X_val, y_val = check_X_y(X_val, y_val, accept_sparse=False)
            X_train_labeled = X_labeled
            y_train_labeled = y_labeled

        # Create NaN labels for unlabeled data
        y_unlabeled = np.full(X_unlabeled.shape[0], np.nan)

        # Determine the intrinsic dimension
        if self.n_intrinsic == "auto":
            self.n_intrinsic_ = self._find_intrinsic_dimension(X_train_labeled)
        else:
            self.n_intrinsic_ = self.n_intrinsic

        # Calculate the unlabeled proportion
        unlabeled_proportion = X_unlabeled.shape[0] / (
            X_train_labeled.shape[0] + X_unlabeled.shape[0]
        )

        # Generate the semi-supervised batch
        X_train_ssl, y_train_ssl = dataset.generate_semi_supervised_batch(
            X_train_labeled,
            X_unlabeled,
            y_train_labeled,
            y_unlabeled,
            self.batch_size,
            unlabeled_proportion,
        )

        # Initialize dataloaders
        trainds_ssl_loader = dataset.construct_dataloaders(
            X_train_ssl, y_train_ssl, batch_size=self.batch_size
        )
        trainds_sl_loader = dataset.construct_dataloaders(
            X_train_labeled, y_train_labeled, batch_size=self.batch_size
        )
        valds_loader = dataset.construct_dataloaders(
            X_val, y_val, batch_size=self.batch_size
        )

        # Determine which dataloader to use based on ssl_weight
        train_loader = (
            trainds_ssl_loader if self.ssl_weight > 0.0 else trainds_sl_loader
        )

        # Initialize the model
        self.model_ = SemiSupervisedAutoEncoder(
            n_input=self.n_features_in_,
            n_intrinsic=self.n_intrinsic_,
            n_classes=self.n_classes_,
            ssl_w=self.ssl_weight,
            random_state=self.random_state if self.random_state else 42,
            mode=self.mode,
            lr=self.learning_rate,
            weight_decay=self.weight_decay,
            dropout=self.dropout,
        )

        # Configure trainer
        callbacks = [
            pl.callbacks.EarlyStopping(
                monitor="val_loss", mode="min", patience=self.stopping_crit
            )
        ]

        if self.verbose:
            callbacks.append(pl.callbacks.RichProgressBar(leave=True))

        trainer = pl.Trainer(
            max_epochs=self.num_epochs,
            enable_checkpointing=False,
            logger=False,
            deterministic=True,
            callbacks=callbacks,
            enable_progress_bar=self.verbose,
        )

        # Train the model
        start_time = time.time()
        trainer.fit(self.model_, train_loader, valds_loader)
        end_time = time.time()
        self.execution_time_ = end_time - start_time
        self.trainer_ = trainer
        return self

    def predict(self, X_test):
        """
        Predict class labels for samples in X.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            The input samples.

        Returns
        -------
        y : ndarray of shape (n_samples,)
            The predicted classes.
        """
        check_is_fitted(self)
        X_test = check_array(X_test, accept_sparse=False)

        test_loader = dataset.construct_dataloaders(
            X_test, np.zeros(X_test.shape[0]), batch_size=self.batch_size
        )
        self.model_.y_prediction_test = np.empty(0)
        self.trainer_.test(self.model_, test_loader)
        y_pred = self.model_.y_prediction_test

        return y_pred

    def predict_proba(self, X_test):
        """
        Predict class probabilities for samples in X.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            The input samples.

        Returns
        -------
        y_proba : ndarray of shape (n_samples, n_classes)
            The class probabilities of the input samples.
        """
        check_is_fitted(self)
        X_test = check_array(X_test, accept_sparse=False)

        test_loader = dataset.construct_dataloaders(
            X_test, np.zeros(X_test.shape[0]), batch_size=self.batch_size
        )

        self.model_.y_proba_test = np.empty(0)
        self.trainer_.test(self.model_, test_loader)
        y_score = self.model_.y_proba_test

        return y_score

    def score(self, y_test, y_test_predictions, y_test_probabilities):
        """
        Get the performance metrics of the model.

        Parameters
        ----------
        y_test : array-like of shape (n_samples,)
            The true labels of the input samples.

        y_test_probabilities : array-like of shape (n_samples, n_classes)
            The predicted probabilities of the input samples.

        y_test_predictions : array-like of shape (n_samples,)
            The predicted labels of the input samples.

        Returns
        -------
        perf_dict : dict
            A dictionary containing the performance metrics of the model.
        """
        from sklearn.metrics import (
            accuracy_score,
            average_precision_score,
            f1_score,
            precision_score,
            recall_score,
            roc_auc_score,
        )

        # applies in the case of binary classification, where the probability of class 1 is taken
        if y_test_probabilities.shape[1] == 2:
            y_test_probabilities = y_test_probabilities[:, 1]

        accuracy = accuracy_score(y_test, y_test_predictions)
        auprc_macro = average_precision_score(
            y_test, y_test_probabilities, average="macro"
        )
        auprc_weighted = average_precision_score(
            y_test, y_test_probabilities, average="weighted"
        )
        auc_ovo = roc_auc_score(y_test, y_test_probabilities, multi_class="ovo")
        f1_macro = f1_score(y_test, y_test_predictions, average="macro")
        f1_weighted = f1_score(y_test, y_test_predictions, average="weighted")
        precision_macro = precision_score(y_test, y_test_predictions, average="macro")
        precision_weighted = precision_score(
            y_test, y_test_predictions, average="weighted"
        )
        recall_macro = recall_score(y_test, y_test_predictions, average="macro")
        recall_weighted = recall_score(y_test, y_test_predictions, average="weighted")

        perf_dict = {
            "accuracy": accuracy,
            "auprc_macro": auprc_macro,
            "auprc_weighted": auprc_weighted,
            "auc_ovo": auc_ovo,
            "f1_macro": f1_macro,
            "f1_weighted": f1_weighted,
            "precision_macro": precision_macro,
            "precision_weighted": precision_weighted,
            "recall_macro": recall_macro,
            "recall_weighted": recall_weighted,
            "execution_time": self.execution_time_,
        }

        return perf_dict
