# stdlib
from typing import Union

# third party
import numpy as np
from pydantic import BaseModel, Field
import torch

"""
Questions:
Are the _true_probabilities (now `predicted_label_probabilities`) used for anything?
Is there a need to store them? 

Is the `detach` causing errors during training?
"""


class EpochProbabilities(BaseModel):
    """Dataclass to store the probabilities of the ground truth label and the
    predicted label for each sample for multiple epochs.
    """

    ground_truth_probabilities: np.ndarray = Field(
        description="""Probability of the ground truth label for each sample. 
        np.ndarray of shape (n_samples, n_epochs)""",
    )
    predicted_label_probabilities: np.ndarray = Field(
        description="""Probability of the predicted label for each sample. 
        np.adarray of shape (n_samples, n_epochs)""",
    )

    class Config:
        arbitrary_types_allowed = True


class BatchProbabilities(BaseModel):
    """Dataclass to store the probabilities of the ground truth label and the
    predicted label for each sample for a single epoch."""

    ground_truth_probabilities: np.ndarray = Field(
        description="""Probability of the ground truth label for each sample. 
        np.ndarray of shape (n_samples,)""",
    )
    predicted_label_probabilities: np.ndarray = Field(
        description="""Probability of the predicted label for each sample. 
        np.ndarray of shape (n_samples,)""",
    )

    class Config:
        arbitrary_types_allowed = True


class DataIQTorch:
    def __init__(self, n_samples: int):
        self.n_samples = n_samples
        self.current_sample_idx = 0

        self.probs = None
        self._epoch_probs = BatchProbabilities(
            ground_truth_probabilities=np.zeros(n_samples),
            predicted_label_probabilities=np.zeros(n_samples),
        )

    def on_batch_end(
        self,
        y_true: Union[np.ndarray, torch.Tensor],
        y_pred: Union[np.ndarray, torch.Tensor],
    ) -> None:
        """Add the probabilities of the ground truth label and the predicted
        label for each sample. For each batch in an epoch, the probabilities
        are added to the _epoch_probs attribute. At the end of the epoch, the
        probabilities are added to the probs attribute.

        Args:
            y_true (np.ndarray): Ground truth labels. np.ndarray of shape
                (batch_size,)
            y_pred (np.ndarray): Predicted labels. np.ndarray of shape
                (batch_size, n_classes)
        """
        batch_size = y_true.shape[0]

        y_true, y_pred = self._convert_to_numpy(y_true), self._convert_to_numpy(y_pred)
        # get the probabilities of the ground truth label
        ground_truth_probs = self._get_ground_truth_probs(y_true=y_true, y_pred=y_pred)

        # get the probabilities of the predicted label
        predicted_label_probs = self._get_predicted_label_probs(y_pred=y_pred)

        # add the probabilities to the _epoch_probs attribute for the current batch
        self._epoch_probs.ground_truth_probabilities[
            self.current_sample_idx : self.current_sample_idx + batch_size
        ] = ground_truth_probs
        self._epoch_probs.predicted_label_probabilities[
            self.current_sample_idx : self.current_sample_idx + batch_size
        ] = predicted_label_probs

        # update the current sample index
        self.current_sample_idx += batch_size
        # if the current sample index is equal to the number of samples, then
        # the epoch is over and the probabilities for the epoch are added to the
        # probs attribute
        if self.current_sample_idx == self.n_samples:
            self._add_epoch_probs()
            self.current_sample_idx = 0

    def _get_ground_truth_probs(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
    ) -> np.ndarray:
        """Get the probabilities of the ground truth label for each sample in the batch.

        Args:
            y_true (np.ndarray): Ground truth labels. np.ndarray of shape (batch_size,)
            y_pred (np.ndarray): Predicted labels. np.ndarray of shape (batch_size, n_classes)

        Returns:
            np.ndarray: Probabilities of the ground truth label for each sample.
            np.ndarray of shape (batch_size,)
        """
        return y_pred[np.arange(y_pred.shape[0]), y_true]

    def _get_predicted_label_probs(self, y_pred: np.ndarray) -> np.ndarray:
        """Get the probabilities of the predicted label for each sample in the
        batch.

        Args:
            y_pred (np.ndarray): Predicted labels. np.ndarray of shape
                (batch_size, n_classes)

        Returns:
            np.ndarray: Probabilities of the predicted label for each sample.
            np.ndarray of shape (batch_size,)
        """
        return np.max(y_pred, axis=1)

    def _add_epoch_probs(self) -> None:
        """Add the probabilities for the current epoch to the probs attribute."""
        if self.probs is None:
            self.probs = EpochProbabilities(
                ground_truth_probabilities=self._epoch_probs.ground_truth_probabilities[
                    :, None
                ],
                predicted_label_probabilities=self._epoch_probs.predicted_label_probabilities[
                    :, None
                ],
            )
        else:
            self.probs.ground_truth_probabilities = np.hstack(
                (
                    self.probs.ground_truth_probabilities,
                    self._epoch_probs.ground_truth_probabilities[:, None],
                )
            )
            self.probs.predicted_label_probabilities = np.hstack(
                (
                    self.probs.predicted_label_probabilities,
                    self._epoch_probs.predicted_label_probabilities[:, None],
                )
            )

    def _convert_to_numpy(self, x: Union[np.ndarray, torch.Tensor]) -> np.ndarray:
        """Convert a torch tensor to a numpy array.

        Args:
            x (Union[np.ndarray, torch.Tensor]): Array to be converted.

        Returns:
            np.ndarray: Converted array.
        """
        return x.detach().numpy() if isinstance(x, torch.Tensor) else x

    def aleatoric_uncertainty(self) -> np.ndarray:
        """Compute the aleatoric uncertainty of the ground truth label
        probability across epochs

        Returns:
            np.ndarray: Aleatoric uncertainty. np.ndarray of shape (n_samples,)
        """
        preds = self.probs.ground_truth_probabilities
        return np.mean(preds * (1 - preds), axis=1)
