import tensorflow as tf
from abc import ABC, abstractmethod

class ArxivClassifierBase(tf.keras.Model, ABC):
    """
    Abstract Base Class to ensure consistency between Baseline and Transformer.
    """
    def __init__(self, num_classes, **kwargs):
        super().__init__(**kwargs)
        self.num_classes = num_classes

    @abstractmethod
    def call(self, inputs, training=False):
        """Standard Keras call method."""
        pass

    @abstractmethod
    def get_model_type(self) -> str:
        """Returns 'baseline' or 'SciBert' or 'specter' for logging logic."""
        pass

    def get_config(self):
        """Required for Keras model serialization."""
        config = super().get_config()
        config.update({
            "num_classes": self.num_classes,
        })
        return config