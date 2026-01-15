import wandb
import tensorflow as tf
try:
    from wandb.integration.keras import WandbMetricsLogger
except ImportError:
    # Fallback for older wandb versions
    try:
        from wandb.keras import WandbMetricsLogger
    except ImportError:
        WandbMetricsLogger = None
from ..utils import MultiLabelFocalLoss, F1Score
from ..models import ArxivClassifierBase

class ArxivTrainer:
    def __init__(self, model: ArxivClassifierBase, config):
        self.model = model
        self.config = config # settings.model or wandb.config

    def compile_and_fit(self, train_ds, val_ds, callbacks=None):
        # 1. Use Focal Loss as requested
        loss_fn = MultiLabelFocalLoss(
            gamma=self.config.get('focal_gamma', 2.0),
            alpha=self.config.get('focal_alpha', 0.7)  # Higher alpha for sparse multi-label
        )

        self.model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=self.config['learning_rate']),
            loss=loss_fn,
            metrics=[
                tf.keras.metrics.BinaryAccuracy(name='accuracy'),
                tf.keras.metrics.Precision(name='precision'),
                tf.keras.metrics.Recall(name='recall'),
                F1Score(name='f1_score'),
                tf.keras.metrics.AUC(multi_label=True, name='auc')
            ]
        )

        initial_callbacks = [tf.keras.callbacks.EarlyStopping(patience=3)]
        if callbacks:
            initial_callbacks.extend(callbacks)

        return self.model.fit(
            train_ds,
            validation_data=val_ds,
            epochs=self.config['epochs'],
            callbacks=initial_callbacks
        )