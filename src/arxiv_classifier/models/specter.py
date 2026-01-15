import tensorflow as tf
from .base import ArxivClassifierBase
from transformers import TFAutoModel
import keras


@keras.saving.register_keras_serializable(package="arxiv_classifier")
class SpecterClassifier(ArxivClassifierBase):
    def __init__(self, num_classes, model_name="allenai/specter", dropout_rate=0.1, **kwargs):
        super().__init__(num_classes, **kwargs)
        self.model_name = model_name
        self.dropout_rate = dropout_rate
        self.bert = TFAutoModel.from_pretrained(model_name)
        self.dropout = tf.keras.layers.Dropout(dropout_rate)
        # Initialize with negative bias so sigmoid starts near 0 (sparse labels)
        self.classifier = tf.keras.layers.Dense(
            num_classes, 
            activation="sigmoid",
            bias_initializer=tf.keras.initializers.Constant(-1.0)  # sigmoid(-1) ≈ 0.27
        )

    def call(self, inputs, training=False):
        # In TF graph mode, inputs is a dict-like KerasTensor, not a Python dict
        # Always try to access as dict first since that's how tf.data provides it
        try:
            input_ids = inputs['input_ids']
            attention_mask = inputs['attention_mask']
            outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask, training=training)
        except (TypeError, KeyError):
            # Fallback for direct tensor input
            outputs = self.bert(inputs, training=training)
        # Specter usually uses the [CLS] token (pooler_output) or mean pooling. 
        # Using pooler_output as it's standard for sentence embeddings in Specter 1.0
        x = self.dropout(outputs.pooler_output, training=training)
        return self.classifier(x)

    def get_model_type(self): 
        return "specter"

    def get_config(self):
        config = super().get_config()
        config.update({
            "model_name": self.model_name,
            "dropout_rate": self.dropout_rate,
        })
        return config

    @classmethod
    def from_config(cls, config):
        return cls(**config)
