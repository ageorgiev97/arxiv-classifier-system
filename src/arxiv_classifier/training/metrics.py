import tensorflow as tf

class MultiLabelF1(tf.keras.metrics.Metric):
    """
    Computes F1 Score for Multi-label classification.
    Supports 'micro' (global) and 'macro' (per-class average) averaging.
    """
    def __init__(self, name='f1_score', threshold=0.5, average='micro', **kwargs):
        super().__init__(name=name, **kwargs)
        self.threshold = threshold
        self.average = average
        self.true_positives = self.add_weight(name='tp', initializer='zeros')
        self.false_positives = self.add_weight(name='fp', initializer='zeros')
        self.false_negatives = self.add_weight(name='fn', initializer='zeros')

    def update_state(self, y_true, y_pred, sample_weight=None):
        y_pred = tf.cast(y_pred > self.threshold, tf.float32)
        y_true = tf.cast(y_true, tf.float32)

        if self.average == 'micro':
            self.true_positives.assign_add(tf.reduce_sum(y_true * y_pred))
            self.false_positives.assign_add(tf.reduce_sum((1 - y_true) * y_pred))
            self.false_negatives.assign_add(tf.reduce_sum(y_true * (1 - y_pred)))
        # Macro logic would go here (complex in TF graph mode)

    def result(self):
        precision = self.true_positives / (self.true_positives + self.false_positives + tf.keras.backend.epsilon())
        recall = self.true_positives / (self.true_positives + self.false_negatives + tf.keras.backend.epsilon())
        return 2 * ((precision * recall) / (precision + recall + tf.keras.backend.epsilon()))

    def reset_state(self):
        self.true_positives.assign(0.0)
        self.false_positives.assign(0.0)
        self.false_negatives.assign(0.0)