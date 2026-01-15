import tensorflow as tf
import keras


@keras.saving.register_keras_serializable(package="ArxivClassifier")
class MultiLabelFocalLoss(tf.keras.losses.Loss):
    def __init__(self, gamma=2.0, alpha=0.25, name="focal_loss"):
        super().__init__(name=name)
        self.gamma = gamma
        self.alpha = alpha

    def call(self, y_true, y_pred):
        """
        y_true: multi-hot encoded labels
        y_pred: sigmoid probabilities
        """
        y_true = tf.cast(y_true, tf.float32)
        y_pred = tf.clip_by_value(y_pred, tf.keras.backend.epsilon(), 1.0 - tf.keras.backend.epsilon())
        
        # Calculate p_t: p if y=1, 1-p if y=0
        p_t = (y_true * y_pred) + ((1 - y_true) * (1 - y_pred))
        
        # Calculate alpha_t: alpha if y=1, 1-alpha if y=0
        alpha_t = (y_true * self.alpha) + ((1 - y_true) * (1 - self.alpha))
        
        # Calculate Focal Loss
        # FL = -alpha_t * (1 - p_t)^gamma * log(p_t)
        focal_loss = -alpha_t * tf.math.pow((1 - p_t), self.gamma) * tf.math.log(p_t)
        
        return tf.reduce_mean(tf.reduce_sum(focal_loss, axis=1))