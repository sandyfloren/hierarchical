import tensorflow as tf


class PearsonCorrelationMetric(tf.keras.metrics.Metric):
    def __init__(self, name="pearson_correlation", **kwargs):
        super(PearsonCorrelationMetric, self).__init__(name=name, **kwargs)
        # Sum of products of elements
        self.sum_xy = self.add_weight(name="sum_xy", initializer="zeros")
        # Sum of the elements in y
        self.sum_y = self.add_weight(name="sum_y", initializer="zeros")
        # Sum of the elements in x
        self.sum_x = self.add_weight(name="sum_x", initializer="zeros")
        # Sum of squares of elements in x
        self.sum_x2 = self.add_weight(name="sum_x2", initializer="zeros")
        # Sum of squares of elements in y
        self.sum_y2 = self.add_weight(name="sum_y2", initializer="zeros")
        # Count of elements
        self.count = self.add_weight(name="count", initializer="zeros")

    def update_state(self, y_true, y_pred, sample_weight=None):
        y_true = tf.cast(y_true, dtype=self.dtype)
        y_pred = tf.cast(y_pred, dtype=self.dtype)

        self.sum_xy.assign_add(tf.reduce_sum(y_true * y_pred))
        self.sum_y.assign_add(tf.reduce_sum(y_true))
        self.sum_x.assign_add(tf.reduce_sum(y_pred))
        self.sum_x2.assign_add(tf.reduce_sum(tf.square(y_pred)))
        self.sum_y2.assign_add(tf.reduce_sum(tf.square(y_true)))
        self.count.assign_add(tf.cast(tf.size(y_true), self.dtype))

    def result(self):
        numerator = self.count * self.sum_xy - self.sum_x * self.sum_y
        denominator = tf.sqrt(
            (self.count * self.sum_x2 - tf.square(self.sum_x))
            * (self.count * self.sum_y2 - tf.square(self.sum_y))
        )
        # To avoid division by zero
        return numerator / (denominator + tf.keras.backend.epsilon())

    def reset_state(self):
        self.sum_xy.assign(0.0)
        self.sum_y.assign(0.0)
        self.sum_x.assign(0.0)
        self.sum_x2.assign(0.0)
        self.sum_y2.assign(0.0)
        self.count.assign(0.0)
