import tensorflow as tf


class PearsonCorrelation(tf.keras.metrics.Metric):
    def __init__(self, name="pearson_correlation", **kwargs):
        super().__init__(name=name, **kwargs)
        self.covariance = self.add_weight(name="covariance", initializer="zeros")
        self.sum_x = self.add_weight(name="sum_x", initializer="zeros")
        self.sum_y = self.add_weight(name="sum_y", initializer="zeros")
        self.sum_x_squared = self.add_weight(name="sum_x_squared", initializer="zeros")
        self.sum_y_squared = self.add_weight(name="sum_y_squared", initializer="zeros")
        self.count = self.add_weight(name="count", initializer="zeros")

    def update_state(self, y_true, y_pred, sample_weight=None):
        y_true = tf.cast(y_true, self.dtype)
        y_pred = tf.cast(y_pred, self.dtype)
        n = tf.cast(tf.size(y_true), self.dtype)

        sum_x = tf.reduce_sum(y_true)
        sum_y = tf.reduce_sum(y_pred)
        sum_x_squared = tf.reduce_sum(tf.square(y_true))
        sum_y_squared = tf.reduce_sum(tf.square(y_pred))
        covariance = tf.reduce_sum((y_true - sum_x / n) * (y_pred - sum_y / n))

        self.sum_x.assign_add(sum_x)
        self.sum_y.assign_add(sum_y)
        self.sum_x_squared.assign_add(sum_x_squared)
        self.sum_y_squared.assign_add(sum_y_squared)
        self.covariance.assign_add(covariance)
        self.count.assign_add(n)

    def result(self):
        mean_x = self.sum_x / self.count
        mean_y = self.sum_y / self.count
        std_x = tf.sqrt((self.sum_x_squared / self.count) - tf.square(mean_x))
        std_y = tf.sqrt((self.sum_y_squared / self.count) - tf.square(mean_y))
        covariance = self.covariance / self.count

        return covariance / (std_x * std_y)

    def reset_state(self):
        self.covariance.assign(0.0)
        self.sum_x.assign(0.0)
        self.sum_y.assign(0.0)
        self.sum_x_squared.assign(0.0)
        self.sum_y_squared.assign(0.0)
        self.count.assign(0.0)
