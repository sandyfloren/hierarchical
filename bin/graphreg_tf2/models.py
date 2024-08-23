import tensorflow as tf
from tensorflow.keras.layers import (
    Layer,
    Dropout,
    Conv1D,
    BatchNormalization,
    MaxPool1D,
    Reshape,
    LeakyReLU,
)
from tensorflow.keras import (
    Model,
    Sequential,
    activations,
    constraints,
    initializers,
    regularizers,
)
from tensorflow.keras.regularizers import l2
from tensorflow.keras import backend as K


def poisson_loss(y_true, mu_pred, eps=1e-20):
    nll = tf.reduce_mean(
        tf.math.lgamma(y_true + 1)
        + mu_pred
        - y_true
        * tf.math.log(
            # tf.clip_by_value(
            mu_pred,  # clip_value_min=eps, clip_value_max=tf.float32.max )
        )
    )
    return nll


class SeqCNN(Model):
    def __init__(self, dropout=0.5, l2_reg=0.0):
        super().__init__()
        self.l2_reg = l2_reg
        self.dropout = Dropout(dropout)
        self.conv_tower = Sequential(
            [
                self.ConvBlock(256, 21),
                BatchNormalization(),
                MaxPool1D(2),
                self.dropout,
                self.ConvBlock(128, 3),
                BatchNormalization(),
                MaxPool1D(2),
                self.dropout,
                self.ConvBlock(128, 3),
                BatchNormalization(),
                MaxPool1D(5),
                self.dropout,
                self.ConvBlock(128, 3),
                BatchNormalization(),
                MaxPool1D(5),
                self.dropout,
                self.ConvBlock(64, 3),
                BatchNormalization(),
            ]
        )
        self.dilatedconv1 = self.DilatedConvBlock(64, 3, 2**1)
        self.dilatedconv2 = self.DilatedConvBlock(64, 3, 2**2)
        self.dilatedconv3 = self.DilatedConvBlock(64, 3, 2**3)
        self.dilatedconv4 = self.DilatedConvBlock(64, 3, 2**4)
        self.dilatedconv5 = self.DilatedConvBlock(64, 3, 2**5)
        self.dilatedconv6 = self.DilatedConvBlock(64, 3, 2**6)

        self.bn1 = BatchNormalization()
        self.bn2 = BatchNormalization()
        self.bn3 = BatchNormalization()
        self.bn4 = BatchNormalization()
        self.bn5 = BatchNormalization()
        self.bn6 = BatchNormalization()

        self.conv_h3k4me3 = Sequential(
            [Conv1D(1, 5, activation="exponential", padding="same"), Reshape([1000])]
        )
        self.conv_h3k27ac = Sequential(
            [Conv1D(1, 5, activation="exponential", padding="same"), Reshape([1000])]
        )
        self.conv_dnase = Sequential(
            [Conv1D(1, 5, activation="exponential", padding="same"), Reshape([1000])]
        )
        self.reshape = Reshape([1000])

    def ConvBlock(self, filters, kernel_width):
        return Conv1D(
            filters,
            kernel_width,
            activation="relu",
            padding="same",
            kernel_regularizer=l2(self.l2_reg),
            bias_regularizer=l2(self.l2_reg),
        )

    def DilatedConvBlock(self, filters, kernel_width, dilation_rate):
        return Conv1D(
            filters,
            kernel_width,
            activation="relu",
            dilation_rate=dilation_rate,
            padding="same",
            kernel_regularizer=l2(self.l2_reg),
            bias_regularizer=l2(self.l2_reg),
        )

    def call(self, x):

        # Apply convolutions
        x = self.conv_tower(x)

        # Save bottleneck representations
        h = x

        # Apply dilated residual convolutions
        x = self.dropout(x)
        x = self.dilatedconv1(x) + x
        x = self.bn1(x)

        x = self.dropout(x)
        x = self.dilatedconv2(x) + x
        x = self.bn2(x)

        x = self.dropout(x)
        x = self.dilatedconv3(x) + x
        x = self.bn3(x)

        x = self.dropout(x)
        x = self.dilatedconv4(x) + x
        x = self.bn4(x)

        x = self.dropout(x)
        x = self.dilatedconv5(x) + x
        x = self.bn5(x)

        x = self.dropout(x)
        x = self.dilatedconv6(x) + x
        x = self.bn6(x)

        # Apply separate head for each assay
        mu_h3k4me3 = self.conv_h3k4me3(x)
        mu_h3k27ac = self.conv_h3k27ac(x)
        mu_dnase = self.conv_dnase(x)

        return mu_h3k4me3, mu_h3k27ac, mu_dnase, h


class GraphReg(Model):
    def __init__(self, dropout=0.5, l2_reg=0.0, N=1200, F_=32, n_attn_heads=4):
        super().__init__()
        self.l2_reg = l2_reg
        self.dropout_rate = dropout
        self.dropout = Dropout(self.dropout_rate)
        self.F_ = F_
        self.n_attn_heads = n_attn_heads
        self.conv_tower = Sequential(
            [
                self.ConvBlock(128, 3),
                BatchNormalization(),
                MaxPool1D(2),
                self.dropout,
                self.ConvBlock(128, 3),
                BatchNormalization(),
                MaxPool1D(5),
                self.dropout,
                self.ConvBlock(128, 3),
                BatchNormalization(),
                MaxPool1D(5),
            ]
        )
        self.gat1 = self.GATBlock()
        self.gat2 = self.GATBlock()
        self.gat3 = self.GATBlock()

        self.batchnorm1 = BatchNormalization()
        self.batchnorm2 = BatchNormalization()
        self.batchnorm3 = BatchNormalization()

        self.final_conv = Sequential(
            [
                self.dropout,
                self.ConvBlock(64, 1),
                BatchNormalization(),
                self.ConvBlock(1, 1, activation="exponential"),
                Reshape([N]),
            ]
        )

    def ConvBlock(self, filters, kernel_width, activation="relu"):
        return Conv1D(
            filters,
            kernel_width,
            activation=activation,
            padding="same",
            kernel_regularizer=l2(self.l2_reg),
            bias_regularizer=l2(self.l2_reg),
        )

    def GATBlock(self):
        return GraphAttention(
            self.F_,
            attn_heads=self.n_attn_heads,
            attn_heads_reduction="concat",
            dropout_rate=self.dropout_rate,
            activation="elu",
            kernel_regularizer=l2(self.l2_reg),
            attn_kernel_regularizer=l2(self.l2_reg),
        )

    def call(self, inputs):

        x, A_in = inputs

        x = self.conv_tower(x)  # (B x N x F) F=128

        x, att1 = self.gat1([x, A_in])
        x = self.batchnorm1(x)  # (B x N x 4F') F'=32, 4F'=128

        x, att2 = self.gat2([x, A_in])
        x = self.batchnorm2(x)  # (B x N x 4F') F'=32, 4F'=128

        x, att3 = self.gat3([x, A_in])
        x = self.batchnorm3(x)  # (B x N x 4F') F'=32, 4F'=128

        x = self.final_conv(x)  # (B x N x 64) -> (B x N x 1) -> (B x N)

        return x, [att1, att2, att3]


class GraphAttention(Layer):

    def __init__(
        self,
        F_,
        attn_heads=1,
        attn_heads_reduction="concat",  # {'concat', 'average'}
        dropout_rate=0.0,
        activation="relu",
        use_bias=False,
        kernel_initializer="glorot_uniform",
        bias_initializer="zeros",
        attn_kernel_initializer="glorot_uniform",
        kernel_regularizer=None,
        bias_regularizer=None,
        attn_kernel_regularizer=None,
        activity_regularizer=None,
        kernel_constraint=None,
        bias_constraint=None,
        attn_kernel_constraint=None,
        **kwargs
    ):
        if attn_heads_reduction not in {"concat", "average"}:
            raise ValueError("Possbile reduction methods: concat, average")

        self.F_ = F_  # Number of output features
        self.attn_heads = attn_heads  # Number of attention heads
        self.attn_heads_reduction = attn_heads_reduction
        self.dropout_rate = dropout_rate  # Internal dropout rate
        self.activation = activations.get(activation)
        self.use_bias = use_bias

        self.kernel_initializer = initializers.get(kernel_initializer)
        self.bias_initializer = initializers.get(bias_initializer)
        self.attn_kernel_initializer = initializers.get(attn_kernel_initializer)

        self.kernel_regularizer = regularizers.get(kernel_regularizer)
        self.bias_regularizer = regularizers.get(bias_regularizer)
        self.attn_kernel_regularizer = regularizers.get(attn_kernel_regularizer)
        self.activity_regularizer = regularizers.get(activity_regularizer)

        self.kernel_constraint = constraints.get(kernel_constraint)
        self.bias_constraint = constraints.get(bias_constraint)
        self.attn_kernel_constraint = constraints.get(attn_kernel_constraint)
        self.supports_masking = False

        # Populated by build()
        self.kernels = []  # Layer kernels for attention heads
        self.biases = []  # Layer biases for attention heads
        self.attn_kernels = []  # Attention kernels for attention heads

        if attn_heads_reduction == "concat":
            # Output will have shape (..., K * F')
            self.output_dim = self.F_ * self.attn_heads
        else:
            # Output will have shape (..., F')
            self.output_dim = self.F_

        super(GraphAttention, self).__init__(**kwargs)

    def build(self, input_shape):
        assert len(input_shape) >= 2
        F = input_shape[0][-1]

        # Initialize weights for each attention head
        for head in range(self.attn_heads):
            # Layer kernel
            kernel_self = self.add_weight(
                shape=(F, self.F_),
                initializer=self.kernel_initializer,
                regularizer=self.kernel_regularizer,
                constraint=self.kernel_constraint,
                trainable=True,
                name="kernel_self_{}".format(head),
            )
            kernel_neighs = self.add_weight(
                shape=(F, self.F_),
                initializer=self.kernel_initializer,
                regularizer=self.kernel_regularizer,
                constraint=self.kernel_constraint,
                trainable=True,
                name="kernel_neighs_{}".format(head),
            )
            self.kernels.append([kernel_self, kernel_neighs])

            # Layer bias
            if self.use_bias:
                bias = self.add_weight(
                    shape=(self.F_,),
                    initializer=self.bias_initializer,
                    regularizer=self.bias_regularizer,
                    constraint=self.bias_constraint,
                    trainable=True,
                    name="bias_{}".format(head),
                )
                self.biases.append(bias)

            # Attention kernels
            attn_kernel_self = self.add_weight(
                shape=(self.F_, 1),
                initializer=self.attn_kernel_initializer,
                regularizer=self.attn_kernel_regularizer,
                constraint=self.attn_kernel_constraint,
                trainable=True,
                name="attn_kernel_self_{}".format(head),
            )
            attn_kernel_neighs = self.add_weight(
                shape=(self.F_, 1),
                initializer=self.attn_kernel_initializer,
                regularizer=self.attn_kernel_regularizer,
                constraint=self.attn_kernel_constraint,
                trainable=True,
                name="attn_kernel_neigh_{}".format(head),
            )
            self.attn_kernels.append([attn_kernel_self, attn_kernel_neighs])

        self.built = True

    def call(self, inputs):
        X = inputs[0]  # Node features (B x N x F)
        A = inputs[1]  # Adjacency matrix (B x N x N)

        outputs = []
        Att = []
        for head in range(self.attn_heads):
            kernel_self = self.kernels[head][0]  # W in the paper (F x F')
            kernel_neighs = self.kernels[head][1]
            attention_kernel = self.attn_kernels[
                head
            ]  # Attention kernel a in the paper (2F' x 1)

            # Compute inputs to attention network
            features_self = K.dot(X, kernel_self)  # (B x N x F')
            features_neighs = K.dot(X, kernel_neighs)

            # Compute feature combinations
            # Note: [[a_1], [a_2]]^T [[Wh_i], [Wh_2]] = [a_1]^T [Wh_i] + [a_2]^T [Wh_j]
            attn_for_self = K.dot(
                features_self, attention_kernel[0]
            )  # (B x N x 1), [a_1]^T [Wh_i]
            attn_for_neighs = K.dot(
                features_neighs, attention_kernel[1]
            )  # (B x N x 1), [a_2]^T [Wh_j]

            # Attention head a(Wh_i, Wh_j) = a^T [[Wh_i], [Wh_j]]
            attn_for_self_permute = K.permute_dimensions(
                attn_for_self, (1, 0, 2)
            )  # (N x B x 1)
            attn_for_neighs_permute = K.permute_dimensions(
                attn_for_neighs, (1, 0, 2)
            )  # (N x B x 1)
            att = attn_for_self_permute + K.transpose(
                attn_for_neighs_permute
            )  # (N x B x N) via broadcasting
            att = K.permute_dimensions(att, (1, 0, 2))  # (B x N x N)

            # Add nonlinearty
            att = LeakyReLU(alpha=0.2)(att)

            # Mask values before activation (Vaswani et al., 2017)
            mask = -10e15 * (1.0 - A)
            att += mask

            # Apply sigmoid to get attention coefficients
            att = K.sigmoid(att)
            att_sum = K.sum(att, axis=-1, keepdims=True)
            att = att / (1 + att_sum)
            beta_promoter = 1 / (1 + att_sum)

            Att.append(att)

            # Apply dropout to features and attention coefficients
            # dropout_attn = Dropout(self.dropout_rate)(att)                    # (B x N x N)
            dropout_feat_neigh = Dropout(self.dropout_rate)(
                features_neighs
            )  # (B x N x F')
            dropout_feat_self = Dropout(self.dropout_rate)(
                features_self
            )  # (B x N x F')

            # Linear combination with neighbors' features
            node_features = dropout_feat_self * beta_promoter + K.batch_dot(
                att, dropout_feat_neigh
            )  # (B x N x F')

            if self.use_bias:
                node_features = K.bias_add(node_features, self.biases[head])

            # Add output of attention head to final output
            outputs.append(node_features)

        # Aggregate the heads' output according to the reduction method
        if self.attn_heads_reduction == "concat":
            output = K.concatenate(outputs)  # (B x N x KF')
        else:
            output = K.mean(K.stack(outputs), axis=0)  # (B x N x F')

        output = self.activation(output)

        return output, Att

    def compute_output_shape(self, input_shape):
        output_shape = input_shape[0][0], self.output_dim
        return output_shape

    def get_config(self):
        config = super().get_config().copy()
        config.update(
            {
                "F_": self.F_,
                "attn_heads": self.attn_heads,
                "attn_heads_reduction": self.attn_heads_reduction,
                "dropout_rate": self.dropout_rate,
                "activation": self.activation,
                "use_bias": self.use_bias,
                "kernel_initializer": self.kernel_initializer,
                "bias_initializer": self.bias_initializer,
                "attn_kernel_initializer": self.attn_kernel_initializer,
                "kernel_regularizer": self.kernel_regularizer,
                "bias_regularizer": self.bias_regularizer,
                "attn_kernel_regularizer": self.attn_kernel_regularizer,
                "activity_regularizer": self.activity_regularizer,
                "kernel_constraint": self.kernel_constraint,
                "bias_constraint": self.bias_constraint,
                "attn_kernel_constraint": self.attn_kernel_constraint,
            }
        )
        return config
