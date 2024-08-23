#!/usr/bin/env python
# Code modified from GraphReg

from __future__ import division
import argparse
import yaml
import os

import numpy as np
import tensorflow as tf
from tensorflow.keras.optimizers import Adam
from tensorflow.keras import backend as K


from data import dataset_iterator, read_tf_record_1shot
from models import SeqCNN, GraphReg, poisson_loss
from metrics import PearsonCorrelationMetric


@tf.function
def train_step(model, optimizer, inputs, labels, loss_obj, metric_obj):
    with tf.GradientTape() as tape:
        Y_cage = labels
        x_in_gat, adj = inputs
        # idx is range(400, 800)
        Y_hat_cage, _ = model([x_in_gat, adj], training=True)
        Y_hat_cage_idx = Y_hat_cage[
            400:800
        ]  # tf.gather(Y_hat_cage, idx, axis=1) # selects middle 400 CAGE values
        Y_cage_idx = Y_cage[
            400:800
        ]  # tf.gather(Y_cage, idx, axis=1) # selects middle 400 CAGE values

        # Monitor NaNs
        if tf.math.reduce_any(tf.math.is_nan(Y_hat_cage)):
            print("NaN detected in CAGE!")

        # Compute loss
        train_loss = poisson_loss(Y_cage_idx, Y_hat_cage_idx)

    # Backprop
    gradients = tape.gradient(train_loss, model.trainable_variables)
    optimizer.apply_gradients(zip(gradients, model.trainable_variables))

    # Track loss and metric
    loss_obj(train_loss)
    metric_obj(Y_hat_cage, Y_cage)


@tf.function
def val_step(model, inputs, labels, loss_obj, metric_obj):
    Y_cage = labels
    x_in_gat, adj = inputs
    Y_hat_cage, _ = model([x_in_gat, adj], training=True)
    Y_hat_cage_idx = Y_hat_cage[400:800]  # tf.gather(Y_hat_cage, idx, axis=1)
    Y_cage_idx = Y_cage[400:800]  # tf.gather(Y_cage, idx, axis=1)

    # Compute loss
    val_loss = poisson_loss(Y_cage_idx, Y_hat_cage_idx)

    # Track loss and metric
    loss_obj(val_loss)
    metric_obj(Y_hat_cage, Y_cage)


@tf.function
def test_step(model, inputs, labels, loss_obj, metric_obj):
    Y_cage = labels
    x_in_gat, adj = inputs
    Y_hat_cage, _ = model([x_in_gat, adj], training=True)
    Y_hat_cage_idx = Y_hat_cage[400:800]  # tf.gather(Y_hat_cage, idx, axis=1)
    Y_cage_idx = Y_cage[400:800]  # tf.gather(Y_cage, idx, axis=1)

    # Compute loss
    test_loss = poisson_loss(Y_cage_idx, Y_hat_cage_idx)

    # Track loss and metric
    loss_obj(test_loss)
    metric_obj(Y_hat_cage, Y_cage)


def main():
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    # tf.debugging.enable_check_numerics()
    parser = argparse.ArgumentParser()
    parser.add_argument("config", type=str, help="The config to use.")
    parser.add_argument(
        "-v", "--val_chr", default="1,11", type=str, help="Validation chromosomes."
    )
    parser.add_argument(
        "-t", "--test_chr", default="2,12", type=str, help="Test chromosomes."
    )
    parser.add_argument(
        "-c", "--cell_type", default="k562", type=str, help="Cell type."
    )
    parser.add_argument(
        "--no_wandb",
        default=False,
        action="store_true",
        help="Flag to skip logging to Weights and Biases.",
    )
    parser.add_argument(
        "-d",
        "--deterministic",
        default=False,
        action="store_true",
        help="Flag to require deterministic ops.",
    )
    args = parser.parse_args()

    with open(args.config, "r") as infile:
        config = yaml.safe_load(infile)

    assert os.path.exists(os.path.join(config["data_dir"], args.cell_type))
    assert os.path.exists(os.path.join(config["save_dir"], args.cell_type))

    tf.keras.utils.set_random_seed(config["seed"])
    if args.deterministic:
        tf.config.experimental.enable_op_determinism()

    valid_chr_list = [int(c) for c in args.val_chr.split(",")]
    test_chr_list = [int(c) for c in args.test_chr.split(",")]
    train_chr_list = [
        c for c in range(1, 1 + 22) if c not in valid_chr_list + test_chr_list
    ]
    model_file_name = f"val_{'_'.join(str(c) for c in valid_chr_list)}_test_{'_'.join(str(c) for c in test_chr_list)}"

    if not args.no_wandb:
        import wandb

        wandb.init(
            job_type="train", project="graphreg", name=config["name"], config=config
        )

    ########## training ##########

    best_loss = 1e20
    early_stopping_counter = 1
    max_epochs = config["max_epochs"]
    max_early_stopping = config["patience"]
    optimizer = Adam(
        learning_rate=config["learning_rate"], weight_decay=config["weight_decay"]
    )
    b_size = config["batch_size"]

    train_loss = tf.keras.metrics.Mean(name="train_loss")
    val_loss = tf.keras.metrics.Mean(name="val_loss")
    test_loss = tf.keras.metrics.Mean(name="test_loss")

    train_pearson = PearsonCorrelationMetric(name="train_pearson")
    val_pearson = PearsonCorrelationMetric(name="val_pearson")
    test_pearson = PearsonCorrelationMetric(name="test_pearson")

    # Parameters
    F = 4

    # Build models
    model_cnn_base = SeqCNN(dropout=config["dropout"], l2_reg=config["l2_reg"])
    model_cnn_base.load_weights(config["seq_cnn_base_model"])
    model_cnn_base._name = "Seq-CNN_base"
    model_cnn_base.trainable = False

    model_gat = GraphReg(dropout=config["dropout"], l2_reg=config["l2_reg"])
    model_gat._name = "Seq-GraphReg"

    for epoch in range(1, max_epochs + 1):
        train_loss.reset_state()
        val_loss.reset_state()
        test_loss.reset_state()

        for i in train_chr_list:
            file_name_train = os.path.join(
                config["data_dir"], args.cell_type, f"chr{i}.tfr"
            )
            iterator_train = dataset_iterator(file_name_train, batch_size=1)
            for item_idx, item in enumerate(iterator_train):
                data_exist, seq, Y, adj, Y_h3k4me3, Y_h3k27ac, Y_dnase, idx, tss_idx = (
                    read_tf_record_1shot(item)
                )
                print(
                    f"train chr: {i}\t\texample: {item_idx}".ljust(80),
                    end="\r",
                    flush=True,
                )

                if data_exist:
                    if tf.reduce_sum(tf.gather(tss_idx, tf.range(400, 800))) > 0:
                        for jj in range(0, 60, 10):
                            seq_batch = seq[jj : jj + 10]
                            _, _, _, h = model_cnn_base(seq_batch)
                            H.append(h)

                        x_in_gat = K.concatenate(H, axis=0)
                        x_in_gat = K.reshape(x_in_gat, [1, 60000, 64])

                        inputs = (x_in_gat, adj)
                        labels = Y
                        train_step(
                            model_gat,
                            optimizer,
                            inputs,
                            labels,
                            train_loss,
                            train_pearson,
                        )

                else:
                    break

        for i in valid_chr_list:

            file_name_val = os.path.join(
                config["data_dir"], args.cell_type, f"chr{i}.tfr"
            )
            iterator_val = dataset_iterator(file_name_val, batch_size=1)
            for item_idx, item in enumerate(iterator_val):
                data_exist, seq, Y, adj, Y_h3k4me3, Y_h3k27ac, Y_dnase, idx, tss_idx = (
                    read_tf_record_1shot(item)
                )
                print(
                    f"valid chr: {i}\t\texample: {item_idx}".ljust(80),
                    end="\r",
                    flush=True,
                )
                H = []
                if data_exist:
                    if tf.reduce_sum(tf.gather(tss_idx, tf.range(400, 800))) > 0:
                        for jj in range(0, 60, 10):
                            seq_batch = seq[jj : jj + 10]
                            _, _, _, h = model_cnn_base(seq_batch)
                            H.append(h)

                        x_in_gat = K.concatenate(H, axis=0)
                        x_in_gat = K.reshape(x_in_gat, [1, 60000, 64])

                        inputs = (x_in_gat, adj)
                        labels = Y
                        val_step(model_gat, inputs, labels, val_loss, val_pearson)

                else:
                    break
        train_loss_epoch = train_loss.result()
        val_loss_epoch = val_loss.result()

        if val_loss_epoch < best_loss:
            early_stopping_counter = 1
            best_loss = val_loss_epoch
            model_gat.save_weights(
                os.path.join(
                    config["save_dir"],
                    args.cell_type,
                    f"{config['name']}_{model_file_name}_validloss={val_loss_epoch:0.2f}",
                ),
                save_format="h5",
            )

        else:
            early_stopping_counter += 1
            if early_stopping_counter == max_early_stopping:
                print(f"Stopping early on epoch {epoch}")
                break

        print(
            "\n################################\n"
            + f"Epoch {epoch}\n"
            + f"Loss: {train_loss_epoch:0.2f}\n"
            + f"Validation Loss: {val_loss_epoch:0.2f}\n"
            + "################################\n"
        )

        if not args.no_wandb:
            wandb.log({"train_loss": train_loss_epoch, "val_loss": val_loss_epoch})

    if not args.no_wandb:
        wandb.finish()


################################################################################
# __main__
################################################################################
if __name__ == "__main__":
    main()
