#!/usr/bin/env python
# Code modified from GraphReg

from __future__ import division
import argparse
import yaml
import os

import numpy as np
import tensorflow as tf
from tensorflow.keras.optimizers import Adam


from data import dataset_iterator, read_tf_record_1shot
from models import SeqCNN, poisson_loss


@tf.function
def train_step(
    model,
    optimizer,
    sequences,
    labels,
    loss_obj,
):
    with tf.GradientTape() as tape:
        Y_h3k4me3, Y_h3k27ac, Y_dnase = labels
        Y_hat_h3k4me3, Y_hat_h3k27ac, Y_hat_dnase, _ = model(sequences, training=True)

        # Monitor NaNs
        if tf.math.reduce_any(tf.math.is_nan(Y_hat_h3k4me3)):
            print("NaN detected in H3K4Me3!")
        if tf.math.reduce_any(tf.math.is_nan(Y_hat_h3k27ac)):
            print("NaN detected in H3K27Ac!")
        if tf.math.reduce_any(tf.math.is_nan(Y_hat_dnase)):
            print("NaN detected in DNase!")

        # Compute loss
        loss_h3k4me3 = poisson_loss(Y_h3k4me3, Y_hat_h3k4me3)
        loss_h3k27ac = poisson_loss(Y_h3k27ac, Y_hat_h3k27ac)
        loss_dnase = poisson_loss(Y_dnase, Y_hat_dnase)
        train_loss = (loss_h3k4me3 + loss_h3k27ac + loss_dnase) / 3

    # Backprop
    gradients = tape.gradient(train_loss, model.trainable_variables)
    optimizer.apply_gradients(zip(gradients, model.trainable_variables))

    # Track loss
    loss_obj(train_loss)


@tf.function
def val_step(model, sequences, labels, loss_obj):
    Y_h3k4me3, Y_h3k27ac, Y_dnase = labels
    Y_hat_h3k4me3, Y_hat_h3k27ac, Y_hat_dnase, _ = model(sequences, training=False)

    loss_h3k4me3 = poisson_loss(Y_h3k4me3, Y_hat_h3k4me3)
    loss_h3k27ac = poisson_loss(Y_h3k27ac, Y_hat_h3k27ac)
    loss_dnase = poisson_loss(Y_dnase, Y_hat_dnase)
    val_loss = (loss_h3k4me3 + loss_h3k27ac + loss_dnase) / 3

    # Track loss
    loss_obj(val_loss)


@tf.function
def test_step(model, sequences, labels, loss_obj):

    Y_h3k4me3, Y_h3k27ac, Y_dnase = labels
    Y_hat_h3k4me3, Y_hat_h3k27ac, Y_hat_dnase, _ = model(sequences, training=False)

    loss_h3k4me3 = poisson_loss(Y_h3k4me3, Y_hat_h3k4me3)
    loss_h3k27ac = poisson_loss(Y_h3k27ac, Y_hat_h3k27ac)
    loss_dnase = poisson_loss(Y_dnase, Y_hat_dnase)
    test_loss = (loss_h3k4me3 + loss_h3k27ac + loss_dnase) / 3

    # Track loss
    loss_obj(test_loss)


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

    # Parameters
    F = 4

    # Build model
    model = SeqCNN(dropout=config["dropout"], l2_reg=config["l2_reg"])

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
                    if tf.reduce_sum(tf.gather(tss_idx, idx)) > 0:
                        for jj in range(
                            20, 40, b_size
                        ):  # 20,40 is to take the middle of each window (so as to not predict same data twice)
                            seq_batch = seq[jj : jj + b_size]

                            Y_h3k4me3_batch = Y_h3k4me3[jj : jj + b_size]
                            Y_h3k27ac_batch = Y_h3k27ac[jj : jj + b_size]
                            Y_dnase_batch = Y_dnase[jj : jj + b_size]

                            labels = (Y_h3k4me3_batch, Y_h3k27ac_batch, Y_dnase_batch)
                            train_step(model, optimizer, seq_batch, labels, train_loss)

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

                if data_exist:
                    if tf.reduce_sum(tf.gather(tss_idx, idx)) > 0:
                        for jj in range(
                            20, 40, b_size
                        ):  # 20,40 is to take the middle of each window (so as to not predict same data twice)
                            seq_batch = seq[jj : jj + b_size]

                            Y_h3k4me3_batch = Y_h3k4me3[jj : jj + b_size]
                            Y_h3k27ac_batch = Y_h3k27ac[jj : jj + b_size]
                            Y_dnase_batch = Y_dnase[jj : jj + b_size]

                            labels = (Y_h3k4me3_batch, Y_h3k27ac_batch, Y_dnase_batch)
                            val_step(model, seq_batch, labels, val_loss)

                else:
                    break

        train_loss_epoch = train_loss.result()
        val_loss_epoch = val_loss.result()

        if val_loss_epoch < best_loss:
            early_stopping_counter = 1
            best_loss = val_loss_epoch
            model.save_weights(
                os.path.join(
                    config["save_dir"],
                    args.cell_type,
                    f"{model_file_name}_validloss={val_loss_epoch:0.2f}.h5",
                ),
            )

        else:
            early_stopping_counter += 1
            # TODO: remove this
            model.save_weights(
                os.path.join(
                    config["save_dir"],
                    args.cell_type,
                    f"{config['name']}_{model_file_name}_trainloss={train_loss_epoch:0.2f}.h5",
                ),
            )

            if (
                config["early_stopping"]
                and early_stopping_counter == max_early_stopping
            ):
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
