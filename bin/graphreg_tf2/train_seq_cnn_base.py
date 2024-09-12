#!/usr/bin/env python
# Code modified from GraphReg

from __future__ import division
import argparse
import yaml
import os

import numpy as np
import tensorflow as tf
from tensorflow.keras.optimizers import Adam
from scipy.stats import pearsonr


from data import dataset_iterator, read_tf_record_1shot, parse_proto
from models import SeqCNN, poisson_loss
from metrics import PearsonCorrelation


@tf.function
def train_step(
    model,
    optimizer,
    sequences,
    labels,
    loss_obj,
    pearson_h3k4me3,
    pearson_h3k27ac,
    pearson_dnase,
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
        eps = 1e-20
        loss_h3k4me3 = poisson_loss(Y_h3k4me3 + eps, Y_hat_h3k4me3 + eps)
        loss_h3k27ac = poisson_loss(Y_h3k27ac + eps, Y_hat_h3k27ac + eps)
        loss_dnase = poisson_loss(Y_dnase + eps, Y_hat_dnase + eps)
        loss_total = (loss_h3k4me3 + loss_h3k27ac + loss_dnase) / 3

    # Backprop
    gradients = tape.gradient(loss_total, model.trainable_variables)
    optimizer.apply_gradients(zip(gradients, model.trainable_variables))

    # Track loss
    print(loss_obj.name)
    loss_obj(loss_total)

    # Track pearson
    pearson_h3k4me3.update_state(tf.math.log1p(Y_h3k4me3), tf.math.log1p(Y_hat_h3k4me3))
    pearson_h3k27ac.update_state(tf.math.log1p(Y_h3k4me3), tf.math.log1p(Y_hat_h3k4me3))
    pearson_dnase.update_state(tf.math.log1p(Y_h3k4me3), tf.math.log1p(Y_hat_h3k4me3))


@tf.function
def val_step(
    model, sequences, labels, loss_obj, pearson_h3k4me3, pearson_h3k27ac, pearson_dnase
):
    Y_h3k4me3, Y_h3k27ac, Y_dnase = labels
    Y_hat_h3k4me3, Y_hat_h3k27ac, Y_hat_dnase, _ = model(sequences, training=False)

    eps = 1e-20
    loss_h3k4me3 = poisson_loss(Y_h3k4me3 + eps, Y_hat_h3k4me3 + eps)
    loss_h3k27ac = poisson_loss(Y_h3k27ac + eps, Y_hat_h3k27ac + eps)
    loss_dnase = poisson_loss(Y_dnase + eps, Y_hat_dnase + eps)
    loss_total = (loss_h3k4me3 + loss_h3k27ac + loss_dnase) / 3

    # Track loss
    loss_obj(loss_total)

    # Track pearson
    pearson_h3k4me3.update_state(tf.math.log1p(Y_h3k4me3), tf.math.log1p(Y_hat_h3k4me3))
    pearson_h3k27ac.update_state(tf.math.log1p(Y_h3k4me3), tf.math.log1p(Y_hat_h3k4me3))
    pearson_dnase.update_state(tf.math.log1p(Y_h3k4me3), tf.math.log1p(Y_hat_h3k4me3))


@tf.function
def test_step(
    model, sequences, labels, loss_obj, pearson_h3k4me3, pearson_h3k27ac, pearson_dnase
):

    Y_h3k4me3, Y_h3k27ac, Y_dnase = labels
    Y_hat_h3k4me3, Y_hat_h3k27ac, Y_hat_dnase, _ = model(sequences, training=False)

    eps = 1e-20
    loss_h3k4me3 = poisson_loss(Y_h3k4me3 + eps, Y_hat_h3k4me3 + eps)
    loss_h3k27ac = poisson_loss(Y_h3k27ac + eps, Y_hat_h3k27ac + eps)
    loss_dnase = poisson_loss(Y_dnase + eps, Y_hat_dnase + eps)
    loss_total = (loss_h3k4me3 + loss_h3k27ac + loss_dnase) / 3

    # Track loss
    loss_obj(loss_total)

    # Track pearson
    pearson_h3k4me3.update_state(tf.math.log1p(Y_h3k4me3), tf.math.log1p(Y_hat_h3k4me3))
    pearson_h3k27ac.update_state(tf.math.log1p(Y_h3k4me3), tf.math.log1p(Y_hat_h3k4me3))
    pearson_dnase.update_state(tf.math.log1p(Y_h3k4me3), tf.math.log1p(Y_hat_h3k4me3))


def main():
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

    os.environ["CUDA_VISIBLE_DEVICES"] = str(config["device"])

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

    train_pearson_h3k4me3 = PearsonCorrelation(name="train_pearson_h3k4me3")
    train_pearson_h3k27ac = PearsonCorrelation(name="train_pearson_h3k27ac")
    train_pearson_dnase = PearsonCorrelation(name="train_pearson_dnase")

    val_pearson_h3k4me3 = PearsonCorrelation(name="val_pearson_h3k4me3")
    val_pearson_h3k27ac = PearsonCorrelation(name="val_pearson_h3k27ac")
    val_pearson_dnase = PearsonCorrelation(name="val_pearson_dnase")

    test_pearson_h3k4me3 = PearsonCorrelation(name="test_pearson_h3k4me3")
    test_pearson_h3k27ac = PearsonCorrelation(name="test_pearson_h3k27ac")
    test_pearson_dnase = PearsonCorrelation(name="test_pearson_dnase")

    # Parameters
    F = 4

    # Build model
    model = SeqCNN(dropout=config["dropout"], l2_reg=config["l2_reg"])

    train_filenames = [
        os.path.join(config["data_dir"], args.cell_type, f"chr{i}.tfr")
        for i in train_chr_list
    ]
    val_filenames = [
        os.path.join(config["data_dir"], args.cell_type, f"chr{i}.tfr")
        for i in valid_chr_list
    ]
    train_dataset = (
        tf.data.TFRecordDataset(train_filenames, compression_type="ZLIB")
        .map(parse_proto)
        .batch(1)
        .shuffle(buffer_size=50, reshuffle_each_iteration=True)
    )
    val_dataset = (
        tf.data.TFRecordDataset(val_filenames, compression_type="ZLIB")
        .map(parse_proto)
        .batch(1)
        .shuffle(buffer_size=50, reshuffle_each_iteration=True)
    )

    for epoch in range(1, max_epochs + 1):
        train_loss.reset_state()
        val_loss.reset_state()
        test_loss.reset_state()

        train_pearson_h3k4me3.reset_state()
        train_pearson_h3k27ac.reset_state()
        train_pearson_dnase.reset_state()
        val_pearson_h3k4me3.reset_state()
        val_pearson_h3k27ac.reset_state()
        val_pearson_dnase.reset_state()
        test_pearson_h3k4me3.reset_state()
        test_pearson_h3k27ac.reset_state()
        test_pearson_dnase.reset_state()

        for i, item in enumerate(train_dataset):
            data_exist, seq, Y, adj, Y_h3k4me3, Y_h3k27ac, Y_dnase, idx, tss_idx = (
                read_tf_record_1shot(item)
            )
            print(
                f"train: \t\texample {i}".ljust(80),
                end="\r",
                flush=True,
            )

            if data_exist:
                if tf.reduce_sum(tf.gather(tss_idx, idx)) > 0:

                    # shuffle batches
                    shuffled_idxs = np.arange(20, 40)
                    np.random.shuffle(shuffled_idxs)

                    quotient, remainder = divmod(20, b_size)
                    batch_sizes = [b_size] * quotient
                    if remainder > 0:
                        batch_sizes.append(remainder)

                    j = 0
                    for size in batch_sizes:
                        batch_idxs = shuffled_idxs[j : j + size]
                        j += size

                        # for jj in range(
                        #    20, 40, b_size
                        # ):  # 20,40 is to take the middle of each window (so as to not predict same data twice)
                        seq_batch = tf.gather(seq, batch_idxs)  # [jj : jj + b_size]

                        Y_h3k4me3_batch = tf.gather(
                            Y_h3k4me3, batch_idxs
                        )  # [jj : jj + b_size]
                        Y_h3k27ac_batch = tf.gather(
                            Y_h3k27ac, batch_idxs
                        )  # [jj : jj + b_size]
                        Y_dnase_batch = tf.gather(
                            Y_dnase, batch_idxs
                        )  # [jj : jj + b_size]

                        rc_prob = config["rc_prob"]
                        if rc_prob > 0.0:

                            # Convert to numpy because TF tensors are immutable
                            seq_batch = seq_batch.numpy()
                            Y_h3k4me3_batch = Y_h3k4me3_batch.numpy()
                            Y_h3k27ac_batch = Y_h3k27ac_batch.numpy()
                            Y_dnase_batch = Y_dnase_batch.numpy()

                            revcomp_array = np.random.rand(size) > rc_prob

                            for example_idx in range(size):
                                if revcomp_array[example_idx]:
                                    seq_batch[example_idx] = np.flip(
                                        seq_batch[example_idx], axis=[0, 1]
                                    )
                                    Y_h3k4me3_batch[example_idx] = np.flip(
                                        Y_h3k4me3_batch[example_idx]
                                    )
                                    Y_h3k27ac_batch[example_idx] = np.flip(
                                        Y_h3k27ac_batch[example_idx]
                                    )
                                    Y_dnase_batch[example_idx] = np.flip(
                                        Y_dnase_batch[example_idx]
                                    )

                            # Convert back to TF tensors
                            seq_batch = tf.convert_to_tensor(seq_batch)
                            Y_h3k4me3_batch = tf.convert_to_tensor(Y_h3k4me3_batch)
                            Y_h3k27ac_batch = tf.convert_to_tensor(Y_h3k27ac_batch)
                            Y_dnase_batch = tf.convert_to_tensor(Y_dnase_batch)

                        labels = (Y_h3k4me3_batch, Y_h3k27ac_batch, Y_dnase_batch)
                        train_step(
                            model,
                            optimizer,
                            seq_batch,
                            labels,
                            train_loss,
                            train_pearson_h3k4me3,
                            train_pearson_h3k27ac,
                            train_pearson_dnase,
                        )

            else:
                break

        for i, item in enumerate(val_dataset):
            data_exist, seq, Y, adj, Y_h3k4me3, Y_h3k27ac, Y_dnase, idx, tss_idx = (
                read_tf_record_1shot(item)
            )
            print(
                f"validation: \t\texample {i}".ljust(80),
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
                        val_step(
                            model,
                            seq_batch,
                            labels,
                            val_loss,
                            val_pearson_h3k4me3,
                            val_pearson_h3k27ac,
                            val_pearson_dnase,
                        )

            else:
                break

        train_loss_epoch = train_loss.result()
        val_loss_epoch = val_loss.result()

        train_pearson_h3k4me3_epoch = train_pearson_h3k4me3.result()
        train_pearson_h3k27ac_epoch = train_pearson_h3k27ac.result()
        train_pearson_dnase_epoch = train_pearson_dnase.result()

        val_pearson_h3k4me3_epoch = val_pearson_h3k4me3.result()
        val_pearson_h3k27ac_epoch = val_pearson_h3k27ac.result()
        val_pearson_dnase_epoch = val_pearson_dnase.result()

        if val_loss_epoch < best_loss:
            early_stopping_counter = 1
            best_loss = val_loss_epoch
            model.save_weights(
                os.path.join(
                    config["save_dir"],
                    args.cell_type,
                    f"{config['name']}_{model_file_name}_validloss={val_loss_epoch:0.2f}.h5",
                ),
            )

        else:
            early_stopping_counter += 1
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
            + f"Train Pearson, H3K4Me3: {train_pearson_h3k4me3_epoch:0.2f}\n"
            + f"Train Pearson, H3K27Ac: {train_pearson_h3k27ac_epoch:0.2f}\n"
            + f"Train Pearson, DNase: {train_pearson_dnase_epoch:0.2f}\n"
            + f"Valid Pearson, H3K4Me3: {val_pearson_h3k4me3_epoch:0.2f}\n"
            + f"Valid Pearson, H3K27Ac: {val_pearson_h3k27ac_epoch:0.2f}\n"
            + f"Valid Pearson, DNase: {val_pearson_dnase_epoch:0.2f}\n"
            + "################################\n"
        )

        if not args.no_wandb:
            wandb.log(
                {
                    "train_loss": train_loss_epoch,
                    "train_pearson_h3k4me3": train_pearson_h3k4me3_epoch,
                    "train_pearson_h3k27ac": train_pearson_h3k27ac_epoch,
                    "train_pearson_dnase": train_pearson_dnase_epoch,
                    "val_loss": val_loss_epoch,
                    "val_pearson_h3k4me3": val_pearson_h3k4me3_epoch,
                    "val_pearson_h3k27ac": val_pearson_h3k27ac_epoch,
                    "val_pearson_dnase": val_pearson_dnase_epoch,
                }
            )

    if not args.no_wandb:
        wandb.finish()


################################################################################
# __main__
################################################################################
if __name__ == "__main__":
    main()
