import tensorflow as tf


def parse_proto(example_protos):
    features = {
        "last_batch": tf.io.FixedLenFeature([1], tf.int64),
        "adj": tf.io.FixedLenFeature([], tf.string),
        #'adj_real': tf.io.FixedLenFeature([], tf.string),
        "tss_idx": tf.io.FixedLenFeature([], tf.string),
        "X_1d": tf.io.FixedLenFeature([], tf.string),
        "Y": tf.io.FixedLenFeature([], tf.string),
        "sequence": tf.io.FixedLenFeature([], tf.string),
    }
    parsed_features = tf.io.parse_example(example_protos, features=features)
    last_batch = parsed_features["last_batch"]
    adj = tf.io.decode_raw(parsed_features["adj"], tf.float16)
    adj = tf.cast(adj, tf.float32)
    # adj_real = tf.io.decode_raw(parsed_features['adj_real'], tf.float16)
    # adj_real = tf.cast(adj_real, tf.float32)
    tss_idx = tf.io.decode_raw(parsed_features["tss_idx"], tf.float16)
    tss_idx = tf.cast(tss_idx, tf.float32)
    X_epi = tf.io.decode_raw(parsed_features["X_1d"], tf.float16)
    X_epi = tf.cast(X_epi, tf.float32)
    Y = tf.io.decode_raw(parsed_features["Y"], tf.float16)
    Y = tf.cast(Y, tf.float32)
    seq = tf.io.decode_raw(parsed_features["sequence"], tf.float64)
    seq = tf.cast(seq, tf.float32)
    return {
        "seq": seq,
        "last_batch": last_batch,
        "X_epi": X_epi,
        "Y": Y,
        "adj": adj,
        "tss_idx": tss_idx,
    }


def file_to_records(filename):
    return tf.data.TFRecordDataset(filename, compression_type="ZLIB")


def dataset_iterator(file_name, batch_size):
    dataset = tf.data.Dataset.list_files(file_name)
    dataset = dataset.flat_map(file_to_records)
    dataset = dataset.batch(batch_size)
    dataset = dataset.map(parse_proto)
    # iterator = tf.compat.v1.data.make_one_shot_iterator(dataset)
    return dataset  # iterator


def read_tf_record_1shot(item):
    try:
        next_datum = item
        data_exist = True
    except tf.errors.OutOfRangeError:
        data_exist = False
    if data_exist:
        T = 400  # number of 5kb bins inside middle 2Mb region
        b = 50  # number of 100bp bins inside 5Kb region
        F = 4  # number of ACGT (4)
        seq = next_datum["seq"]
        batch_size = tf.shape(seq)[0]

        seq = tf.reshape(seq, [60, 100000, F])
        adj = next_datum["adj"]
        adj = tf.reshape(adj, [batch_size, 3 * T, 3 * T])
        # last_batch = next_datum["last_batch"]
        tss_idx = next_datum["tss_idx"]
        tss_idx = tf.reshape(tss_idx, [3 * T])
        idx = tf.range(T, 2 * T)

        Y = next_datum["Y"]  # (1200,)
        # Y = tf.reshape(Y, [3*T, b]) # (1200, 50)
        # Y = tf.reduce_sum(Y, axis=-1) # (1200,)
        # Y = tf.reshape(Y, [1, 3*T]) # (1, 1200)

        X_epi = next_datum["X_epi"]
        X_epi = tf.reshape(X_epi, [60, 1000, 3])
        Y_h3k4me3 = X_epi[:, :, 0]
        Y_h3k27ac = X_epi[:, :, 1]
        Y_dnase = X_epi[:, :, 2]

    else:
        seq = 0
        Y = 0
        Y_h3k4me3 = 0
        Y_h3k27ac = 0
        Y_dnase = 0
        tss_idx = 0
        idx = 0
    return data_exist, seq, Y, adj, Y_h3k4me3, Y_h3k27ac, Y_dnase, idx, tss_idx
