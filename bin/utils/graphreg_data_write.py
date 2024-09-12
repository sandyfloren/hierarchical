#!/usr/bin/env python
# Copyright 2017 Calico LLC

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     https://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# =========================================================================

# This code is copied from https://github.com/calico/basenji and modified.

# =========================================================================

from optparse import OptionParser
import collections
import random
import os
import sys
import math
import tempfile
import subprocess
import shutil
import h5py
import numpy as np
import pysam
import scipy.sparse


# from basenji_data import ModelSeq
ModelSeq = collections.namedtuple("ModelSeq", ["chr", "start", "end", "label"])


import tensorflow as tf

"""
Write TF Records for batches of model sequences.
"""


################################################################################
# main
################################################################################
def main():
    usage = "usage: %prog [options] <fasta_file> <seqs_bed_file> <seqs_cov_dir> <adj_mat_dir> <tss_dir>"
    parser = OptionParser(usage)
    parser.add_option(
        "-s",
        dest="start_i",
        default=0,
        type="int",
        help="Sequence start index [Default: %default]",
    )
    parser.add_option(
        "-d",
        dest="directory",
        default=None,
        type="str",
        help="Directory to which to write TFRecord files.",
    )
    parser.add_option(
        "-e",
        dest="end_i",
        default=None,
        type="int",
        help="Sequence end index [Default: %default]",
    )
    parser.add_option(
        "--te",
        dest="target_extend",
        default=None,
        type="int",
        help="Extend targets vector [Default: %default]",
    )
    parser.add_option(
        "--ts",
        dest="target_start",
        default=0,
        type="int",
        help="Write targets into vector starting at index [Default: %default",
    )
    parser.add_option(
        "-w",
        dest="pool_width",
        default=5000,
        type="int",
        help="Sum pool width [Default: %default]",
    )
    parser.add_option("-u", dest="umap_bed", help="Unmappable bed file")
    parser.add_option(
        "--umap_t",
        dest="umap_t",
        default=0.5,
        type="float",
        help="Remove sequences with more than this unmappable bin % [Default: %default]",
    )
    parser.add_option(
        "--umap_set",
        dest="umap_set",
        default=None,
        type="float",
        help="Sequence distribution value to set unmappable positions to, eg 0.25.",
    )
    (options, args) = parser.parse_args()

    ################################################################
    # Inputs
    directory = options.directory
    if directory is None:
        directory = os.getcwd()
    fasta_file = args[0]
    seqs_bed_file = args[1]
    seqs_cov_dir = args[2]
    adj_mat_dir = args[3]
    tss_dir = args[4]

    model = "seq"  # seq/epi

    np.random.seed(0)

    # read model sequences
    model_seq_dict = {}
    n_seqs_dict = {}
    chr_list = []
    for line in open(seqs_bed_file):

        a = line.split()
        if a[0] not in model_seq_dict.keys():
            chr_list.append(a[0])
            model_seq_dict[a[0]] = [ModelSeq(a[0], int(a[1]), int(a[2]), None)]
            n_seqs_dict[a[0]] = 1
        else:
            model_seq_dict[a[0]].append(ModelSeq(a[0], int(a[1]), int(a[2]), None))
            n_seqs_dict[a[0]] += 1

    chr_list = [k for k in n_seqs_dict.keys() if k != "chrY"]
    # chr_list = ["chr1"]  ########## CHANGE LATER
    # q = np.zeros(3)
    seq_idx = 0
    for chr_temp in chr_list:
        print(f"{chr_temp}\t {seq_idx} : {seq_idx + n_seqs_dict[chr_temp]}")

        # tfr_file = data_path+'/data/tfrecords/tfr_'+model+'_'+cell_line+'_'+assay_type+'_FDR_'+fdr+'_'+chr_temp+'.tfr'
        # tfr_file = data_path+'/data/tfrecords/tfr_'+model+'_RPGC_'+cell_line+'_'+assay_type+'_FDR_'+fdr+'_'+chr_temp+'.tfr'
        tfr_file = os.path.join(directory, chr_temp + ".tfr")

        ################################################################

        # if options.end_i is None:
        model_seqs = model_seq_dict[chr_temp]
        options.end_i = len(model_seqs)

        num_seqs = options.end_i - options.start_i
        print(num_seqs)
        ################################################################
        # determine sequence coverage files

        seqs_cov_files = [
            os.path.join(seqs_cov_dir, "cage.h5"),
            os.path.join(seqs_cov_dir, "h3k4me3.h5"),
            os.path.join(seqs_cov_dir, "h3k27ac.h5"),
            os.path.join(seqs_cov_dir, "dnase.h5"),
        ]
        seq_pool_len = h5py.File(seqs_cov_files[1], "r")["targets"].shape[1]

        num_targets = len(seqs_cov_files)

        ################################################################
        # extend targets
        num_targets_tfr = num_targets
        if options.target_extend is not None:
            assert options.target_extend >= num_targets_tfr
            num_targets_tfr = options.target_extend

        # initialize targets
        targets = np.zeros((num_seqs, seq_pool_len, num_targets_tfr), dtype="float32")
        # read each target

        for ti in range(num_targets):
            seqs_cov_open = h5py.File(seqs_cov_files[ti], "r")
            tii = options.target_start + ti

            tmp = seqs_cov_open["targets"][seq_idx : seq_idx + n_seqs_dict[chr_temp]]

            if ti > 0:
                if model == "epi":
                    tmp = np.log2(tmp + 1)  # log normalize
                    # if chr_temp == "chr1":
                    #  q[ti-1] = np.max(tmp.ravel())
                    # x_max = q[ti-1]
                    # x_min = np.min(tmp.ravel())
                    # tmp = (tmp - x_min)/(x_max - x_min)   # in range [0, 1]

                # print(ti, np.sort(tmp.ravel())[-200:])
                targets[:, :, ti] = tmp

            elif ti == 0:
                # targets[:,:,ti] = tmp
                targets_y = tmp
                # print(ti, np.sort(tmp.ravel())[-200:])

            seqs_cov_open.close()
            # print("target shape: ", targets.shape)
        seq_idx = seq_idx + n_seqs_dict[chr_temp]
        ################################################################
        # modify unmappable
        mseqs_unmap = None
        if options.umap_bed is not None:
            print("Unmappable")
            if shutil.which("bedtools") is None:
                print("Install Bedtools to annotate unmappable sites", file=sys.stderr)
                exit(1)

            # annotate unmappable positions
            seq_length = 6000000
            crop_bp = 0
            seq_tlength = seq_length - 2 * crop_bp

            # TODO: pool_width of 5000 results in every bin intersecting an unmappable region
            # mseqs_unmap should be an array of size (num_seqs, 1200) with True when unmappable, False otherwise
            mseqs_unmap = annotate_unmap(
                model_seqs, options.umap_bed, seq_tlength, options.pool_width
            )
            print(mseqs_unmap.shape)  # (122, 1200)
            # filter unmappable
            mseqs_map_mask = mseqs_unmap.mean(axis=1, dtype="float64") < options.umap_t
            print(mseqs_unmap.mean(axis=1, dtype="float64"))
            model_seqs = [
                model_seqs[i] for i in range(len(model_seqs)) if mseqs_map_mask[i]
            ]
            mseqs_unmap = mseqs_unmap[mseqs_map_mask, :]

            # write to file
            # unmap_npy = '%s/mseqs_unmap.npy' % options.out_dir
            # np.save(unmap_npy, mseqs_unmap)

            # write sequences to BED
            # seqs_bed_file = '%s/sequences.bed' % options.out_dir
            # write_seqs_bed(seqs_bed_file, mseqs, True)

        # if options.umap_npy is not None and options.umap_set is not None:
        if mseqs_unmap is not None and options.umap_set is not None:
            unmap_mask = mseqs_unmap  # np.load(options.umap_npy)

            for si in range(num_seqs):
                msi = options.start_i + si

                # determine unmappable null value
                seq_target_null = np.percentile(
                    targets[si], q=[100 * options.umap_set], axis=0
                )[0]

                # set unmappable positions to null
                targets[si, unmap_mask[msi, :], :] = np.minimum(
                    targets[si, unmap_mask[msi, :], :], seq_target_null
                )

        ################################################################
        # write TFRecords

        # Graph from HiC
        # hic_matrix_file = data_path+'/data/'+cell_line+'/hic/'+assay_type+'/'+assay_type+'_matrix_FDR_'+fdr+'_'+chr_temp+'.npz'
        hic_matrix_file = os.path.join(adj_mat_dir, chr_temp + "_adjacency_matrix.npz")
        sparse_matrix = scipy.sparse.load_npz(hic_matrix_file)
        hic_matrix = sparse_matrix.todense()
        # print("hic_matrix shape: ", hic_matrix.shape)

        # tss_bin_file = data_path+'/data/tss/'+organism+'/'+genome+'/tss_bins_'+chr_temp+'.npy'
        tss_bin_file = os.path.join(tss_dir, chr_temp + "_n_tss_per_bin.npy")
        tss_bin = np.load(tss_bin_file, allow_pickle=True)
        # print("num tss:", np.sum(tss_bin))

        # bin_start_file = data_path+'/data/tss/'+organism+'/'+genome+'/bin_start_'+chr_temp+'.npy'
        bin_start_file = os.path.join(tss_dir, chr_temp + "_bin_starts.npy")
        bin_start = np.load(bin_start_file, allow_pickle=True)
        # print("bin start:", bin_start)

        # open FASTA
        if model == "seq":
            fasta_open = pysam.Fastafile(fasta_file)

        T = 400
        TT = T + T // 2
        n_batch = 0

        adj = None
        # define options
        tf_opts = tf.io.TFRecordOptions(compression_type="ZLIB")
        with tf.io.TFRecordWriter(tfr_file, tf_opts) as writer:
            for si in range(num_seqs):
                print(si + 1, end="\r")
                n_batch = n_batch + 1
                hic_start_idx = (si) * T + TT
                hic_slice = hic_matrix[
                    hic_start_idx - TT : hic_start_idx + TT,
                    hic_start_idx - TT : hic_start_idx + TT,
                ]
                adj_real = np.copy(hic_slice)

                adj_real[adj_real >= 1000] = 1000
                adj_real = np.log2(adj_real + 1)
                #                try:
                adj_real = adj_real * (np.ones([3 * T, 3 * T]) - np.eye(3 * T))
                #               except ValueError: # deal with final window being too small
                #                  continue
                adj = np.copy(adj_real)
                adj[adj > 0] = 1
                #
                if np.abs(num_seqs - si < 0):
                    last_batch = 1
                else:
                    last_batch = 0

                tss_idx = tss_bin[hic_start_idx - TT : hic_start_idx + TT]
                bin_idx = bin_start[hic_start_idx - TT : hic_start_idx + TT]

                # Y = targets[si-TT:si+TT,:,0]
                Y = targets_y[si, :]  # [hic_start_idx-TT:hic_start_idx+TT]
                X_1d = targets[si, :, 1:]  # [hic_start_idx-TT:hic_start_idx+TT,:,1:]

                X_1d = X_1d.astype(np.float16)
                adj = adj.astype(np.float16)
                adj_real = adj_real.astype(np.float16)
                Y = Y.astype(np.float16)
                bin_idx = bin_idx.astype(np.int64)
                tss_idx = tss_idx.astype(np.float16)

                # read FASTA
                if model == "seq":
                    seq_1hot = np.zeros([1, 4])

                    mseq = model_seqs[si]
                    # print(mseq)
                    seq_dna = fasta_open.fetch(mseq.chr, mseq.start, mseq.end)
                    # one hot code
                    seq_1hot = np.append(seq_1hot, dna_1hot(seq_dna.upper()), axis=0)

                    seq_1hot = np.delete(seq_1hot, 0, axis=0)
                    # print("seq: ", np.shape(seq_1hot), seq_1hot)
                if model == "seq":
                    example = tf.train.Example(
                        features=tf.train.Features(
                            feature={
                                "last_batch": _int_feature(last_batch),
                                "sequence": _bytes_feature(
                                    seq_1hot.flatten().tostring()
                                ),
                                "adj": _bytes_feature(adj.flatten().tostring()),
                                #'adj_real': _bytes_feature(adj_real.flatten().tostring()),
                                "X_1d": _bytes_feature(X_1d.flatten().tostring()),
                                "tss_idx": _bytes_feature(tss_idx.flatten().tostring()),
                                "bin_idx": _bytes_feature(bin_idx.flatten().tostring()),
                                "Y": _bytes_feature(Y.flatten().tostring()),
                            }
                        )
                    )
                elif model == "epi":
                    example = tf.train.Example(
                        features=tf.train.Features(
                            feature={
                                "last_batch": _int_feature(last_batch),
                                "adj": _bytes_feature(adj.flatten().tostring()),
                                #'adj_real': _bytes_feature(adj_real.flatten().tostring()),
                                "X_1d": _bytes_feature(X_1d.flatten().tostring()),
                                "tss_idx": _bytes_feature(tss_idx.flatten().tostring()),
                                "bin_idx": _bytes_feature(bin_idx.flatten().tostring()),
                                "Y": _bytes_feature(Y.flatten().tostring()),
                            }
                        )
                    )

                writer.write(example.SerializeToString())

            if model == "seq":
                fasta_open.close()

            print("check symetric: ", check_symmetric(adj))
            print("number of batches: ", n_batch)
            print("###########################################")


def _bytes_feature(value):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))


def _int_feature(value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))


def check_symmetric(a, tol=1e-4):
    return np.all(np.abs(a - a.transpose()) < tol)


# from basenji
def dna_1hot(seq, seq_len=None, n_uniform=False, n_sample=False):
    """dna_1hot

    Args:
      seq:       nucleotide sequence.
      seq_len:   length to extend/trim sequences to.
      n_uniform: represent N's as 0.25, forcing float16,
      n_sample:  sample ACGT for N

    Returns:
      seq_code: length by nucleotides array representation.
    """
    if seq_len is None:
        seq_len = len(seq)
        seq_start = 0
    else:
        if seq_len <= len(seq):
            # trim the sequence
            seq_trim = (len(seq) - seq_len) // 2
            seq = seq[seq_trim : seq_trim + seq_len]
            seq_start = 0
        else:
            seq_start = (seq_len - len(seq)) // 2

    seq = seq.upper()

    # map nt's to a matrix len(seq)x4 of 0's and 1's.
    if n_uniform:
        seq_code = np.zeros((seq_len, 4), dtype="float16")
    else:
        seq_code = np.zeros((seq_len, 4), dtype="bool")

    for i in range(seq_len):
        if i >= seq_start and i - seq_start < len(seq):
            nt = seq[i - seq_start]
            if nt == "A":
                seq_code[i, 0] = 1
            elif nt == "C":
                seq_code[i, 1] = 1
            elif nt == "G":
                seq_code[i, 2] = 1
            elif nt == "T":
                seq_code[i, 3] = 1
            else:
                if n_uniform:
                    seq_code[i, :] = 0.25
                elif n_sample:
                    ni = random.randint(0, 3)
                    seq_code[i, ni] = 1

    return seq_code


def annotate_unmap(mseqs, unmap_bed, seq_length, pool_width):
    """Intersect the sequence segments with unmappable regions
        and annoate the segments as NaN to possible be ignored.

    Args:
    mseqs: list of ModelSeq's
    unmap_bed: unmappable regions BED file
    seq_length: sequence length (after cropping)
    pool_width: pooled bin width

    Returns:
    seqs_unmap: NxL binary NA indicators
    """

    # print sequence segments to file
    seqs_temp = tempfile.NamedTemporaryFile()
    seqs_bed_file = seqs_temp.name
    write_seqs_bed(seqs_bed_file, mseqs)

    # hash segments to indexes
    chr_start_indexes = {}
    for i in range(len(mseqs)):
        chr_start_indexes[(mseqs[i].chr, mseqs[i].start)] = i

    # initialize unmappable array
    pool_seq_length = seq_length // pool_width
    seqs_unmap = np.zeros((len(mseqs), pool_seq_length), dtype="bool")

    # intersect with unmappable regions
    p = subprocess.Popen(
        "bedtools intersect -wo -a %s -b %s" % (seqs_bed_file, unmap_bed),
        shell=True,
        stdout=subprocess.PIPE,
    )
    for line in p.stdout:
        line = line.decode("utf-8")
        a = line.split()

        seq_chrom = a[0]
        seq_start = int(a[1])
        seq_end = int(a[2])
        seq_key = (seq_chrom, seq_start)

        unmap_start = int(a[4])
        unmap_end = int(a[5])

        overlap_start = max(seq_start, unmap_start)
        overlap_end = min(seq_end, unmap_end)

        pool_seq_unmap_start = math.floor((overlap_start - seq_start) / pool_width)
        pool_seq_unmap_end = math.ceil((overlap_end - seq_start) / pool_width)

        # skip minor overlaps to the first
        first_start = seq_start + pool_seq_unmap_start * pool_width
        first_end = first_start + pool_width
        first_overlap = first_end - overlap_start
        if first_overlap < 0.1 * pool_width:
            pool_seq_unmap_start += 1

        # skip minor overlaps to the last
        last_start = seq_start + (pool_seq_unmap_end - 1) * pool_width
        last_overlap = overlap_end - last_start
        if last_overlap < 0.1 * pool_width:
            pool_seq_unmap_end -= 1

        seqs_unmap[
            chr_start_indexes[seq_key], pool_seq_unmap_start:pool_seq_unmap_end
        ] = True
        assert (
            seqs_unmap[
                chr_start_indexes[seq_key], pool_seq_unmap_start:pool_seq_unmap_end
            ].sum()
            == pool_seq_unmap_end - pool_seq_unmap_start
        )

    return seqs_unmap


def write_seqs_bed(bed_file, seqs, labels=False):
    """Write sequences to BED file."""
    bed_out = open(bed_file, "w")
    for i in range(len(seqs)):
        line = "%s\t%d\t%d" % (seqs[i].chr, seqs[i].start, seqs[i].end)
        if labels:
            line += "\t%s" % seqs[i].label
        print(line, file=bed_out)
    bed_out.close()


################################################################################
# __main__
################################################################################
if __name__ == "__main__":
    main()
