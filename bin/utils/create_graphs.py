#!/usr/bin/env python
# Code modified from https://github.com/karbalayghareh/GraphReg/blob/f5cec7fc784c2f503c359f2c7da95a8353c38e5c/utils/hic_to_graph.py
# This script takes a file of 3D interactions and outputs an adjacency matrix (in sparse matrix form) for each chromosome included in the input file. The entries represent counts or normalized counts.
# Only intrachromosomal interactions are included.
import argparse
import os
import scipy
import numpy as np


def main():
    # usage: python create_graph.py <interactions file> <bins file>
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "interactions",
        type=str,
        help="3D interactions .txt file from HiC-DC+. This file must have the column `chrI` in sorted order.",
    )
    parser.add_argument(
        "bins", type=str, help=".bed file containing coordinates of genomic bins."
    )
    parser.add_argument(
        "-d",
        "--directory",
        type=str,
        help="Directory in which to output TFRecords.",
    )
    parser.add_argument(
        "-n",
        "--normalize",
        action="store_true",
        help="Whether to output observed/expected counts instead of observed (raw) counts.",
    )
    parser.add_argument(
        "-v",
        "--visualize",
        action="store_true",
        help="Whether to print a 10x10 section of the sparse matrix.",
    )

    args = parser.parse_args()
    directory = args.directory
    if directory is None:
        directory = os.getcwd()

    def save_sparse(mat: np.array):
        if args.visualize:
            print(mat[2000:2010, 2000:2010])
        sparse_mat = scipy.sparse.csr_matrix(mat, dtype=np.float32)
        scipy.sparse.save_npz(fname, sparse_mat)

    # Create dict of chromosome: number of bins
    chromosomes = ["chr" + str(i) for i in range(1, 23)] + ["chrX", "chrY"]
    n_bins_dict = {k: 0 for k in chromosomes}
    binsize = None
    with open(args.bins, "r") as infile:
        for bin in infile:
            chrom, start, end = bin.split()
            if binsize is None:
                binsize = int(end) - int(start)
            n_bins_dict[chrom] += 1

    chrs = set()
    with open(args.interactions, "r") as infile:
        header = infile.readline().split()
        # order of columns is not necessarly known
        chrI_idx = header.index("chrI")
        startI_idx = header.index("startI")
        endI_idx = header.index("endI")
        chrJ_idx = header.index("chrJ")
        startJ_idx = header.index("startJ")
        counts_idx = header.index("counts")
        mu_idx = header.index("mu")

        chrI = None
        new_chr = True
        mat = None

        for line in infile:
            values = line.split()

            if chrI is None:
                chrI = values[chrI_idx]
            elif values[chrI_idx] != chrI:
                # new chromosome
                new_chr = True
                chrs.add(chrI)
                assert (
                    values[chrI_idx] not in chrs
                ), "chrI column must be in sorted order"
                chrI = values[chrI_idx]

            if new_chr:
                if mat is not None:
                    # save previous matrix
                    save_sparse(mat)

                # initialize new matrix and set up output filename
                n_bins = n_bins_dict[chrI]
                print(f"{chrI}: {n_bins} bins")

                mat = np.zeros((n_bins, n_bins))
                norm = "_normalized" if args.normalize else ""
                fname = os.path.join(directory, f"{chrI}_adjacency_matrix{norm}")

            new_chr = False

            # exclude interchromosomal interactions
            if chrI != values[chrJ_idx]:
                continue

            # update matrix
            i, j = (
                int(values[startI_idx]) // binsize,
                int(values[startJ_idx]) // binsize,
            )
            if args.normalize:
                # use observed/expected counts
                mat[i, j] = int(values[counts_idx]) / float(values[mu_idx])
                mat[j, i] = mat[i, j]  # symmetric
            else:
                # Use observed (raw) counts
                mat[i, j] = int(values[counts_idx])
                mat[j, i] = mat[i, j]  # symmetric

        # save last matrix
        save_sparse(mat)


if __name__ == "__main__":
    main()
