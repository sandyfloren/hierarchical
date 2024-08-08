#!/usr/bin/env python
# Code modified from https://github.com/karbalayghareh/GraphReg/blob/f5cec7fc784c2f503c359f2c7da95a8353c38e5c/utils/find_tss.py
import argparse
import os
import scipy
import numpy as np
import pandas as pd
# The average human gene has 4 TSSs

# GTF file format
gtf_col_dict = {
    'chrom': 0,
    'source': 1,
    'feature': 2,
    'start': 3,
    'end': 4,
    'score': 5,
    'strand': 6,
    'frame': 7,
    'attribute': 8
}

def main():
    # usage: python create_graph.py <annotations file> <bins file>
    parser = argparse.ArgumentParser()

    parser.add_argument('annotations', type=str, help='Annotations .gtf file. Must be in sorted order by chromosome/seqname.')
    parser.add_argument('bins', type=str, help='.bed file containing coordinates of genomic bins.')
    parser.add_argument('-d', '--directory', type=str, help='Output directory.')
    parser.add_argument('--protein', action='store_true', help='Whether to save TSSs from protein-coding genes only.')

    args = parser.parse_args()
    directory = args.directory
    if directory is None:
        directory = os.getcwd()

    def save_tss(
            bin_starts: np.array, 
            n_tss_per_bin: np.array, 
            gene_names: np.array, 
            tss_pos_in_bins: np.array,
            chrom: str
            ):
        print(f'Saving TSSs for {chrom}')
        np.save(os.path.join(directory, f'{chrom}_bin_starts.npy'), bin_starts)
        np.save(os.path.join(directory, f'{chrom}_n_tss_per_bin.npy'), n_tss_per_bin)
        np.save(os.path.join(directory, f'{chrom}_gene_names.npy'), gene_names)
        np.save(os.path.join(directory, f'{chrom}_tss_pos_in_bins.npy'), tss_pos_in_bins)
        

    # Create dict of chromosome: number of bins
    chromosomes = ['chr' + str(i) for i in range(1, 23)] + ['chrX', 'chrY']
    n_bins_dict = {k: 0 for k in chromosomes}
    binsize = None
    with open(args.bins, 'r') as infile:
        for bin in infile:
            chrom, start, end = bin.split()
            if binsize is None:
                binsize = int(end) - int(start)
            n_bins_dict[chrom] += 1


    bins_df = pd.read_table(args.bins, header=None, sep='\t', names=['chrom', 'start', 'stop'])
    chrs = set()
    bin_starts = None
    with open(args.annotations, 'r') as infile:
        chrom = None
        new_chr = True

        for line in infile:
            # skip header lines 
            if line[0:2] == '##':
                continue
            values = line.split('\t')

            # skip chromosomes not in dict
            if values[gtf_col_dict['chrom']] not in chromosomes:
                continue
            
            if chrom is None:
                chrom = values[gtf_col_dict['chrom']]
                # deal with chromosomes given just as numbers
                if not 'chr' in chrom: 
                    chrom = 'chr' + chrom
            elif values[gtf_col_dict['chrom']] != chrom:
                # new chromosome
                new_chr = True
                chrs.add(chrom)
                chrom_prev = chrom
                assert values[gtf_col_dict['chrom']] not in chrs, 'chromosome/seqname column must be sorted order'
                chrom = values[gtf_col_dict['chrom']]
                # deal with chromosomes given just as numbers
                if not 'chr' in chrom: 
                    chrom = 'chr' + chrom
            if new_chr:
                if bin_starts is not None:
                    # save previous datastructure
                    save_tss(bin_starts, n_tss_per_bin, gene_names, tss_pos_in_bins, chrom_prev)

                
                # initialize new datastructure and set up output filenames
                bin_starts = bins_df[bins_df['chrom'] == chrom]['start'].to_numpy()
                n_tss_per_bin = np.zeros(n_bins_dict[chrom], dtype=int)
                gene_names = np.empty(n_bins_dict[chrom], dtype=object)
                tss_pos_in_bins = np.zeros(n_bins_dict[chrom], dtype=int)

            new_chr = False

            # only consider transcripts
            # TODO: we need to think about whether to use the transcript annotation
            # or the first exon of each transcript (the Basenji approach)
            if values[gtf_col_dict['feature']] != 'transcript':
                continue

            # update datastructure
            attribute = {k: v for k, v in [x.split() for x in values[gtf_col_dict['attribute']].split(';') if x not in {'', '\n'}]}

            # transcript_id = attribute['transcript_id']
            # gene_type = attribute['gene_type']

            if args.protein:
                if attribute['gene_type'] != '"protein_coding"':
                    continue
            # get TSS position
            if values[gtf_col_dict['strand']] == '+':
                tss_pos = int(values[gtf_col_dict['start']])
            elif values[gtf_col_dict['strand']] == '-':
                tss_pos = int(values[gtf_col_dict['end']])
            else:
                raise ValueError('strand column must be either `+` or `-`')

            # Get bin index
            bin_idx = tss_pos // binsize

            # increment of TSSs in bin
            n_tss_per_bin[bin_idx] += 1

            # add gene name
            if gene_names[bin_idx] is None:
                gene_names[bin_idx] = attribute['gene_name'].replace('"', '')
                prev_gene_name = attribute['gene_name'].replace('"', '')
            elif attribute['gene_name'].replace('"', '') == prev_gene_name:
                pass
            else:
                gene_names[bin_idx] += '+' + attribute['gene_name'].replace('"', '')
                prev_gene_name = attribute['gene_name'].replace('"', '')
            # TSS positions in bin
            if tss_pos_in_bins[bin_idx] == 0:
                tss_pos_in_bins[bin_idx] = tss_pos % binsize
            else:
                # Use first TSS
                tss_pos_in_bins[bin_idx] = min(tss_pos_in_bins[bin_idx], tss_pos % binsize)


        # save last datastructure
        save_tss(bin_starts, n_tss_per_bin, gene_names, tss_pos_in_bins, chrom)


if __name__ == '__main__':
    main()