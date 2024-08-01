#!/usr/bin/env Rscript

library(HiCDCPlus)
library(optparse)
# Usage: ./hicdcplus.R <input .hic file> <output .hic, .txt, or .txt.gz file>

# This program is now deterministic, becuase the parallelized functions are no longer used. 
# That the Y chromosome causes an error in HiCDCPlus, and neither chrX nor chrY were included in GraphReg
# Currently this script only does intrachromosomal interactions
# There were MORE significant interactions when using a mappability file, not fewer.

options(error=traceback)
option_list <- list(
    make_option(c("--map_bw", "-m"), default=NULL, help="Mappability bigwig file."),
    make_option(c("--binsize", "-b"), default=5000, help="Binsize in bp if bin_type=’Bins-uniform’ (or number of RE fragment cut sites if bin_type=’Bins-RE-sites’)."),
    make_option(c("--bin_type", "-t"), default="Bins-uniform", help="’Bins-uniform’ if uniformly binned by binsize in bp, or ’Bins-RE-sites’ if binned by number of restriction enzyme fragments."),
    make_option(c("--feature_type", "-f"), default="RE-based", help="’RE-based’ if features are to be computed based on restriction enzyme fragments. ’RE-agnostic’ ignores restriction enzyme cutsite information and computes features gc and map based on binwide averages. bin_type has to be ’Bins-uniform’ if feature_type=’RE-agnostic’."),
    make_option(c("--max_dist", "-d"), default=2e6, help="Max genomic interaction distance."),
    make_option(c("--re_cutsite", "-r"), default="GATC", help="Restriction enzyme cutsite sequence(s). Example for multiple cutsites: -r GATC,GANTC"),
    make_option(c("--chrs", "-c"), default=NULL, help="Chromosomes to include. Example for multiple chromosomes: -c chr1,chr2,chrX"),
    make_option(c("--mode"), default="normcounts", help="What scores in the output .hic file should represent. Allowable options are: ’pvalue’ for -log10 significance p-value, ’qvalue’ for -log10 FDR corrected p-value, ’normcounts’ for observed/expected counts, and ’zvalue’ for standardized counts (observed-expected counts)/(modeled standard deviation of expected counts) and ’raw’ to pass-through raw counts."),
    make_option(c("--bintolen"), default=NULL, help="Path to which to write bintolen file. If file already exists, it will be overwritten unless --use_existing is specified"),
    make_option(c("--use_existing"), default=FALSE, action="store_true", help="Whether to use an existing bintolen file instead of creating a new one with construct_features. Must also specify --bintolen ’path/to/bintolen’"),
    make_option(c("--alpha"), default=0.1, help="Significance threshold for q-value."),
    make_option(c("--genome"), default="Hsapiens", help="Organism."),
    make_option(c("--build"), default="hg38", help="Genome build."),
    make_option(c("--memory"), default=8, help="Java memory to generate .hic files. Defaults to 8. Up to 64 is recommended for higher resolutions.")
)
parser <- OptionParser(option_list = option_list)
arguments <- parse_args(parser, positional_arguments=2)
options <- arguments$options
infile <- arguments$args[1]
outfile <- arguments$args[2]
re_cutsites <- strsplit(options$re_cutsite, ",")[[1]]
if(!is.null(options$chrs)) {
    chrs <- strsplit(options$chrs, ",")[[1]]
} else {
    chrs <- paste0("chr", c(1:21, "X"))
}
bintolen_path <- options$bintolen
use_existing <- options$use_existing
overwrite <- TRUE
binned_uniform <- if(options$bin_type == "Bins-uniform") TRUE else FALSE
set.seed(123) #HiC-DC downsamples rows for modeling


if(use_existing) {
    if(!is.null(bintolen_path) & file.exists(bintolen_path)) {
        overwrite <- FALSE
    } else {
        stop("Must supply a valid path to --bintolen when using --use_existing.")
    }
} else {
    if(!is.null(bintolen_path)){
        bintolen_path <- bintolen_path
    } else {
        bintolen_path <- paste0(getwd(), "/", options$build, "_", round(options$binsize / 1000), "kb_bintolen.txt.gz")
    }
}

if(overwrite) {
    # Create genomic features
    #construct_features_parallel(
    construct_features(
        output_path=sub("_bintolen\\.txt\\.gz$", "", bintolen_path),
        gen=options$genome,
        gen_ver=options$build,
        sig=re_cutsites,             
        bin_type=options$bin_type, 
        feature_type=options$feature_type,
        binsize=options$binsize,
        chrs=chrs,
        wg_file=options$map_bw
    #   ncore=options$cores
        )
}

# Create genomic interactions list
gi_list <- generate_bintolen_gi_list(
    bintolen_path=bintolen_path,
    chrs=chrs,
    Dthreshold=options$max_dist,
    binned=binned_uniform,
    binsize=options$binsize,
    gen=options$genome,
    gen_ver=options$build
    )

# Add HiC data
gi_list <- add_hic_counts(gi_list, hic_path = infile, chrs=chrs)
gi_list <- expand_1D_features(gi_list, chrs=chrs)

# Find significant interactions
gi_list <- HiCDCPlus(
    gi_list,
    chrs=chrs,
    binned=binned_uniform,
    Dmax=options$max_dist
    )


if(strsplit(outfile, "\\.")[[1]][-1] == "hic") {
    # Write to .hic file
    hicdc2hic(
        gi_list,
        outfile,
        mode=options$mode,
        chrs=chrs,
        gen_ver=options$build,
        memory=options$memory
        )
} else {
    # Write to .txt or .txt.gz file
    gi_list_write(
        gi_list,
        outfile,
        chrs=chrs,
        columns="minimal",
        rows=if(!is.null(options$alpha)) "significant" else "all",
        significance_threshold=options$alpha,
        score=options$mode
    )
}
