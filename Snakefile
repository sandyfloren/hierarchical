import glob

# Set the default cell types list to an empty list if not defined
CELL_TYPES = ['k562']
CHROMOSOMES_GRAPHREG = ['chr' + str(x) for x in range(1, 23)] + ['chrX'] # no Y chromosome in GraphReg

FASTA_PATH = "/pollard/data/vertebrate_genomes/human/hg38/hg38/hg38.fa"

import pysam
# Copied from basenji/basenji/genome.py
def load_chromosomes(genome_file):
    """
    Load genome segments from either a FASTA file or chromosome length table.
    """
    file_fasta = (open(genome_file).readline()[0] == '>')
    chrom_segments = {}
    if file_fasta:
        fasta_open = pysam.Fastafile(genome_file)
        for i in range(len(fasta_open.references)):
            chrom_segments[fasta_open.references[i]] = [(0, fasta_open.lengths[i])]
        fasta_open.close()
    else:
        for line in open(genome_file):
            a = line.split()
            chrom_segments[a[0]] = [(0, int(a[1]))]
    return chrom_segments



rule all:
    input:
        #f"bin/graphreg_tf2/weights/seq_graphreg/k562/seq_graphreg_val_1_11_test_2_12.h5"
        [
            f"bin/graphreg_tf2/weights/seq_graphreg/{cell_type}/seq_graphreg_val_{val_1}_{val_2}_test_{test_1}_{test_2}.h5"
            for val_1, val_2, test_1, test_2 in zip(range(1, 11), range(11, 22), range(2, 12), range(12, 22))
            for cell_type in CELL_TYPES
        ]

rule download_annotations:
    output:  
        # TSSs 
        gencode_annotations_gz = "data/genome_annotations/gencode/gencode.v46.annotation.gtf.gz",
        gencode_annotations = "data/genome_annotations/gencode/gencode.v46.annotation.gtf",
        gencode_annotations_genes = "data/genome_annotations/gencode/gencode.v46.annotation_genes.gtf",
        gencode_annotations_transcripts = "data/genome_annotations/gencode/gencode.v46.annotation_transcripts.gtf",

        # Exclusion list
        repeatmasker = "data/genome_annotations/exclusion_list/hg38.fa.out",
        exclusion_list = "data/genome_annotations/exclusion_list/exclusion_list.bed",

        # Unmappable regions
        unmappable_bw = "data/genome_annotations/unmappable_regions/k24.Umap.MultiTrackMappability.bw",
        unmappable_macro = "data/genome_annotations/unmappable_regions/unmap_macro.bed",

    shell:
        """
        mkdir -p data/genome_annotations/exclusion_list
        mkdir -p data/genome_annotations/gencode
        mkdir -p data/genome_annotations/unmappable_regions

        wget -O {output.gencode_annotations_gz} https://ftp.ebi.ac.uk/pub/databases/gencode/Gencode_human/release_46/gencode.v46.annotation.gtf.gz
        wget -O {output.repeatmasker_gz} https://www.repeatmasker.org/genomes/hg38/RepeatMasker-rm405-db20140131/hg38.fa.out.gz
        wget -O {output.exclusion_list_gz} https://www.encodeproject.org/files/ENCFF356LFX/@@download/ENCFF356LFX.bed.gz
        
        wget -O {output.unmappable_bw} http://hgdownload.soe.ucsc.edu/gbdb/hg38/hoffmanMappability/k24.Umap.MultiTrackMappability.bw
        wget -O {output.unmappable_macro} https://raw.githubusercontent.com/calico/basenji/9e1c2e2f5b1b37ad11cfd2a1486d786d356d78a5/tutorials/data/unmap_macro.bed

        gunzip {output.gencode_annotations_gz}
        awk '{if ($3 == "gene") {print $0}}' {output.gencode_annotations} > gencode.v46.annotation_genes.gtf
        awk '{if ($3 == "transcript") {print $0}}' {output.gencode_annotations} > gencode.v46.annotation_transcripts.gtf
        """
    

rule download_data:
    output:
        # bam files
        cage_bam = "data/encode/cage/{cell_type}_cage.bam",
        
        # bw files
        dnase_bw = "data/encode/dnase/{cell_type}_dnase.bw",
        h3k4me3_bw = "data/encode/h3k4me3/{cell_type}_h3k4me3.bw",
        h3k27ac_bw = "data/encode/h3k27ac/{cell_type}_h3k27ac.bw",

        # HiC data
        combined_30_hic = "data/ncbi_geo/hic/{cell_type}/combined_30.hic"

    shell:
        """
        mkdir -p data/encode/cage
        mkdir -p data/encode/dnase
        mkdir -p data/encode/h3k4me3
        mkdir -p data/encode/h3k27ac

        
        mkdir -p data/ncbi_geo/hic/{wildcards.cell_type}


        wget -O {output.cage_bam} https://www.encodeproject.org/files/ENCFF754FAU/@@download/ENCFF754FAU.bam
        wget -O {output.h3k4me3_bw} https://www.encodeproject.org/files/ENCFF089RDX/@@download/ENCFF089RDX.bigWig 
        wget -O {output.h3k27ac_bw} https://www.encodeproject.org/files/ENCFF381NDD/@@download/ENCFF381NDD.bigWig
        wget -O {output.dnase_bw} https://www.encodeproject.org/files/ENCFF414OGC/@@download/ENCFF414OGC.bigWig

        
        wget -O {output.combined_30_hic} https://ftp.ncbi.nlm.nih.gov/geo/series/GSE63nnn/GSE63525/suppl/GSE63525_K562_combined_30.hic
        """



rule prepare_unmappable_regions:
    input:
        repeatmasker = "data/genome_annotations/exclusion_list/hg38.fa.out",
        exclusion_list_bed = "data/genome_annotations/exclusion_list/exclusion_list.bed",
        unmappable_bw = "data/genome_annotations/unmappable_regions/k24.Umap.MultiTrackMappability.bw",

    output:
        satellites_bed = "data/genome_annotations/exclusion_list/satellite_repeats.bed",
        exclusion_list_repeats_bed = "data/genome_annotations/exclusion_list/exclusion_list_with_repeats.bed",
        unmappable_k24_l32_bed = "data/genome_annotations/unmappable_regions/unmap_k24_l32.bed",
        unmappable_plus_exclusion_bed = "data/genome_annotations/unmappable_regions/unmappable_regions_plus_exclusion_list_with_repeats.bed"

    shell:
        """
        grep -i satellite {input.repeatmasker} | awk -v OFS='\t' '{{print $5,$6,$7}}' > {output.satellites_bed}
        cat {input.exclusion_list_bed} {output.satellites_bed} | bedtools sort | bedtools merge > {output.exclusion_list_repeats_bed}
        python ~/hierarchical/bin/utils/unmappable_bed.py -l 32 {input.unmappable_bw} > {output.unmappable_k24_l32_bed}
        cat {output.exclusion_list_repeats_bed} {output.unmappable_k24_l32_bed} | awk 'BEGIN {{OFS="\t"}} {{print $1,$2,$3}}' | bedtools sort | bedtools merge > {output.unmappable_plus_exclusion_bed}
        """

rule prepare_bed_files:
    output:
        windows_bed = 'data/graphreg_data/seqs/windows_6Mb_no_gaps.bed',
        bins_bed = 'data/graphreg_data/seqs/genome_bins.bed'
    
    params:
        window_size = 6000000,
        stride = 2000000,
        binsize = 5000
    run:
        # Generate chrom_segments
        chromosomes = ['chr' + str(i) for i in range(1, 23)] + ['chrX', 'chrY']
        chrom_segments = load_chromosomes(FASTA_PATH)
        chrom_segments = {k: v for k, v in chrom_segments.items() if k in chromosomes}

        # Write genome windows file
        window_size = params.window_size
        stride = params.stride
        with open(output.windows_bed, 'w') as file:
            for chrom, segments in chrom_segments.items():
                for segment in segments:
                    begin, end = segment
                    while begin + window_size <= end:
                        file.write(f'{chrom}\t{begin}\t{begin + window_size}\n')
                        begin += stride

        # Write genome bins file
        binsize = params.binsize
        with open(output.bins_bed, 'w') as file:
            for chrom, segments in chrom_segments.items():
                for segment in segments:
                    begin, end = segment
                    while begin + binsize <= end:
                        file.write(f'{chrom}\t{begin}\t{begin + binsize}\n')
                        begin += binsize


rule bam_to_bigwig:
    input:
        # bam files
        cage_bam = "data/encode/cage/{cell_type}_cage.bam",
        #dnase_bam = "data/encode/dnase/{cell_type}_dnase.bam",
        #h3k4me3_bam = "data/encode/h3k4me3/{cell_type}_h3k4me3.bam",
        #h3k27ac_bam = "data/encode/h3k27ac/{cell_type}_h3k27ac.bam",

    output:
        # bigwig files
        cage_bw = "data/encode/cage/{cell_type}_cage.bw",
        #h3k4me3_bw = "data/encode/h3k4me3/{cell_type}_h3k4me3.bw",
        #h3k27ac_bw = "data/encode/h3k27ac/{cell_type}_h3k27ac.bw",
        #dnase_bw = "data/encode/dnase/{cell_type}_dnase.bw",

    shell:
        """
        python bin/utils/bam_cov.py {input.cage_bam} {output.cage_bw}
        """
        #python bin/utils/bam_cov.py {input.h3k4me3_bam} {output.h3k4me3_bw}
        #python bin/utils/bam_cov.py {input.h3k27ac_bam} {output.h3k27ac_bw}
        #python bin/utils/bam_cov.py {input.dnase_bam} {output.dnase_bw}
        #"""
        
rule bigwig_to_h5_cage:
    input:
        # bigwig files
        cage_bw = "data/encode/cage/{cell_type}_cage.bw",

        # bed files
        unmappable_plus_exclusion_bed = "data/genome_annotations/unmappable_regions/unmappable_regions_plus_exclusion_list_with_repeats.bed",
        windows_bed = "data/graphreg_data/seqs/windows_6Mb_no_gaps.bed"
    output:
        # h5 files
        cage_h5 = "data/graphreg_data/1d/{cell_type}/cage.h5",
        
    shell:
        """
        python bin/utils/basenji_data_read.py -b {input.unmappable_plus_exclusion_bed} --black_pct 0.1 -c 50000 -u sum -w 5000 {input.cage_bw} {input.windows_bed} {output.cage_h5}
        """

rule bigwig_to_h5_h3k4me3:
    input:
        # bigwig files
        h3k4me3_bw = "data/encode/h3k4me3/{cell_type}_h3k4me3.bw",
        
        # bed files
        unmappable_plus_exclusion_bed = "data/genome_annotations/unmappable_regions/unmappable_regions_plus_exclusion_list_with_repeats.bed",
        windows_bed = "data/graphreg_data/seqs/windows_6Mb_no_gaps.bed"
    output:
        # h5 files
        h3k4me3_h5 = "data/graphreg_data/1d/{cell_type}/h3k4me3.h5",

    shell:
        """
        python bin/utils/basenji_data_read.py -b {input.unmappable_plus_exclusion_bed} --black_pct 0.1 -c 50000 -u sum -w 100 {input.h3k4me3_bw} {input.windows_bed} {output.h3k4me3_h5}
        """

rule bigwig_to_h5_h3k27ac:
    input:
        # bigwig files
        h3k27ac_bw = "data/encode/h3k27ac/{cell_type}_h3k27ac.bw",

        # bed files
        unmappable_plus_exclusion_bed = "data/genome_annotations/unmappable_regions/unmappable_regions_plus_exclusion_list_with_repeats.bed",
        windows_bed = "data/graphreg_data/seqs/windows_6Mb_no_gaps.bed"
    output:
        # h5 files
        h3k27ac_h5 = "data/graphreg_data/1d/{cell_type}/h3k27ac.h5"

    shell:
        """
        python bin/utils/basenji_data_read.py -b {input.unmappable_plus_exclusion_bed} --black_pct 0.1 -c 50000 -u sum -w 100 {input.h3k27ac_bw} {input.windows_bed} {output.h3k27ac_h5}
        """

rule bigwig_to_h5_dnase:
    input:
        # bigwig files
        dnase_bw = "data/encode/dnase/{cell_type}_dnase.bw",

        # bed files
        unmappable_plus_exclusion_bed = "data/genome_annotations/unmappable_regions/unmappable_regions_plus_exclusion_list_with_repeats.bed",
        windows_bed = "data/graphreg_data/seqs/windows_6Mb_no_gaps.bed"
    output:
        # h5 files
        dnase_h5 = "data/graphreg_data/1d/{cell_type}/dnase.h5",

    shell:
        """
        python bin/utils/basenji_data_read.py -b {input.unmappable_plus_exclusion_bed} --black_pct 0.1 -c 50000 -u sum -w 100 {input.dnase_bw} {input.windows_bed} {output.dnase_h5}
        """

rule process_3d_data:
    input:
        combined_30_hic = "data/ncbi_geo/hic/{cell_type}/combined_30.hic",
        unmappable_bw = "data/genome_annotations/unmappable_regions/k24.Umap.MultiTrackMappability.bw",
        bins_bed = "data/graphreg_data/seqs/genome_bins.bed"
    output:
        combined_hic_result = "data/ncbi_geo/hic/{cell_type}/combined_result.txt",
        adj_mats = expand("data/graphreg_data/3d/{{cell_type}}/{chrom}_adjacency_matrix.npz", chrom=CHROMOSOMES_GRAPHREG)
    shell:
        """
        Rscript --no-save bin/utils/hicdcplus.R --map_bw {input.unmappable_bw} {input.combined_30_hic} {output.combined_hic_result}
        python bin/utils/create_graphs.py -d data/graphreg_data/3d/{wildcards.cell_type} {output.combined_hic_result} {input.bins_bed}
        """

rule process_annotations:
    input:
        gencode_annotations = "data/genome_annotations/gencode/gencode.v46.annotation.gtf",
        bins_bed = "data/graphreg_data/seqs/genome_bins.bed"
    output:
        bin_starts = expand("data/graphreg_data/tss/{{cell_type}}/{chrom}_bin_starts.npy", chrom=CHROMOSOMES_GRAPHREG),
        gene_names = expand("data/graphreg_data/tss/{{cell_type}}/{chrom}_gene_names.npy", chrom=CHROMOSOMES_GRAPHREG),
        n_tss_per_bin = expand("data/graphreg_data/tss/{{cell_type}}/{chrom}_n_tss_per_bin.npy", chrom=CHROMOSOMES_GRAPHREG),
        tss_pos_in_bins = expand("data/graphreg_data/tss/{{cell_type}}/{chrom}_tss_pos_in_bins.npy", chrom=CHROMOSOMES_GRAPHREG)
    shell:
        """
        python bin/utils/tss.py --protein -d data/graphreg_data/tss/{wildcards.cell_type} {output.gencode_annotations} {input.bins_bed}
        """


rule write_tfrecords:
    input:
        windows_bed = "data/graphreg_data/seqs/windows_6Mb_no_gaps.bed",
        cage_h5 = "data/graphreg_data/1d/{cell_type}/cage.h5",
        h3k4me3_h5 = "data/graphreg_data/1d/{cell_type}/h3k4me3.h5",
        h3k27ac_h5 = "data/graphreg_data/1d/{cell_type}/h3k27ac.h5",
        dnase_h5 = "data/graphreg_data/1d/{cell_type}/dnase.h5",
        adj_mats = expand("data/graphreg_data/3d/{{cell_type}}/{chrom}_adjacency_matrix.npz", chrom=CHROMOSOMES_GRAPHREG)
    output:
        tfrs = expand("data/graphreg_data/tfr/{{cell_type}}/{chrom}.tfr", chrom=CHROMOSOMES_GRAPHREG)
    shell:
        """
        python bin/utils/graphreg_data_write.py -d data/graphreg_data/tfr/{wildcards.cell_type} {FASTA_PATH} {input.windows_bed} data/graphreg_data/1d/{wildcards.cell_type} data/graphreg_data/3d/{wildcards.cell_type} data/graphreg_data/tss
        """

#############################################################################################################################
#                                                                                                                           #
#                                                   Train GraphReg                                                          #
#                                                                                                                           #    
#############################################################################################################################


rule train_seq_cnn:
    input:
        tfrs = expand("data/graphreg_data/tfr/{{cell_type}}/{chrom}.tfr", chrom=CHROMOSOMES_GRAPHREG),
        config = "bin/graphreg_tf2/configs/seq_cnn_base.yaml"
    output:
        cnn_weights = "bin/graphreg_tf2/weights/seq_cnn_base/{cell_type}/seq_cnn_base_val_{val_1}_{val_2}_test_{test_1}_{test_2}.h5"
    shell:
        """
        python bin/graphreg_tf2/train_seq_cnn_base.py -v {wildcards.val_1},{wildcards.val_2} -t {wildcards.test_1},{wildcards.test_2} -c {wildcards.cell_type} --deterministic --overwrite --no_wandb {input.config}
        """

rule train_seq_graphreg:
    input:
        expand("data/graphreg_data/tfr/{{cell_type}}/{chrom}.tfr", chrom=CHROMOSOMES_GRAPHREG),
        cnn_weights="bin/graphreg_tf2/weights/seq_cnn_base/{cell_type}/seq_cnn_base_val_{val_1}_{val_2}_test_{test_1}_{test_2}.h5",
        config = "bin/graphreg_tf2/configs/seq_graphreg.yaml",
    output:
        graphreg_weights="bin/graphreg_tf2/weights/seq_graphreg/{cell_type}/seq_graphreg_val_{val_1}_{val_2}_test_{test_1}_{test_2}.h5"
    shell:
        """
        python bin/graphreg_tf2/train_seq_graphreg.py -v {wildcards.val_1},{wildcards.val_2} -t {wildcards.test_1},{wildcards.test_2} -c {wildcards.cell_type} --deterministic --overwrite --no_wandb {input.config}
        """

