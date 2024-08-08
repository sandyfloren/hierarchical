# Import the glob module to handle pathnames
import glob

# Set the default cell types list to an empty list if not defined
CELL_TYPES = ['k562']
CHROMOSOMES_GRAPHREG = ['chr' + x for x in range(1, 23)] + ['chrX'] # no Y chromosome in GraphReg

# Rule to handle all cell types
rule all:
    input:
        expand("data/graphreg_data/tfr/{cell_type}/{chrom}.tfr", cell_type=CELL_TYPES, chrom=CHROMOSOMES_GRAPHREG)


# Define other rules similarly, ensuring that each rule has `{wildcards.cell_type}` in paths where necessary
# For example, updating the download_data rule:
rule download_data:
    output:
        cage_rep1_bam = "data/encode/cage/{wildcards.cell_type}_cage_rep1.bam"
        cage_rep2_bam = "data/encode/cage/{wildcards.cell_type}_cage_rep2.bam"
        dnase_bam = "data/encode/dnase/{wildcards.cell_type}_dnase.bam"
        h3k4me3_bw = "data/encode/h3k4me3/{wildcards.cell_type}_h3k4me3.bigWig"
        h3k27ac_bw = "data/encode/h3k27ac/{wildcards.cell_type}_h3k27ac.bigWig"
        
        gencode_annotations = "data/genome_annotations/gencode/gencode.v46.annotation"

        repeatmasker_gz = "data/genome_annotations/exclusion_list/hg38.fa.out.gz"
        exclusion_list_gz = "data/genome_annotations/exclusion_list/exclusion_list.bed.gz"

        unmappable_bed = "data/genome_annotations/unmappable_regions/unmappable_regions_plus_exclusion_list_with_repeats.bed"
        unmappable_bw = "data/genome_annotations/unmappable_regions/k24.Umap.MultiTrackMappability.bw"
        unmappable_macro = "data/genome_annotations/unmappable_regions/unmap_macro.bed"

        combined_30_hic = "data/ncbi_geo/hic/{wildcards.cell_type}/combined_30.hic"

    shell:
        """
        mkdir -p data/encode/cage/{wildcards.cell_type}
        mkdir -p data/encode/dnase/{wildcards.cell_type}
        mkdir -p data/encode/h3k4me3/{wildcards.cell_type}
        mkdir -p data/encode/h3k27ac/{wildcards.cell_type}

        mkidr -p data/genome_annotations/exclusion_list
        mkidr -p data/genome_annotations/gencode
        mkidr -p data/genome_annotations/unmappable_regions

        mkidr -p data/ncbi_geo/hic/{wildcards.cell_type}

        wget -O {output.cage_rep1_bam} https://www.encodeproject.org/files/ENCFF754FAU/@@download/ENCFF754FAU.bam
        wget -O {output.cage_rep2_bam} https://www.encodeproject.org/files/ENCFF366MWI/@@download/ENCFF366MWI.bam
        wget -O {output.dnase_bam} https://www.encodeproject.org/files/ENCFF257HEE/@@download/ENCFF257HEE.bam
        wget -O {output.h3k4me3_bw} https://www.encodeproject.org/files/ENCFF253TOF/@@download/ENCFF253TOF.bigWig
        wget -O {output.h3k27ac_bw} https://www.encodeproject.org/files/ENCFF381NDD/@@download/ENCFF381NDD.bigWig

        wget -O {output.repeatmasker_gz} https://www.repeatmasker.org/genomes/hg38/RepeatMasker-rm405-db20140131/hg38.fa.out.gz
        wget -O {output.exclusion_list_gz} https://www.encodeproject.org/files/ENCFF356LFX/@@download/ENCFF356LFX.bed.gz
        
        wget -O {output.unmappable_bw} http://hgdownload.soe.ucsc.edu/gbdb/hg38/hoffmanMappability/k24.Umap.MultiTrackMappability.bw
        wget -O {output.unmappable_macro} https://raw.githubusercontent.com/calico/basenji/9e1c2e2f5b1b37ad11cfd2a1486d786d356d78a5/tutorials/data/unmap_macro.bed
        
        wget -O {output.combined_30_hic} https://ftp.ncbi.nlm.nih.gov/geo/series/GSE63nnn/GSE63525/suppl/GSE63525_K562_combined_30.hic
        """






rule prepare_unmappable_regions:
    input:
        repeatmasker_gz = "data/genome_annotations/exclusion_list/hg38.fa.out.gz"
        exclusion_list_gz = "data/genome_annotations/exclusion_list/exclusion_list.bed.gz"
        unmappable_bw = "data/genome_annotations/unmappable_regions/k24.Umap.MultiTrackMappability.bw"

    output:
        repeatmasker = "data/genome_annotations/exclusion_list/hg38.fa.out"
        satellites_bed = "data/genome_annotations/exclusion_list/satellite_repeats.bed"
        exclusion_list_bed = "data/genome_annotations/exclusion_list/exclusion_list.bed"
        exclusion_list_repeats_bed = "data/genome_annotations/exclusion_list/exclusion_list_with_repeats.bed"
        unmappable_k24_l32_bed = "data/genome_annotations/unmappable_regions/unmap_k24_l32.bed"

    shell:
        """
        gunzip {input.repeatmasker_gz}
        gunzip {input.exclusion_list_gz}

        grep -i satellite {output.repeatmasker} | awk -v OFS='\t' '{ print $5,$6,$7 }' > {output.satellites_bed}
        
        cat {output.exclusion_list_bed} {output.satellites_bed} | bedtools sort | bedtools merge > {output.exclusion_list_repeats_bed}
        
        python ~/hierarchical/bin/utils/unmappable_bed.py -l 32 {input.unmappable_bw} > {output.unmappable_k24_l32_bed}
        
        cat {output.exclusion_list_repeats_bed} {output.unmappable_k24_l32_bed} | awk 'BEGIN {OFS="\t"} {print $1,$2,$3}' | bedtools sort | bedtools merge > {output.unmappable_bed}
        """

rule process_1d_data:
    input:
        cage_rep1_bam = "data/encode/cage/{wildcards.cell_type}_cage_rep1.bam"
        cage_rep2_bam = "data/encode/cage/{wildcards.cell_type}_cage_rep2.bam"
        dnase_bam = "data/encode/dnase/{wildcards.cell_type}_dnase.bam"
        h3k4me3_bw = "data/encode/h3k4me3/{wildcards.cell_type}_h3k4me3.bigWig"
        h3k27ac_bw = "data/encode/h3k27ac/{wildcards.cell_type}_h3k27ac.bigWig"

        exclusion_list_bed = "data/encode/exclusion_list/exclusion_list_with_repeats.bed"
        windows_bed = "data/graphreg_data/seqs/windows_6Mb_no_gaps.bed"
    output:
        cage_rep1_bw = "data/encode/cage/{wildcards.cell_type}_cage_rep1.bw"
        cage_rep2_bw = "data/encode/cage/{wildcards.cell_type}_cage_rep2.bw"
        dnase_bw = "data/encode/dnase/{wildcards.cell_type}_dnase.bw"

        cage_rep1_h5 = "data/graphreg_data/1d/{wildcards.cell_type}/cage_rep1.h5"
        cage_rep2_h5 = "data/graphreg_data/1d/{wildcards.cell_type}/cage_rep2.h5"
        dnase_h5 = "data/graphreg_data/1d/{wildcards.cell_type}/dnase.h5"
        h3k4me3_h5 = "data/graphreg_data/1d/{wildcards.cell_type}/h3k4me3.h5"
        h3k27ac_h5 = "data/graphreg_data/1d/{wildcards.cell_type}/h3k27ac.h5"

    shell:
        """
        python bin/utils/bam_cov.py {input.cage_rep1_bam} {output.cage_rep1_bw}
        python bin/utils/bam_cov.py {input.cage_rep2_bam} {output.cage_rep2_bw}
        python bin/utils/bam_cov.py {input.dnase_bam} {output.dnase_bw}

        python bin/utils/basenji_data_read.py -b {input.exclusion_list_bed} -u sum -w 5000 {output.cage_rep1_bw} {input.windows_bed} {output.cage_rep1_h5}
        python bin/utils/basenji_data_read.py -b {input.exclusion_list_bed} -u sum -w 5000 {output.cage_rep2_bw} {input.windows_bed} {output.cage_rep2_h5}
        python bin/utils/basenji_data_read.py -b {input.exclusion_list_bed} -u sum -w 100 {output.dnase_bw} {input.windows_bed} {output.dnase_h5}
        python bin/utils/basenji_data_read.py -b {input.exclusion_list_bed} -u sum -w 100 {input.h3k4me3_bw} {input.windows_bed} {output.h3k4me3_h5}
        python bin/utils/basenji_data_read.py -b {input.exclusion_list_bed} -u sum -w 100 {input.h3k27ac_bw} {input.windows_bed} {output.h3k27ac_h5}

        """

rule process_3d_data:
    input:
        raw_data = rules.download_data.output.raw_data
    output:
        processed_3d = PROCESSED_DIR + "processed_3d.txt"
    shell:
        """
        python scripts/process_3d_data.py {input.raw_data} {output.processed_3d}
        """

rule process_dna_sequences:
    input:
        raw_data = rules.download_data.output.raw_data
    output:
        processed_dna = PROCESSED_DIR + "processed_dna.txt"
    shell:
        """
        python scripts/process_dna_sequences.py {input.raw_data} {output.processed_dna}
        """

rule process_annotations:
    input:
        raw_data = rules.download_data.output.raw_data
    output:
        processed_annotations = PROCESSED_DIR + "processed_annotations.txt"
    shell:
        """
        python scripts/process_annotations.py {input.raw_data} {output.processed_annotations}
        """

rule write_to_tfrecords:
    input:
        processed_1d = rules.process_1d_data.output.processed_1d,
        processed_3d = rules.process_3d_data.output.processed_3d,
        processed_dna = rules.process_dna_sequences.output.processed_dna,
        processed_annotations = rules.process_annotations.output.processed_annotations,
        unmappable_regions = rules.prepare_unmappable_regions.output.unmappable_regions
    output:
        tfrecords = TFRECORDS_DIR + "final_tfrecords.tfrecord"
    shell:
        """
        python scripts/write_to_tfrecords.py {input.processed_1d} {input.processed_3d} {input.processed_dna} {input.processed_annotations} {input.unmappable_regions} {output.tfrecords}
        """
