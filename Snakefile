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
        cage_rep1_bam = "~/hierarchical/data/encode/cage/{wildcards.cell_type}_cage_rep1.bam"
        cage_rep2_bam = "~/hierarchical/data/encode/cage/{wildcards.cell_type}_cage_rep2.bam"
        dnase_bam = "~/hierarchical/data/encode/dnase/{wildcards.cell_type}_dnase.bam"
        h3k4me3_bw = "~/hierarchical/data/encode/h3k4me3/{wildcards.cell_type}_h3k4me3.bigWig"
        h3k27ac_bw = "~/hierarchical/data/encode/h3k27ac/{wildcards.cell_type}_h3k27ac.bigWig"
        
        gencode_annotations = "~/hierarchical/data/genome_annotations/gencode/gencode.v46.annotation"

        unmappable_bed = "~/hierarchical/data/genome_annotations/unmappable_regions/unmappable_regions_plus_exclusion_list_with_repeats.bed"
        unmappable_bw = "~/hierarchical/data/genome_annotations/unmappable_regions/k24.Umap.MultiTrackMappability.bw"
        unmappable_macro = "~/hierarchical/data/genome_annotations/unmappable_regions/unmap_macro.bed"

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

        cd data/genome_annotations/exclusion_list
        wget https://www.repeatmasker.org/genomes/hg38/RepeatMasker-rm405-db20140131/hg38.fa.out.gz
        gunzip hg38.fa.out.gz
        grep -i satellite hg38.fa.out | awk -v OFS='\t' '{ print $5,$6,$7 }' > satellite_repeats.bed
        wget https://www.encodeproject.org/files/ENCFF356LFX/@@download/ENCFF356LFX.bed.gz
        gunzip ENCFF356LFX.bed
        mv ENCFF356LFX.bed exclusion_list.bed
        cat exclusion_list.bed satellite_repeats.bed | bedtools sort | bedtools merge > exclusion_list_with_repeats.bed
        
        cd ../unmappable_regions
        wget http://hgdownload.soe.ucsc.edu/gbdb/hg38/hoffmanMappability/k24.Umap.MultiTrackMappability.bw
        python ~/hierarchical/bin/utils/unmappable_bed.py -l 32 k24.Umap.MultiTrackMappability.bw > unmap_k24_l32.bed
        wget https://raw.githubusercontent.com/calico/basenji/9e1c2e2f5b1b37ad11cfd2a1486d786d356d78a5/tutorials/data/unmap_macro.bed
        cat ../exclusion_list/exclusion_list_with_repeats.bed unmap_k24_l32.bed | awk 'BEGIN {OFS="\t"} {print $1,$2,$3}' | bedtools sort | bedtools merge > {output.unmappable_bed}



        """






rule prepare_unmappable_regions:
    input:
        raw_data = rules.download_data.output.raw_data
    output:
        unmappable_regions = UNMAPPABLE_DIR + "unmappable_regions.txt"
    shell:
        """
        python scripts/prepare_unmappable_regions.py {input.raw_data} {output.unmappable_regions}
        """

rule process_1d_data:
    input:
        raw_data = rules.download_data.output.raw_data
    output:
        processed_1d = PROCESSED_DIR + "processed_1d.txt"
    shell:
        """
        python scripts/process_1d_data.py {input.raw_data} {output.processed_1d}
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
