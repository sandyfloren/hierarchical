
cd hierarchical/data

# Exclusion list

mkdir -p genome_annotations && cd -p genome_annotations
mkdir -p exclusion_list && cd exclusion_list
wget https://www.repeatmasker.org/genomes/hg38/RepeatMasker-rm405-db20140131/hg38.fa.out.gz
gunzip hg38.fa.out.gz
grep -i satellite hg38.fa.out | awk -v OFS='\t' '{ print $5,$6,$7 }' > satellite_repeats.bed
wget https://www.encodeproject.org/files/ENCFF356LFX/@@download/ENCFF356LFX.bed.gz
gunzip ENCFF356LFX.bed
mv ENCFF356LFX.bed exclusion_list.bed
cat exclusion_list.bed satellite_repeats.bed | bedtools sort | bedtools merge > exclusion_list_with_repeats.bed
cd ..


# Unmappable regions
# Unmappable >32 bp regions and large gaps (from basenji repo)
# TODO: check where file unmap_macro.bed actually comes from
# TODO: use up-to-date assembly gaps file from UCSC?

mkdir -p unmappable_regions && cd unmappable_regions
wget http://hgdownload.soe.ucsc.edu/gbdb/hg38/hoffmanMappability/k24.Umap.MultiTrackMappability.bw
python utils/unmappable_bed.py -l 32 ../data/genome_annotations/unmappable_regions/k24.Umap.MultiTrackMappability.bw > ../data/genome_annotations/unmappable_regions/unmap_k24_l32.bed
wget https://raw.githubusercontent.com/calico/basenji/9e1c2e2f5b1b37ad11cfd2a1486d786d356d78a5/tutorials/data/unmap_macro.bed
cat ../exclusion_list/exclusion_list_with_repeats.bed unmap_k24_l32.bed | awk 'BEGIN {OFS="\t"} {print $1,$2,$3}' | bedtools sort | bedtools merge > unmappable_regions_plus_exclusion_list_with_repeats.bed
cd ..






# ENCODE data
# all these files are in hg38
# TODO: collapse cell type folders, create master file of accessions with descriptions

cd encode
mkdir -p atac && cd atac
mkdir -p heart_left_ventricle && cd heart_left_ventricle
wget ftp://ftp.ncbi.nlm.nih.gov/geo/series/GSE215nnn/GSE215505/suppl/GSE215505_ENCFF811VNG_IDR_thresholded_peaks_GRCh38.bed.gz
gunzip GSE215505_ENCFF811VNG_IDR_thresholded_peaks_GRCh38.bed.gz
cd ../..

mkdir -p cage && cd cage
mkdir -p k562 && cd k562
wget https://www.encodeproject.org/files/ENCFF754FAU/@@download/ENCFF754FAU.bam
wget https://www.encodeproject.org/files/ENCFF366MWI/@@download/ENCFF366MWI.bam
cd ../..
mkdir -p dnase && cd dnase
mkdir -p k562 && cd k562
wget https://www.encodeproject.org/files/ENCFF257HEE/@@download/ENCFF257HEE.bam
cd ../..
mkdir -p h3k4me3 && cd h3k4me3
mkdir -p k562 && cd k562
wget https://www.encodeproject.org/files/ENCFF253TOF/@@download/ENCFF253TOF.bigWig
cd ../..
mkdir -p h3k27ac && cd h3k27ac
mkdir -p k562 && cd k562
wget https://www.encodeproject.org/files/ENCFF381NDD/@@download/ENCFF381NDD.bigWig
cd ../..


# Hi-C data

mkdir -p ncbi_geo && cd ncbi_geo
mkdir -p hic && cd hic
mkdir -p k562 && cd k562
# contact matrices (maybe don't use these?)
wget https://ftp.ncbi.nlm.nih.gov/geo/series/GSE63nnn/GSE63525/suppl/GSE63525_K562_intrachromosomal_contact_matrices.tar.gz 
tar -xvzf GSE63525_K562_intrachromosomal_contact_matrices.tar.gz 

# HiC combined
# We are going to use the MAPQ>30 files. According to ENCODE, this is preferred for generating .hic files.
wget https://ftp.ncbi.nlm.nih.gov/geo/series/GSE63nnn/GSE63525/suppl/GSE63525_K562_combined_30.hic

# install juicer (needed for HiCDCPlus)
cd ~
git clone https://github.com/theaidenlab/juicer.git
cd hierarchical/bin
mkdir -p juicer & cd juicer
ln -s ~/juicer/CPU scripts
cd scripts/common
wget https://github.com/aidenlab/Juicebox/releases/download/v2.18.00/juicer_tools.2.18.00.jar
ln -s juicer_tools.2.18.00.jar juicer_tools.jar
cd ../..
mkdir -p references
ln -s /pollard/data/vertebrate_genomes/human/hg38/hg38/hg38.fa references/hg38.fa
ln -s /pollard/data/vertebrate_genomes/human/hg38/hg38/hg38.fa.fai references/hg38.fa.fai

# Run HiC-DC+
cd ../data/ncbi_geo/hic
cd k562
Rscript --no-save ~/hierarchical/bin/utils/hicdcplus.R --map_bw ~/hierarchical/data/genome_annotations/unmappable_regions/k24.Umap.MultiTrackMappability.bw GSE63525_K562_combined_30.hic GSE63525_K562_combined_result.txt
cd ../../..


mkdir -p graphreg_data/1d/k562 
mkdir -p graphreg_data/3d/k562
mkdir -p graphreg_data/seqs
mkdir -p graphreg_data/tfr/k562
mkdir -p graphreg_data/preds/k562



# Create graphs
cd ~/hierarchical
python bin/utils/create_graphs.py -v -d data/graphreg_data/3d/k562 data/ncbi_geo/hic/k562/GSE63525_K562_combined_result.txt data/graphreg_data/seqs/genome_bins.bed


# Get TSSs

mkdir -p ~/hierarchical/data/genome_annotations/gencode && cd ~/hierarchical/data/genome_annotations/gencode
wget https://ftp.ebi.ac.uk/pub/databases/gencode/Gencode_human/release_46/gencode.v46.annotation.gtf.gz
gunzip gencode.v46.annotation.gtf.gz
awk '{if ($3 == "transcript") {print $0}}' gencode.v46.annotation.gtf > gencode.v46.annotation_transcripts.gtf
awk '{if ($3 == "gene") {print $0}}' gencode.v46.annotation.gtf > gencode.v46.annotation_genes.gtf


# Process TSSs
cd ~/hierarchical/data/graphreg_data/tss
# only proetin coding genes for now
python ~/hierarchical/bin/utils/tss.py --protein ../../genome_annotations/gencode/gencode.v46.annotation.gtf ../seqs/genome_bins.bed




# write TFRecords
cd ~/hierarchical/data/graphreg_data/tfr
# no unmap for now
python ~/hierarchical/bin/utils/graphreg_data_write.py /pollard/data/vertebrate_genomes/human/hg38/hg38/hg38.fa ../../seqs/windows_6Mb_no_gaps.bed ../../1d/k562 ../../3d/k562 ../../tss