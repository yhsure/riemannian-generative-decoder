![RGD logo](data/frontpage_rgd.png)
# Riemannain generative decoder

## Abstract
Riemannian representation learning typically relies on approximating densities on chosen manifolds. This involves optimizing difficult objectives, potentially harming models. To completely circumvent this issue, we introduce the Riemannian generative decoder which finds manifold-valued maximum likelihood latents with a Riemannian optimizer while jointly training a decoder network. By discarding the encoder, we vastly simplify the manifold constraint compared to current approaches which can often only handle few specific manifolds. We validate our approach on three case studies — a synthetic branching diffusion process, human migrations inferred from mitochondrial DNA, and cells undergoing a cell division cycle — each showing that learned representations respect the prescribed geometry and capture intrinsic non-Euclidean structure. Our method requires only a decoder, is compatible with existing architectures, and yields interpretable latent spaces aligned with data geometry. 

## Running the project
To reproduce results from the paper, follow the steps below:

1. Cell cycle in human fibroblasts [1]
   - Download the h5ad-formatted file from [Zenodo][hfibro].
   - Move the file to the `data/cellcycle/` directory.
   - Run the `1) CellCycle.ipynb` notebook to generate results.

2. Synthetic branching diffusion process [2]
   - No data download required.
   - Run the `2) BranchingDiffusion.ipynb` notebook to generate results.

3. Human mitochondrial DNA [3]
   - No data download required.
   - Run the `3) hmtDNA*.ipynb` notebooks to generate results.
   - For instructions on scraping new mitochondrial sequences, see [Scraping new mitochondrial sequences](#scraping-new-mitochondrial-sequences) below.


### Scraping new mitochondrial sequences
In order to make an updated scrape of mitochondrial sequences, run the following Python script:
```python
from Bio import Entrez
from tqdm import tqdm

Entrez.email = "your_email@gmail.com"
query = '(00000015400[SLEN] : 00000016700[SLEN]) AND "Homo"[Organism] AND mitochondrion[FILT] NOT (unverified[All Fields] OR ("Homo sapiens subsp. \'Denisova\'"[Organism] OR Homo sp. Altai[All Fields]) OR ("Homo sapiens subsp. \'Denisova\'"[Organism] OR Denisova hominin[All Fields]) OR neanderthalensis[All Fields] OR heidelbergensis[All Fields] OR consensus[All Fields])'

handle = Entrez.esearch(db="nuccore", term=query, retmax=100000)
record = Entrez.read(handle)
id_list = record["IdList"]
with open("seqs.fasta", "w") as outfile:
    for start in tqdm(range(0, len(id_list), 500), desc="Downloading sequences"):
        fetch_ids = id_list[start:start+500]
        handle = Entrez.efetch(db="nuccore", id=fetch_ids, rettype="fasta", retmode="text")
        records = handle.read()
        outfile.write(records)
```
Then, use *haplogrep3* [4] to annotate the sequences using either an RSRS or rCRS tree: 
```bash
$ haplogrep3 classify --in "seqs.fasta" --out seqs_rsrs.txt --tree phylotree-rsrs@17.0 --extend-report
$ haplogrep3 classify --in "seqs.fasta" --out seqs_rcrs.txt --tree phylotree-fu-rcrs@1.2 --extend-report
```
This generates a file with mutations compared to a reference sequence, as well as more metadata — see the [*haplogrep3* documentation][haplogrep] for details. Additional metadata (e.g., geographic location) can be added during a second GenBank scrape; merging such data with the *haplogrep3* output produces the zipped files `data/hmtDNA/63k_out_rcrs.txt.gz` and `data/hmtDNA/63k_out_rsrs.txt.gz`. Upon approval, this pipeline will be made available as a standalone script, and a resulting snapshot will be published via Zenodo.

## Requirements
The code has been tested with Python 3.11.11 using the packages from `requirements.txt`. Use the locally provided geoopt 0.5.0 (including a sphere origin function) with `$ unzip data/geoopt.zip && pip install -e data/geoopt`.

## Bibliography
1. A. Riba. "Cell cycle gene regulation dynamics revealed by rna velocity and deep learning", 2021. URL
https://doi.org/10.5281/zenodo.4719436. Dataset. 
2. E. Mathieu, C. Le Lan, C. J. Maddison, R. Tomioka, and Y. W. Teh. "Continuous hierarchical
representations with poincaré variational auto-encoders". Advances in neural information processing
systems, 32, 2019.
3. MITOMAP: A Human Mitochondrial Genome Database. http://www.mitomap.org, 2023. Database.
4. S. Schönherr, H. Weissensteiner, F. Kronenberg, and L. Forer. Haplogrep 3-an interactive haplogroup
classification and analysis platform. Nucleic acids research, 51(W1):W263–W268, 2023. Software.

[hfibro]: https://zenodo.org/records/4719436/files/velocity_anndata_human_fibroblast_DeepCycle_ISMARA.h5ad?download=1
[branching]: https://github.com/emilemathieu/pvae/blob/master/pvae/datasets/datasets.py
[haplogrep]: https://haplogrep.readthedocs.io/

## Citation
If you use this work, please cite:

```bibtex
redacted
```