# BANKSY niches

Answers the question: **Which tissue domains, or niches, does a section divide into, judged by what each cell and its neighbours express?**

The tool runs [BANKSY](https://github.com/prabhakarlab/Banksy_py) (Singhal, Chou, Lee et al., *Nature Genetics* 2024) with the authors' own Python package, **imported from PyPI as they ship it** (`pybanksy` 1.3.5). It calls the package's functions in the order of the authors' notebooks, with their values, and holds no copy of their method:

1. normalises the counts and keeps the most variable genes (`normalize_total`, `filter_hvg`);
2. builds a spatial graph of each cell's nearest neighbours (`initialize_banksy`);
3. builds the BANKSY matrix: each cell's expression, its neighbours' weighted mean and, by default, the azimuthal Gabor filter (AGF), which responds where expression changes across a neighbourhood (`generate_banksy_matrix`);
4. reduces it with PCA (`pca_umap`) and clusters the components with Leiden (`run_Leiden_partition`).

Called directly on the same counts, with no code of this tool, the authors' functions give the same niches, cell for cell, on Xenium 5K sections of 135,231 and 235,803 cells.

**Map.** Every cell painted by its niche, numbered N1, N2, … from the largest.

**Matrix.** Each niche's cell-type composition, or its marker genes if the dataset has no cell types.

**Table.** Each niche's size, marker genes and main cell types, with a download of every cell's niche.

**Several sections.** A cell's neighbours always come from its own section. **Z-score** sets the rest. Per section, the default, each section gets its own z-scores and the matrices are joined before the PCA, as in the authors' multi-sample notebook. Across all sections, the package runs once over every cell, with the sections set apart in space so that no neighbourhood crosses between them.

**What the package does as shipped.** The tool leaves these as they are:

- The AGF is not centred: `generate_banksy_matrix` passes its `variance_balance` argument (False) into the `center` slot of `create_nbr_matrix`. Results can therefore differ from the authors' R package.
- Ranked weights fail with the AGF, so the tool refuses that combination with a sentence.
- With scaled Gaussian or reciprocal weights, cells on the same spot make the package lose the whole neighbourhood block without a word. The tool refuses, and names the uniform weights.
- The package's other clustering, `mclust`, needs R and `rpy2`, and is not offered.

**Cost.** 135,231 cells take about 6 minutes and 21 GB at 2,000 variable genes, and a selection of 20,000 cells about 30 s. If the memory is not there, the tool says so before it starts, and which control to lower.

`pybanksy` is under the GPL-3.0 licence, and so is this tool.
