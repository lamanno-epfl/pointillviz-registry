# Mclust niches

Answers the question: **Which kinds of surroundings do the cells sit in, judged by the cell types around them?**

Describes every cell by the mix of cell types among its nearest cells, and groups cells with similar mixes into niches with a Gaussian mixture model (mclust). It is a native R tool: the analysis is plain R, and PointillViz hands it the cell types and the neighbours.

## What you get

- **Map.** Every cell painted by its niche, one colour per niche. Niches are numbered N1, N2, … from the largest, and the numbers carry no other order. Cells with no labelled neighbour, or outside a selection, are left unpainted.
- **Table.** Each niche's size and its three main cell types, as shares of the niche's own labelled cells.

## Parameters

- **Neighbours (k).** How many nearest cells make up the mix around a cell. Neighbours are found within each cell's own section.
- **Niches.** How many niches the mixture model makes. If two niches hold the same cell types, ask for fewer.
- **Random seed.** Chooses the 2,000 cells the mixture is fitted to. Change it to see how stable the niches are.

## Method

1. For every cell, the share of each cell type among its *k* nearest cells. Unlabelled neighbours are not counted.
2. `prcomp` reduces the shares to at most 10 principal components. Directions with almost no spread are dropped: the shares sum to one, so the last direction carries nothing.
3. `mclust::Mclust` fits a Gaussian mixture with one covariance shared by all niches (model "EEE") to 2,000 cells drawn with the seed. Fitting on a sample keeps the run short, because mclust starts from a hierarchical clustering whose cost grows with the square of the number of cells.
4. The fitted mixture gives every cell its most likely niche (`predict`).

On a Xenium 5K mouse colon section of 135,000 cells, the default niches follow the layers of the gut wall: stem and transit-amplifying cells at the crypt base, mature enterocytes at the surface, and smooth muscle with interstitial cells of Cajal and glia in the muscle layers.

## Citation

Scrucca L, Fop M, Murphy TB, Raftery AE. mclust 5: clustering, classification and density estimation using Gaussian finite mixture models. *The R Journal* 8(1), 289–317 (2016).
