# Cell-type contact network

Answers the question: **Which cell types touch which, and which of those contacts are more common, or rarer, than the types' abundance predicts?**

The tool runs a Delaunay triangulation on each section's cells, in 2D or 3D, and counts every edge as one contact between two neighbouring cells. Tallying each cell's neighbours by type gives a count matrix *C*, whose rows are then normalised:

$$
W_{ij} = \frac{C_{ij}}{\sum_{k} C_{ik}}
$$

*Wᵢⱼ* is the fraction of type *i*'s contacts that are with type *j*. It is not symmetric, because *i → j* and *j → i* are divided by different totals. The enrichment corrects for abundance, where *pⱼ* is the fraction of all cells that are type *j*:

$$
E_{ij} = \log_{2} \frac{W_{ij} + 10^{-4}}{p_{j} + 10^{-4}}
$$

*E* > 0 means *i* touches *j* more than *j*'s abundance predicts, and *E* < 0 less. With several sections, *W* is computed per section and each row is averaged across them, weighted by how many cells of type *i* each has.

**Network.** One node per cell type, its area growing with its number of cells. Its place comes from a layout of the whole dataset, so nodes stay put between sections. An arrow *i → j* appears where *Wᵢⱼ* reaches the cutoff, with its width growing as √*W* and its colour showing *E*, from blue (depleted) to red (enriched). The ring around a node is its self-contact: the further the ring stands from the node, the larger *Wᵢᵢ*.

**Matrix.** The same *W* and *E* with no cutoff. Colour shows *E*, the numbers are *W* in percent, and grey means no contact. Types with similar contacts sit together.

**Table and downloads.** Every contact between the types shown, the two figures as SVG, and the numbers for all types as CSV.

Delaunay edges are geometric neighbours, not measured contact, and *E* has no p-value, so read it together with the contact count *C* in the table. Cells across a gap are joined unless **Longest contact** is set.
