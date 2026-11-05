# Cell-type contact network

Answers the question: **Which cell types touch which, and which of those contacts are more common, or rarer, than the types' abundance predicts?**

The tool triangulates each section's cells with a Delaunay triangulation, from triangles in 2D and tetrahedra in 3D, and counts every edge as one contact between two neighbouring cells. No contact crosses sections.

**The contact matrix.** For each cell, the tool tallies the types of its neighbours, and adds the tallies up by type into a count matrix *C*. *C* is symmetric, and a contact between two cells of one type counts twice on the diagonal, once from each cell. Each row is then normalised:

$$
W_{ij} = \frac{C_{ij}}{\sum_{k} C_{ik}}
$$

*Wᵢⱼ* is the fraction of type *i*'s contacts that are with type *j*. Each row sums to 1, and *W* is not symmetric, because *i → j* and *j → i* are divided by different totals. A common type touches everything simply because it is everywhere, so the enrichment corrects for abundance:

$$
E_{ij} = \log_{2} \frac{W_{ij} + 10^{-4}}{p_{j} + 10^{-4}}
$$

*pⱼ* is the fraction of all cells that are type *j*. *E* > 0 means *i* touches *j* more than *j*'s abundance predicts, and *E* < 0 less.

**Several sections.** *W* is computed for each section. To pool them, each row is averaged across the sections, weighted by how many cells of type *i* each section has, and a section where type *i* is absent or has no contact is skipped. *E* then uses the pooled *W* and the fraction of all the run's cells that are type *j*.

On a Xenium section of the mouse maternal–fetal interface (235,803 cells, 19 types), pairs of types in one layer of the tissue average *E* = +0.63, and pairs in different layers −3.12. The myometrium gives the labyrinth's first syncytiotrophoblast layer 0.06% of its contacts (*E* = −6.9).

## What you get

**The network.**

- **Nodes.** One per cell type, at most the number the form asks for, the most abundant first. A node's radius grows as the square root of its cells, so its area grows with them, down to a smallest size for the rarest types. The largest node takes the size the figure has room for in this run, so compare nodes with the scale in the key. A node keeps the type's colour on the map, and its name stands beside it.
- **Places.** A node's place comes from a force-directed layout of the whole dataset's pooled contacts, with no distance limit, whatever is isolated, selected or set in the form. So a type stands in the same place in every run on one annotation, and a type that is absent from a run simply has no node there.
- **Arrows.** An arrow *i → j* is drawn where *Wᵢⱼ* reaches the cutoff. Its width grows as √*W*, and its colour is *E*, from blue (depleted) to red (enriched), the same for ±3 and beyond. The two directions of a pair are separate curved lines, each bending to its own right of travel, and they usually differ: read each arrow from the type whose share it is.
- **Rings.** The ring around a node is its self-contact: the further the ring stands from the node, the larger *Wᵢᵢ*. Its colour is *Eᵢᵢ*. No ring means no contact within the type.

**The matrix.** The same *W* and *E*, with no cutoff, for the same types. Each square's colour is *E*, on the same scale, and its number is *W* in percent, from 1% up, for as long as the squares are large enough to hold numbers. Grey means no contact (*W* = 0), and the outlined diagonal holds the self-contacts. A row shows what that type touches, and a column what touches it. Each row of *W* sums to 100% over all the types, shown or not, so the numbers in a row can fall short of it.

Rows and columns share one order, which seats types with similar contacts together: hierarchical clustering (average linkage) of (*W* + *W*ᵀ) / 2, its leaves ordered. Rows are compared by whom they touch, whatever their scale: each is scaled to length 1, and its diagonal counts no more than its largest other share. The order follows this run's *W*, so it can differ from one section to another.

**Table.** Every contact between the types shown, largest *E* first: *W* in percent, *E* and *C*.

**Downloads.**

- `contact-network.svg` and `contact-matrix.svg`: the two figures.
- `contact-matrices.csv`: one row per ordered pair of types in contact, for every type, shown or not, with `section`, `from`, `to`, `C`, `W` and `E`. A pair that is not listed has no contact. With several sections, the rows of `pooled` come first, where `C` is the sum over the sections, and then those of each section, unless they would not fit the output limit; the summary says so, and isolating a section gives its own.
- `contact-types.csv`: one row per type present, with its `cells`, its `fraction` *p*, its `contacts` (the row's sum of *C*), and `network_x` and `network_y`, its place in the figure in px.

A CSV of more than 1 MB is gzipped (`.csv.gz`), which `pandas` and R read as they are.

**Summary.** The cells, contacts and arrows of this run, the lengths of the contacts, the most enriched and depleted arrows, and what was left out and why.

## Parameters

- **Cell-type column.** The annotation whose groups are the nodes: cell types, unless the dataset has none or you choose another column.
- **Longest contact (microns).** Drops contacts longer than this, so that cells across a gap do not count as touching. 0, the default, keeps every contact. A dataset in µm or mm is used as it stands, and one in pixels through the pixel size of an image layer that covers its cells. A dataset that records no physical size reads this value in its own coordinates, and the figures and the summary say so. The summary gives the median contact, as a guide. On a cell-segmented section it is about 8 to 14 µm, and about 1 to 3% of the contacts are longer than three times that.
- **Arrow cutoff (%).** The smallest *W* that gets an arrow, 4% by default. Only the arrows follow it, not the matrix, the table or the downloads. Above every share, the figure says what the largest share is.
- **Cell types shown.** How many of the most abundant types are shown, 30 by default, in the network, the matrix and the table. The others still count in every share, and the downloads hold them.

Nothing else is a choice: the layout has no seed, and the same contacts give the same places.

## Sections and selections

**Sections.** Each section is triangulated on its own, so two sections that share coordinates, or sit side by side on the canvas, never touch. With nothing isolated, the figures pool every section. Isolate a section for its own figures, in the same places. A section with too few cells for a triangulation, or with all its cells stacked or on one line or plane, is left out, and the summary names it; with no other section, the tool refuses and says why.

**Selections.** With cells selected, the run counts only the contacts between two selected cells, over the triangulation of the whole section, so the outline of a selection invents no contact and hides none between the cells it holds. The cell counts and *p* are then those of the selected cells.

**Cells with no type, or no place.** A cell with no coordinates is left out. A cell with no level of the column stays in the triangulation, so that its neighbours do not touch across it, and its contacts are not counted. *p* counts only the cells with a type.

**Stacked cells.** Qhull sets aside a cell that sits exactly on another. It touches that one cell, and no other.

**The layout.** It is computed once per dataset and annotation, from every section, and is kept in the dataset's `plugin_cache/` folder, the one place a tool may write in a dataset, in a small file named after a hash of the positions and the types. If it cannot be written, each run computes the layout again, and it comes out the same.

## Limitations

**Delaunay edges are geometric neighbours, not measured contact.** Two cells joined by an edge have no other cell between them, as far as their centres say. They need not touch: the triangulation knows no membranes, and each cell has about 6 neighbours in 2D and about 15 in 3D, whatever its real surroundings. A section's cells are centres in a thin slice, and a section cannot show the cells above and below it.

**Long edges cross gaps.** The triangulation fills the whole convex hull of a section, so cells on either side of a hole, a tear or a bay of empty slide are joined, unless a limit is set. These edges are few, but on small sections and rare types they can matter.

**E is a descriptive ratio.** It has no p-value and no confidence interval. A type of a few cells can reach any *E* by chance, so read it with its number of contacts, *C*, in the table. The pseudocount of 10⁻⁴ keeps *E* finite and limits it for types rarer than about 0.01%. Sections pool by the abundance in the whole run, so a type confined to one section looks enriched against the pool wherever its neighbours are confined to the same section.

**The figures show the most abundant types.** A rare type's contacts are in the downloads, and it can be shown by raising the number of types or isolating a section.

**The annotation decides what can be seen.** A column with a single group in each section, such as a tissue label on a tissue microarray, has no contacts between groups, and the tool says so rather than draw an empty network. A column of more than 1,000 groups is refused.

**Cost.** On a shared server, a section of 235,803 cells takes about 2 s and 0.25 GB, and one of 465,534 cells about 4 s and 0.4 GB. 708,889 cells in 9 sections take about 5 s and 0.4 GB, and one of those sections under 2 s. A 3D section of 108,486 cells takes about 6 s and 0.4 GB. The first run on a dataset and annotation adds the layout: about 1 s at 20 types, and about 7 s at 300.
