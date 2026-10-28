# {{NAME}}

Paints every cell by the cell type that is most common among its nearest neighbours.

**Map.** One colour per cell type. A cell is painted with the type most of its neighbours have, whatever its own type. Cells with no position, or with no labelled neighbour, are left unpainted.

**Summary.** How many cells were labelled, and the most frequent neighbourhood type.

## Parameters

- **Neighbours.** How many nearest cells make up a neighbourhood. Neighbours are found within each cell's own section. When types tie, the one listed first in the dataset wins.

## Limitations

The neighbourhood is a fixed number of cells, not a fixed distance, so it covers a larger area where cells are sparse.
