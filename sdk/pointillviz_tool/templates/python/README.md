# {{NAME}}

Paints every cell by its total transcript count, and leaves cells below a minimum count unpainted.

**Map.** Transcripts per cell. Unpainted cells are below the minimum count.

**Summary.** How many cells reach the minimum, and the median count among them.

## Parameters

- **Minimum counts.** Cells with fewer transcripts than this are left unpainted. At 0, every cell is painted.

## Limitations

The total is the dataset's own per-cell count. It is not normalised, so it reflects both biology and capture efficiency.
