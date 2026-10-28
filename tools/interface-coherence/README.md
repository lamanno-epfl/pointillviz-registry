# Interface Coherence

Answers the question: **Which group pairs form long, continuous boundaries rather than many short, disconnected contacts?**

To identify boundaries between groups, the tool first connects the cells in each section using a Delaunay triangulation. Three cells form such a triangle if a circle can pass through all three centers without enclosing another cell center.

When two connected cells belong to different groups, the tool creates a small boundary piece between them, perpendicular to the line connecting their centers. Touching boundary pieces are combined into one segment when they separate the same two groups.

For each group pair, the tool measures the lengths *s(1), …, s(q)* of its separate boundary segments and summarizes them as:

$$
Lw = \frac{\sum_{s}s²}{\sum_{s}s}
$$

Because segment length is used as a weight, long connected boundaries contribute more than short contacts. The same value is calculated after repeatedly shuffling the group labels while keeping cell positions fixed. Coherence is then:

$$
C(A,B) = log₂(\frac{Lw(observed)}{Lw(shuffled)})
$$

**0** indicates the shuffled expectation, while **+1** indicates an observed value about **2×** as large.

**Map.** Cells along group boundaries are colored by their local coherence value. This is calculated similarly to *C(A,B)*, but uses the lengths of the specific segments touching the cell instead of the pair-wide observed *Lw*. Cells on longer, more continuous segments than expected appear yellow, whereas cells on shorter, more fragmented segments appear blue.

**Matrix.** Both axes list all groups, and each square shows the coherence score *C(A,B)* defined above for one group pair. Yellow indicates longer, more continuous boundaries than expected overall, while blue indicates shorter, more fragmented boundaries.

**Table.** Each row describes one group pair and gives the cell count for both groups, along with `Coherence length` and `log2 coherence`, which correspond to *Lw* and *C(A,B)* defined above. The table also reports `Interface length`, the total length of all boundary segments; `Segments`, the number of disconnected segments making up this boundary; and `Log2 contact amount`, the observed-versus-shuffled comparison based on this total boundary length rather than *Lw*.
