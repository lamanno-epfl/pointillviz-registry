# Tissue Permeability

Answers the question: **Which cells does one population seal off from the tissue surface or from another population?**

The tool lets the user select one population as a potential barrier and choose a destination for the remaining cells: either the exposed tissue surface or another population. It then identifies which cells can no longer reach it when the barrier is impassable.

To identify possible routes through the tissue, the tool first connects the cells in each section using a Delaunay triangulation. Three cells form such a triangle if a circle can pass through all three centers without enclosing another cell center.

Multi-source Dijkstra routing calculates the shortest network distance to a source with the selected population passable, *d(open)(i)*, and with it treated as impassable, *d(barrier)(i)*. *d(straight)(i)* is the straight-line distance from cell *i* to the nearest source cell, ignoring the network.

A cell is sealed if it loses its route when the selected population becomes impassable. For cells that remain reachable, *extra(i)* measures the added network distance in median cell spacings *s*, while *τ(i)* compares the remaining route with the straight-line distance:

$$
sealed(i) = [d_{open}(i) < ∞ and d_{barrier}(i) = ∞]
$$

$$
extra(i) = \frac{d_{barrier}(i) - d_{open}(i)}{s},   τ(i) = \frac{d_{barrier}(i)}{d_{straight}(i)}
$$

**Map.** The user can choose between three display modes:

- `Sealed compartments` colors only sealed cells. Dark colors indicate smaller compartments, while bright yellow indicates compartments containing more cells.
- `Extra path` colors only reachable cells whose route becomes longer. Dark colors indicate a small additional distance, while bright yellow indicates that the barrier adds many cell spacings.
- `Detour factor` colors only reachable cells with a valid *τ(i) > 1*. Dark colors indicate a route close to the direct distance, while bright yellow indicates a much longer detour.

**Plot.** The x-axis shows cell percentiles after ranking by *τ(i)*; at percentile *q*, the curve gives the detour factor that *q%* of the evaluated cells do not exceed. The y-axis shows *τ(i)*, the network distance divided by the straight-line distance.

The red curve shows detour factors when the selected population is impassable, while the dashed curve shows the same tissue with it passable. The grey band shows the results when the same number of randomly selected cells are treated as barriers. If the red curve stops before the right edge, the missing percentage corresponds to sealed cells; their detour factors are non-finite and therefore not shown.

**Table.** Each row summarizes one population with at least 25 eligible cells, and `Cells` gives its total cell count. For the selected barrier and destination, `Enclosed` and `% enclosed` show the number and proportion of eligible cells that lose their route. Among cells that remain reachable, `% detoured` shows how often the route becomes longer, while `Extra path where detoured` gives the median number of cell spacings added to those longer routes. `% enclosed, random barrier` gives the average expected with equally large random barriers.
