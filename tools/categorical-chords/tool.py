"""Categorical relationships with exact, section-stratified expectations."""
import numpy as np
from pointillviz_tool import PluginError


def run(ds, params):
    mode = params['mode']
    fields = ds.categorical_fields()
    field = params.get('field') or fields[0]['name']
    labels, codes = ds.categorical(field)
    eligible = codes >= 0
    if ds.section_filter is not None:
        eligible &= ds.section == ds.section_filter
    if ds.subset is not None:
        mask = np.zeros(ds.n_cells, dtype=bool)
        mask[ds.subset] = True
        eligible &= mask
    relation = mode == 'annotations'
    other = params.get('other_field') or field
    right, right_codes = ds.categorical(other) if relation else (labels, codes)
    if relation:
        eligible &= right_codes >= 0
    else:
        if (ds.meta.get('spatial') or {}).get('mode') == 'none':
            raise PluginError('no_coordinates', 'Spatial contacts need coordinates. Choose Annotation relationships instead.')
        eligible &= np.isfinite(ds.coords).all(axis=1)
    if not eligible.any():
        raise PluginError('empty_selection', 'No labelled cells remain in this analysis scope.')
    if len(labels) * len(right) > 10000:
        raise PluginError('too_many_levels', 'Choose coarser annotations; the full matrix is limited to 10,000 entries.')
    obs = np.zeros((len(labels), len(right)), dtype=float)
    expected = np.zeros_like(obs)
    total = 0
    for section in np.unique(ds.section[eligible]):
        rows = np.flatnonzero(eligible & (ds.section == section))
        a = codes[rows]
        n = len(rows)
        counts = np.bincount(a, minlength=len(labels)).astype(float)
        if relation:
            b = right_codes[rows]
            obs += np.bincount(a * len(right) + b, minlength=obs.size).reshape(obs.shape)
            expected += np.outer(counts, np.bincount(b, minlength=len(right))) / n
            total += n
        elif n > 1:
            from scipy.spatial import cKDTree
            tree = cKDTree(ds.coords[rows])
            # Bound work BEFORE allocating pairs. No random downsampling.
            pair_count = (int(tree.count_neighbors(tree, params['radius'])) - n) // 2
            if pair_count > 5_000_000:
                raise PluginError('too_many_pairs', 'More than 5 million pairs in a section. Reduce the radius or use a smaller selection.')
            pairs = tree.query_pairs(params['radius'], output_type='ndarray')
            flat = np.bincount(a[pairs[:, 0]] * len(labels) + a[pairs[:, 1]], minlength=obs.size).reshape(obs.shape)
            obs += flat + flat.T
            # Random relabelling WITHOUT replacement, separately in each section.
            probability = (np.outer(counts, counts) - np.diag(counts)) / (n * (n - 1))
            expected += 2 * pair_count * probability
            total += pair_count
        ds.progress(0.6, 'Counting categorical relationships within sections')
    # Within-type contacts have two equal directed entries on the diagonal.
    if not relation:
        np.fill_diagonal(obs, np.diag(obs) / 2)
        np.fill_diagonal(expected, np.diag(expected) / 2)
    score = np.full_like(obs, np.nan)
    measured = expected > 0
    score[measured] = np.log2((obs[measured] + 1) / (expected[measured] + 1))
    max_nodes = int(params['max_nodes'])
    left_counts = np.bincount(codes[eligible], minlength=len(labels))
    right_counts = np.bincount(right_codes[eligible], minlength=len(right))
    def largest(counts, limit):
        return sorted((i for i in range(len(counts)) if counts[i]), key=lambda i: (-counts[i], i))[:limit]
    left_idx = largest(left_counts, max_nodes // 2 if relation else max_nodes)
    right_idx = largest(right_counts, max_nodes // 2) if relation else left_idx
    def node(i, name, levels):
        descriptor = next((f for f in fields if f['name'] == name), {})
        colors = descriptor.get('colors') or []
        c = colors.get(levels[i]) if isinstance(colors, dict) else colors[i] if i < len(colors) else None
        return {'label': str(levels[i]), 'color': c, 'group': descriptor.get('label') or name}
    nodes = [node(i, field, labels) for i in left_idx]
    if relation:
        nodes += [node(i, other, right) for i in right_idx]
    links = []
    for x, i in enumerate(left_idx):
        for y, j in enumerate(right_idx):
            if not relation and (y < x or (i == j and not params['self_links'])):
                continue
            if obs[i, j] < params['min_count']:
                continue
            links.append({'source': x, 'target': len(left_idx) + y if relation else y,
                          'value': float(obs[i, j]), 'score': float(score[i, j])})
    links.sort(key=lambda x: (-x['value'], x['source'], x['target']))
    links = links[:int(params['max_links'])]
    if not links:
        raise PluginError('no_links', 'No relationships pass these controls. Reduce minimum support, increase radius for contacts, or include self-links.')
    rows = []
    for i in range(len(labels)):
        for j in range(len(right)):
            if (not relation and j < i) or not np.isfinite(score[i, j]):
                continue
            rows.append([labels[i], right[j], int(obs[i, j]), float(expected[i, j]), float(score[i, j])])
    rows.sort(key=lambda r: (-r[2], r[0], r[1]))
    shown = int(sum(l['value'] for l in links))
    note = (f'{shown:,} of {total:,} {"cells" if relation else "unordered pairs"} shown in {len(links)} ribbons. '
            'Node/link limits and support filters affect the chord only; the heatmap retains all levels. '
            'Width = count; color = log₂ observed/expected (blue depleted, orange enriched). '
            'Expected counts are calculated within each section. ' +
            ('Both annotations refer to the same cells; ribbons are not spatial contacts.' if relation else
             'Both endpoints must be inside the selected scope. Proximity is not evidence of signaling.'))
    return {
        'chord': {'title': 'Annotation relationships' if relation else 'Spatial contact chords',
                  'nodes': nodes, 'links': links, 'directed': False,
                  'units': 'cells' if relation else 'pairs', 'note': note},
        'matrix': {'title': 'Full log₂ observed/expected', 'rows': list(labels), 'cols': list(right),
                   'values': score.tolist(), 'diverging': True, 'colormap': 'icefire'},
        'table': {'title': 'Relationship counts', 'columns': ['Group A', 'Group B', 'Count', 'Expected', 'log2 O/E'],
                  'rows': rows[:5000], 'sort_by': 'Count'},
        'summary': note + f'\n\nScope: {int(eligible.sum()):,} labelled cells. No significance test is implied.'}
