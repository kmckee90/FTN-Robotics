"""Deterministic, display-only consolidation of a shared task-mask layout."""

import numpy as np


def _geometry(shape, radius, roundness):
    """Undirected grid edges: four-neighbor adjacency plus a circular kernel."""
    height, width = shape
    grid = np.arange(height * width).reshape(shape)
    offsets = []
    for dy in range(min(radius, height - 1) + 1):
        for dx in range(-min(radius, width - 1), min(radius, width - 1) + 1):
            distance2 = dy * dy + dx * dx
            if (dy == 0 and dx <= 0) or not 0 < distance2 <= radius * radius:
                continue
            offsets.append((dy, dx, np.exp(-2 * distance2 / radius**2)))
    # Match the circular term's interior weight to the four-neighbor term.
    total = sum(weight for _, _, weight in offsets)
    left, right, weights = [], [], []
    for dy, dx, weight in offsets:
        x0, x1 = max(0, -dx), min(width, width - dx)
        a = grid[:height - dy, x0:x1].ravel()
        b = grid[dy:, x0 + dx:x1 + dx].ravel()
        w = float(dy * dy + dx * dx == 1) + roundness * 2 * weight / total
        if w:
            left.append(a)
            right.append(b)
            weights.append(np.full(a.size, w))
    if not left:
        return np.array([], dtype=int), np.array([], dtype=int), np.array([])
    return np.concatenate(left), np.concatenate(right), np.concatenate(weights)


def spatial_permutation(masks, shape, previous=None, sweeps=8, *,
                        roundness=2.0, center=0.4, radius=3, temperature=0.05):
    """Return display-position -> original-position indices for flattened grids.

    Maximize mean per-task active-neighbor affinity (four-neighbor plus a
    distance-weighted circular kernel), minus the union's squared distance
    from the canvas center. Complete membership signatures move together, so
    all intersections survive. Disconnected and irregular task regions are
    allowed. No wraparound. Set roundness=center=temperature=0 for adjacency
    alone. ``sweeps`` sets the budget (eight proposals per pixel per sweep).

    Seeds include identity, previous, serpentine grouping, and central sectors.
    The first half of the budget anneals; the rest greedily refines the best
    layout found. Return the best observed layout, never worse than the seeds
    under this combined objective. A private RNG leaves training untouched.
    """
    if (not isinstance(radius, (int, np.integer)) or radius < 1
            or not isinstance(sweeps, (int, np.integer)) or sweeps < 0):
        raise ValueError("radius must be a positive integer; sweeps a nonnegative integer")
    if any(not np.isfinite(v) or v < 0 for v in (roundness, center, temperature)):
        raise ValueError("roundness, center, and temperature must be finite and nonnegative")
    bits = np.asarray(masks) > 0
    height, width = shape
    n = height * width
    if height < 1 or width < 1 or bits.ndim != 2 or bits.shape[1] != n:
        raise ValueError("masks must have shape (tasks, height * width) on a nonempty grid")
    identity = np.arange(n)
    if previous is not None:
        previous = np.asarray(previous)
        if previous.shape != (n,) or not np.array_equal(np.sort(previous), identity):
            raise ValueError("previous must be a permutation of grid positions")
    if n < 2 or not bits.shape[0]:
        return identity
    affinity = np.zeros((n, n), dtype=np.float64)
    for start in range(0, len(bits), 4096):
        batch = bits[start:start + 4096].astype(np.float64)
        affinity += batch.T @ batch
    affinity /= len(bits)
    occupied = bits.any(axis=0).astype(float)
    grid = identity.reshape(shape)
    left, right, weights = _geometry(shape, radius, roundness)
    yy, xx = np.indices(shape, dtype=float)
    yy -= (height - 1) / 2
    xx -= (width - 1) / 2
    distance2 = (xx**2 + yy**2).ravel()
    central_cost = center * distance2 / max(distance2.max(), 1)

    def score(order):
        return weights @ affinity[order[left], order[right]] - central_cost @ occupied[order]

    # Packed membership signatures preserve the original lexicographic grouping.
    signatures = np.packbits(bits.T, axis=1)
    grouped = np.lexsort(signatures[:, ::-1].T)
    snake = grid.copy()
    snake[1::2] = snake[1::2, ::-1]
    candidate = identity.copy()
    candidate[snake.ravel()] = grouped
    candidates = [identity, candidate]
    # Pack the union into the closest pixels and place signature groups in
    # angular sectors. This supplies a compact seed without imposing topology.
    active = grouped[occupied[grouped] > 0]
    central = np.argsort(distance2, kind="stable")
    sites = central[:len(active)]
    sites = sites[np.argsort(np.arctan2(yy.ravel()[sites], xx.ravel()[sites]), kind="stable")]
    packed = identity.copy()
    packed[sites] = active
    packed[central[len(active):]] = grouped[occupied[grouped] == 0]
    candidates.append(packed)
    if previous is not None:
        candidates.append(previous)
    order = max(candidates, key=score).copy()
    best = order.copy()
    best_score = current_score = score(order)

    neighbors = [[] for _ in range(n)]
    neighbor_weights = [[] for _ in range(n)]
    for a, b, w in zip(left, right, weights):
        neighbors[a].append(b)
        neighbors[b].append(a)
        neighbor_weights[a].append(w)
        neighbor_weights[b].append(w)
    neighbors = [np.array(row, dtype=int) for row in neighbors]
    neighbor_weights = [np.array(row) for row in neighbor_weights]

    rng = np.random.default_rng(0)
    anneal_sweeps = sweeps // 2 if temperature else 0
    for sweep in range(sweeps):
        if sweep == anneal_sweeps:
            order = best.copy()
            current_score = best_score
        heat = temperature * (0.02 ** (sweep / max(anneal_sweeps - 1, 1))) if sweep < anneal_sweeps else 0
        for a in rng.permutation(n):
            local = rng.choice(neighbors[a], size=4) if len(neighbors[a]) else []
            for b in [*local, *rng.integers(n, size=4)]:
                if a == b:
                    continue
                old_a, old_b = order[a], order[b]
                na, nb = neighbors[a], neighbors[b]
                wa, wb = neighbor_weights[a], neighbor_weights[b]
                # Edges between a and b contribute the same before and after.
                ia, ib = na != b, nb != a
                delta = wa[ia] @ (affinity[old_b, order[na[ia]]] - affinity[old_a, order[na[ia]]])
                delta += wb[ib] @ (affinity[old_a, order[nb[ib]]] - affinity[old_b, order[nb[ib]]])
                delta -= (occupied[order[b]] - occupied[order[a]]) * (central_cost[a] - central_cost[b])
                if delta > 1e-12 or (heat and delta < -1e-12 and rng.random() < np.exp(delta / heat)):
                    order[a], order[b] = order[b], order[a]
                    current_score += delta
                    if current_score > best_score + 1e-10:
                        best_score = current_score
                        best = order.copy()
    return best
