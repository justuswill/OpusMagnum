"""
bounds.py — analytic lower bounds on the metrics of an optimal solution.

These are provable floors (cost, cycles, area, instructions) derived from the
puzzle's inputs/outputs/available parts alone — no search, no simulation.
A metric is *proven optimal* when one of these bounds equals the best known record.
"""

import itertools
import math
from collections import Counter, deque
from dataclasses import replace
from fractions import Fraction
from typing import Optional
from weakref import WeakKeyDictionary

from puzzle_parser import (
    PuzzleFile, ATOM_NAMES,
    ATOM_SALT, ATOM_FIRE, ATOM_VITAE, ATOM_MORS, ATOM_REPEAT, ATOM_QUINTESSENCE, ATOM_QUICKSILVER,
    ELEMENTALS, BOND_NORMAL, BOND_ANY_TRIPLEX,
    PART_BARON, PART_PROJECTION, PART_PURIFICATION,
    PART_REJECTION, PART_DIVISION, PART_PROLIFERATION, PART_RAVARI,
    PART_CALCIFICATION, PART_DUPLICATION, PART_ANIMISMUS, PART_DISPERSION, PART_DISPOSAL,
    PART_BONDER, PART_UNBONDER, PART_BONDER_PRISMA,
    alternate_repeat_puzzle,
)
from stoichiometry import (
    RecipeResult, METAL_CHAIN, DIVISION_OUTPUTS, necessary_inputs, describe_molecule,
    solve_recipe, solve_recipe_combined, solve_recipe_cheap, solve_recipe_min_waste,
)
from schematic import StateGraph, molecule_signature, _is_drop_and_create, _HEX_DIRS, _rotate60
from schematic_parallel import reachable_states

_cadence_latency_memo: "WeakKeyDictionary[StateGraph, tuple]" = WeakKeyDictionary()


# ── Cycle bounds via optimal path through StateGraph ──────────────────


def _all_paths_nodes(graph: StateGraph, start: int, goal: int) -> list:
    """Every path from `start` to `goal` over graph.edges, each as the
    sequence of node indices along it (including both endpoints). The state
    graph is a finite DAG (schematic.py's module docstring), so no node can
    ever recur on a path — plain DFS enumeration needs no visited-set/cycle
    bookkeeping to stay correct, just to terminate on a finite graph."""
    if start == goal:
        return [[goal]]
    paths = []

    def dfs(node, path):
        for nxt in graph.edges[node]:
            if nxt == start:
                paths.append([nxt] + path)
            else:
                dfs(nxt, [nxt] + path)

    dfs(goal, [goal])
    return paths


def _reagent_sig(pf: PuzzleFile, recipe: RecipeResult) -> dict:
    return {molecule_signature(pf.inputs[i]): i
            for i, count in recipe.reagent_counts.items() if count > 0}


def _path_raw_ls(pf: PuzzleFile, recipe: RecipeResult, graph: StateGraph, path: list) -> dict:
    """Per-reagent-type raw_L values for one path, given in
    forward-chronological order (path[0] = the fully-decomposed raw-reagent
    state, path[-1] = the output state — see _all_paths_nodes). Walk it and,
    for each reagent type, note the forward step k at which each instance
    disappears from the state (i.e. gets grabbed and folded into a
    bond/reaction at that step). raw_L is not k itself but D-k+1 — the
    number of steps from that grab through to the output, *inclusive of the
    grab's own step* — since what actually gates the last copy's completion
    is how much sequential work is still ahead of a given grab, not how much
    came before it: a reagent grabbed early but feeding a long downstream
    chain is more of a bottleneck than one grabbed late that's used
    immediately. Split out from _path_l_spine so _cadence_latency can
    compute it once per distinct path and reuse it across every period-many
    path assignment it tries (see _cadence_latency)."""
    D = len(path) - 1
    reagent_sig = _reagent_sig(pf, recipe)

    def type_counts(node):
        counts = Counter()
        for mol in graph.states[node].molecules:
            sig = molecule_signature(mol)
            if sig in reagent_sig:
                counts[sig] += 1
        return counts

    raw_ls = {}
    prev_counts = type_counts(path[0])
    for k in range(1, D + 1):
        cur_counts = type_counts(path[k])
        for sig, prev_count in prev_counts.items():
            consumed = prev_count - cur_counts.get(sig, 0)
            if consumed > 0:
                raw_ls.setdefault(sig, []).extend([D - k + 1] * consumed)
        prev_counts = cur_counts
    return raw_ls


def _path_l_spine(pf: PuzzleFile, recipe: RecipeResult, graph: StateGraph,
                   paths: list, raw_ls_by_path: dict) -> tuple[int, str]:
    """L_spine for one path assignment (one path per product in the group;
    period==1 outside the uneven-split case, see _cadence_latency — each
    product may take a structurally distinct path). Appends each assigned
    path's raw_L pool (raw_ls_by_path, keyed by id(path); see _path_raw_ls)
    per reagent type before the parallel-glyph adjustment.

    Parallel `repeats` glyphs jointly supply 1 new molecule every 2 cycles,
    so same-type raw_L values don't stack directly: sort a type's values
    descending and subtract 2*(num_blocks - idx//repeats - 1) before taking
    the max (rearrangement inequality: earliest-arriving grab pairs with
    least-urgent need). num_blocks is the recipe's own count/repeats term
    (see cycles_lower_bound) scaled by len(paths). L_spine is the max over
    every reagent type's contribution; the displayed "steps that matter"
    are always the winning path's last L_spine moves (delays compound
    toward the end).

    Only each product's own bottleneck (its max adjusted value) competes
    for the shared output-drop slot; only one product can drop per cycle,
    so bottlenecks are sorted ascending and greedily bumped into
    max(bottleneck, previous_completion + 1) — the standard exchange-
    argument-optimal schedule for single-resource unit-time scheduling
    with release times.

    Which instance lands in which scheduling block is its own choice: pure
    value-descending order can pack several products' top instances into
    one block, manufacturing collisions a p_idx-grouped order would avoid.
    Try plain value order plus, for k=1..len(paths)-1, keeping p_idx
    0..k-1 each in their own sorted group with the rest pooled after —
    len(paths) candidates total — and keep whichever gives the smallest
    L_spine."""
    reagent_sig = _reagent_sig(pf, recipe)

    combined = {}
    for p_idx, path in enumerate(paths):
        for sig, values in raw_ls_by_path[id(path)].items():
            combined.setdefault(sig, []).extend((v, p_idx) for v in values)

    if not combined:
        return 0, "no reagents consumed along path"

    # Waves needed per reagent type = instances actually pooled for it
    # (combined[sig], summed over all len(paths) assigned copies) divided
    # by how many duplicate glyphs handle it at once — counted directly
    # from the real pool rather than re-derived from reagent_counts[i]/
    # products_needed, which silently breaks whenever that ratio doesn't
    # reduce to a whole per-path count (P031b: 6 over 18 products, i.e. 1
    # raw cluster shared by 3 products per unit). Identical to the old
    # formula whenever it was already correct, since combined[sig] is then
    # exactly (instances one path contributes) * len(paths).
    num_blocks = max(
        math.ceil(len(combined.get(sig, [])) / recipe.reagent_group_size.get(i, 1))
        for sig, i in reagent_sig.items()
    )

    def value_order(pairs):
        return sorted(pairs, key=lambda vp: -vp[0])

    def prefix_order(pairs, k):
        groups = [sorted((vp for vp in pairs if vp[1] == p), key=lambda vp: -vp[0]) for p in range(k)]
        groups.append(sorted((vp for vp in pairs if vp[1] >= k), key=lambda vp: -vp[0]))
        return [vp for group in groups for vp in group]

    def solve(order_fn):
        bottleneck_by_p = {}
        for sig, pairs in combined.items():
            i = reagent_sig[sig]
            repeats = recipe.reagent_group_size.get(i, 1)
            for idx, (v, p_idx) in enumerate(order_fn(pairs)):
                adjusted = v - 2 * (num_blocks - idx // repeats - 1)
                cur = bottleneck_by_p.get(p_idx)
                if cur is None or adjusted > cur[0]:
                    bottleneck_by_p[p_idx] = (adjusted, i)

        bottlenecks = sorted(((adjusted, i, p_idx) for p_idx, (adjusted, i) in bottleneck_by_p.items()),
                              key=lambda b: b[0])
        natural_peak = bottlenecks[-1][0]
        running = None
        L_spine, i, p_idx = bottlenecks[0]
        for adjusted, bi, bp_idx in bottlenecks:
            running = adjusted if running is None else max(adjusted, running + 1)
            if running > L_spine:
                L_spine, i, p_idx = running, bi, bp_idx
        return L_spine, natural_peak, i, p_idx

    order_fns = [value_order] + [(lambda pairs, k=k: prefix_order(pairs, k)) for k in range(1, len(paths))]
    L_spine, natural_peak, i, p_idx = min((solve(fn) for fn in order_fns), key=lambda r: r[0])

    path = paths[p_idx]
    D = len(path) - 1
    moves = [graph.edge_move[(path[k], path[k - 1])] for k in range(1, D + 1)]
    blocking_moves = moves[max(D - natural_peak, 0):]
    steps = " + ".join(f"{mv}=1" for mv in blocking_moves) if blocking_moves else "already available"
    penalty = L_spine - natural_peak
    if penalty:
        steps += f" + extra_output_delay={penalty}"
    note = f"{describe_molecule(pf.inputs[i])}: {steps}"
    return L_spine, note


def _prune_dominated_paths(paths: list, raw_ls_by_path: dict, reagent_sig: dict) -> list:
    """Drop any path whose raw_L values are componentwise >= some other
    path's, sig-by-sig, rank-by-rank (both sorted descending first, since
    _path_l_spine only cares about each type's sorted raw_L pool, not which
    grab produced which value). Such a path can never beat the dominating
    one in any combination _cadence_latency tries — substituting the
    dominating path in its place only pushes every merged rank down or
    leaves it unchanged, so the achieved minimum L_spine can only get
    smaller or stay the same — so it's safe to drop before the
    combinations_with_replacement blow-up."""
    sigs = list(reagent_sig)
    sorted_values = {
        id(path): {sig: sorted(raw_ls_by_path[id(path)].get(sig, []), reverse=True) for sig in sigs}
        for path in paths
    }

    def le_all(a, b):
        va, vb = sorted_values[id(a)], sorted_values[id(b)]
        return all(x <= y for sig in sigs for x, y in zip(va[sig], vb[sig]))

    kept = []
    for i, path in enumerate(paths):
        dominated = False
        for j, other in enumerate(paths):
            if i == j:
                continue
            if le_all(other, path) and (not le_all(path, other) or j < i):
                dominated = True
                break
        if not dominated:
            kept.append(path)
    return kept


def _pareto_paths_raw_ls(pf: PuzzleFile, recipe: RecipeResult, graph: StateGraph) -> list:
    """Every Pareto-optimal path from any of graph.input_state_indices to
    the output (node 0), paired with its raw_L values — a single BFS/DP
    that prunes dominated partial paths as it walks, instead of
    enumerating every path first (potentially exponential, see
    _all_paths_nodes) and pruning afterward (_prune_dominated_paths) after
    a second walk for raw_L (_path_raw_ls). That older trio is kept unused
    alongside this one as a fallback/comparison point.

    Walks in graph.edges' own direction, starting at the output (0) — same
    direction _all_paths_nodes' DFS uses, no reverse adjacency needed. This
    is chronologically backward (undoing moves), which makes raw_L trivial
    on-line: raw_L for a consumption event is D-k+1 ("1 + forward steps
    remaining to the output"), and forward-steps-remaining is exactly the
    walk's backward distance j from node 0 so far — no need to know D (the
    eventual path length) at all. Each consumption event gets its final
    raw_L the moment it's discovered.

    Domination (a partial path's raw_L-so-far componentwise <= another's,
    per reagent type) is safe to prune on early for the same reason
    _prune_dominated_paths reasons about complete paths: both partial paths
    sit at the same node, share every possible remaining suffix, and
    appending the identical suffix to both preserves the <=.

    graph.input_state_indices can hold more than one node: molecule_signature
    (unlike the graph's own dedup key _state_signature) ignores Atom.age, so
    the same raw-reagent composition can appear as several distinct nodes
    differing only in how much a free atom "waited" (schematic._READY_AGE)
    along the path that found it. A genuine raw-reagent grab has no
    readiness requirement, so every one is a legitimate start — collecting
    results from all of them and taking the minimum in _cadence_latency
    avoids charging for a wait a different, equally valid start didn't
    need."""
    reagent_sig = _reagent_sig(pf, recipe)
    starts, goal = graph.input_state_indices, 0

    counts_cache = {}

    def type_counts(node):
        c = counts_cache.get(node)
        if c is None:
            c = Counter()
            for mol in graph.states[node].molecules:
                sig = molecule_signature(mol)
                if sig in reagent_sig:
                    c[sig] += 1
            counts_cache[node] = c
        return c

    in_degree = {i: 0 for i in range(len(graph.states))}
    for a, bs in graph.edges.items():
        for b in bs:
            in_degree[b] += 1

    order = []
    remaining = dict(in_degree)
    queue = deque(i for i in range(len(graph.states)) if remaining[i] == 0)
    while queue:
        node = queue.popleft()
        order.append(node)
        for nxt in graph.edges.get(node, []):
            remaining[nxt] -= 1
            if remaining[nxt] == 0:
                queue.append(nxt)

    def dominates(a, j_a, b, j_b):
        """True if `a` (at backward-distance j_a) is at least as good as `b`
        (j_b) for any shared future suffix. Requires j_a <= j_b (a's future
        consumption is anchored no worse than b's) and, per type, equal
        counts so far with each elementwise <= — zip()'s silent truncation
        on unequal-length lists previously let extra, uncompared entries in
        b hide a real advantage, wrongly declaring a dominant."""
        if j_a > j_b:
            return False
        for sig in set(a) | set(b):
            xs, ys = sorted(a.get(sig, [])), sorted(b.get(sig, []))
            if len(xs) != len(ys) or not all(x <= y for x, y in zip(xs, ys)):
                return False
        return True

    profiles = {goal: [({}, [goal], 0)]}
    for node in order:
        entries = profiles.get(node)
        if not entries:
            continue
        cur_counts = type_counts(node)
        for nxt in graph.edges.get(node, []):
            consumed = type_counts(nxt) - cur_counts
            bucket = profiles.setdefault(nxt, [])
            for profile, path, j in entries:
                raw_l = j + 1
                new_profile = {sig: list(ages) for sig, ages in profile.items()}
                for sig, cnt in consumed.items():
                    if cnt > 0:
                        new_profile.setdefault(sig, []).extend([raw_l] * cnt)
                new_path = path + [nxt]
                new_j = j + 1

                dominated = False
                survivors = []
                for other_profile, other_path, other_j in bucket:
                    if dominated or dominates(other_profile, other_j, new_profile, new_j):
                        dominated = True
                        survivors.append((other_profile, other_path, other_j))
                    elif not dominates(new_profile, new_j, other_profile, other_j):
                        survivors.append((other_profile, other_path, other_j))
                    # else new_profile dominates other_profile: drop other
                if not dominated:
                    survivors.append((new_profile, new_path, new_j))
                bucket[:] = survivors

    results = []
    for start in starts:
        for profile, path, _j in profiles.get(start, []):
            results.append((list(reversed(path)), profile))
    return results


def _output_sig(pf: PuzzleFile) -> dict:
    return {molecule_signature(mol): i for i, mol in enumerate(pf.outputs)}


def _pareto_paths_output_ls(pf: PuzzleFile, recipe: RecipeResult, graph: StateGraph) -> list:
    """Mirror of _pareto_paths_raw_ls, walking forward-chronologically
    (input -> output, via graph.edges' reverse adjacency) instead of
    output -> input: tracks the step at which each output-type instance
    first *appears* (becomes free-standing) rather than when a raw
    reagent gets consumed. Feeds _input_latency, the output_constrained
    mirror of _cadence_latency."""
    output_sig = _output_sig(pf)
    starts, goal = graph.input_state_indices, 0

    counts_cache = {}

    def type_counts(node):
        c = counts_cache.get(node)
        if c is None:
            c = Counter()
            for mol in graph.states[node].molecules:
                sig = molecule_signature(mol)
                if sig in output_sig:
                    c[sig] += 1
            counts_cache[node] = c
        return c

    in_degree = {i: 0 for i in range(len(graph.states))}
    for a, bs in graph.edges.items():
        for b in bs:
            in_degree[b] += 1

    order = []
    remaining = dict(in_degree)
    queue = deque(i for i in range(len(graph.states)) if remaining[i] == 0)
    while queue:
        node = queue.popleft()
        order.append(node)
        for nxt in graph.edges.get(node, []):
            remaining[nxt] -= 1
            if remaining[nxt] == 0:
                queue.append(nxt)

    rev_edges = {i: [] for i in range(len(graph.states))}
    for a, bs in graph.edges.items():
        for b in bs:
            rev_edges[b].append(a)

    def dominates(a, j_a, b, j_b):
        if j_a > j_b:
            return False
        for sig in set(a) | set(b):
            xs, ys = sorted(a.get(sig, [])), sorted(b.get(sig, []))
            if len(xs) != len(ys) or not all(x <= y for x, y in zip(xs, ys)):
                return False
        return True

    profiles = {s: [({}, [s], 0)] for s in starts}
    for node in reversed(order):
        entries = profiles.get(node)
        if not entries:
            continue
        cur_counts = type_counts(node)
        for nxt in rev_edges.get(node, []):
            appeared = type_counts(nxt) - cur_counts
            bucket = profiles.setdefault(nxt, [])
            for profile, path, j in entries:
                appear_val = j + 1
                new_profile = {sig: list(vals) for sig, vals in profile.items()}
                for sig, cnt in appeared.items():
                    if cnt > 0:
                        new_profile.setdefault(sig, []).extend([appear_val] * cnt)
                new_path = path + [nxt]
                new_j = j + 1

                dominated = False
                survivors = []
                for other_profile, other_path, other_j in bucket:
                    if dominated or dominates(other_profile, other_j, new_profile, new_j):
                        dominated = True
                        survivors.append((other_profile, other_path, other_j))
                    elif not dominates(new_profile, new_j, other_profile, other_j):
                        survivors.append((other_profile, other_path, other_j))
                if not dominated:
                    survivors.append((new_profile, new_path, new_j))
                bucket[:] = survivors

    results = []
    for profile, path, _j in profiles.get(goal, []):
        results.append((path, profile))
    return results


def _input_latency(pf: PuzzleFile, recipe: RecipeResult, graph: StateGraph) -> tuple[int, str]:
    """output_constrained mirror of _cadence_latency: instead of asking
    when raw reagents get consumed (limited by input availability), asks
    when each output-type instance first appears (limited by how fast the
    build-up produces outputs), walking forward from the raw reagents.

    Per Pareto path (_pareto_paths_output_ls), each output type's pooled
    appear-steps are sorted ascending and adjusted by subtracting their
    rank (0, 1, 2, ...) — the same single-resource release-time
    reasoning _path_l_spine uses, just simplified (no repeats/wave
    grouping): the i-th (0-indexed) instance to appear can be fully
    absorbed by whatever consumes it i cycles later than the first
    without adding delay, so only the max adjusted value is a genuine
    bottleneck. L_spine for a path is the max of that over every output
    type; the puzzle's L_spine is the smallest across all Pareto paths,
    ties broken by lexicographically-smallest path."""
    output_sig = _output_sig(pf)
    pareto = _pareto_paths_output_ls(pf, recipe, graph)

    def path_l_spine(profile):
        """(L_spine, blamed sig) for one path — blamed is whichever output
        type's own adjusted max drove the path's L_spine, mirroring
        _path_l_spine's `i` tracking."""
        if not profile:
            return 0, None
        best = None
        for sig, values in profile.items():
            adjusted_max = max(v - idx for idx, v in enumerate(sorted(values)))
            if best is None or adjusted_max > best[0]:
                best = (adjusted_max, sig)
        return best

    scored = sorted(
        ((*path_l_spine(profile), path) for path, profile in pareto),
        key=lambda r: (r[0], r[2]),
    )
    L_spine, blamed_sig, path = scored[0]

    D = len(path) - 1
    moves = [graph.edge_move[(path[k], path[k - 1])] for k in range(1, D + 1)]
    blocking_moves = moves[:L_spine]
    steps = " + ".join(["grab=1"] + [f"{mv}=1" for mv in blocking_moves])
    blamed = describe_molecule(pf.outputs[output_sig[blamed_sig]]) if blamed_sig is not None else "output"
    note = f"{blamed}: {steps}"
    return L_spine + 1, note


def _cadence_latency(pf: PuzzleFile, recipe: RecipeResult, graph: StateGraph) -> tuple[int, str]:
    """L_spine across every structurally-distinct path from raw reagents to
    the output (schematic.py's _all_paths_nodes, forward-chronological
    order), not just the shortest: the real optimal solution follows
    exactly one of these paths, unknown to us, so only the smallest
    L_spine any of them produces is a provably safe lower bound — a larger
    one risks overshooting whichever path the real solution follows. See
    _path_l_spine for the per-path algorithm.

    When a reagent's per-product demand doesn't split evenly across its
    parallel glyphs, no single product's raw_L pool matches
    recipe.reagent_counts alone — the remainder only evens out after
    `period` products (e.g. 4 needed per product over 3 duplicates:
    2/1/1, 1/2/1, 1/1/2, realigning every 3). So instead of one path per
    product, try every length-`period` multiset of paths
    (combinations_with_replacement, not product — _path_l_spine only pools
    the assigned paths' raw_L values, so "aab" and "aba" score identically),
    keeping the smallest L_spine across all combinations.

    Memoized on `graph` (_cadence_latency_memo, module-level): a graph is
    built for one specific recipe, and callers — notably solve_analysis.py's
    SUM-tightening loop via cycles_lower_bound_for_budget — call this
    repeatedly with the same (pf, recipe, graph), so recomputing the
    combinations_with_replacement search every time would be pure
    waste."""
    cached = _cadence_latency_memo.get(graph)
    if cached is not None and cached[0] == id(recipe):
        return cached[1]

    products_needed = pf.products_needed()
    period = 1
    for i, count in recipe.reagent_counts.items():
        if count <= 0:
            continue
        repeats = recipe.reagent_group_size.get(i, 1)
        unit_count = Fraction(count, products_needed)
        period = math.lcm(period, unit_count.denominator * (repeats // math.gcd(unit_count.numerator, repeats)))

    # paths = _all_paths_nodes(graph, graph.input_state_idx, 0)
    # raw_ls_by_path = {id(path): _path_raw_ls(pf, recipe, graph, path) for path in paths}
    # paths = _prune_dominated_paths(paths, raw_ls_by_path, _reagent_sig(pf, recipe))
    pareto = _pareto_paths_raw_ls(pf, recipe, graph)
    paths = [path for path, _raw_ls in pareto]
    raw_ls_by_path = {id(path): raw_ls for path, raw_ls in pareto}

    best = None
    for assignment in itertools.combinations_with_replacement(paths, period):
        L_spine, note = _path_l_spine(pf, recipe, graph, list(assignment), raw_ls_by_path)
        if best is None or L_spine < best[0]:
            best = (L_spine, note)
    _cadence_latency_memo[graph] = (id(recipe), best)
    return best


def cycles_lower_bound_single(pf: PuzzleFile, recipe: RecipeResult, states: StateGraph) -> tuple[int, str]:
    """
    Throughput (N) + Steps (L) (+ 1) lower bound on cycle count.
    Each input may have their own N, L so one dominates.

    Throughput:
    ───────────
    - Each output accepts at most 1 molecule per 1 cycle
    - Each input glyph spawns at most 1 molecule per 2 cycles (1 grab + 1 move).

    Latency:
    ────────
    Steps from source to output, for the very last piece. This requires
    - grab input
    - L steps from minimal list of actions (bonds, glyphs, ...)
    - move to output
    - for non-repeating puzzles we have to add +1 for the last drop.
    Throughout already counts the first grab and move=action, thus - 2

    ASSUMPTIONS (false)
    - Higher N does not decrease L if using a different sequence of steps.
    - Current recipe has all transformations needed to reach minimum L
    - minimum period is able to achieve minimum L

    refs
    https://biggieblog.com/battling-the-entire-world-in-opus-magnum/
    """
    products_needed = pf.products_needed()

    N_out = products_needed
    note_out = f"output throughput: {products_needed} products, 1/cycle"

    N_in = 0
    note_in = ""

    combined_reagent_indices = set()
    for r in recipe.extra_reactions.values():
        member_indices = [
            i for i, mol in enumerate(pf.inputs)
            if i in recipe.reagent_counts
            and len(mol.atom_type_counts()) == 1
            and next(iter(mol.atom_type_counts())) in (r.alt_reagent or [])
        ]
        if not member_indices:
            continue
        combined_reagent_indices.update(member_indices)
        total_count = sum(recipe.reagent_counts.get(i, 0) for i in member_indices)
        total_repeats = sum(recipe.reagent_group_size.get(i, 1) for i in member_indices)
        if total_count <= 0:
            continue
        cycles = 2 * math.ceil(total_count / total_repeats)
        if cycles > N_in:
            N_in = cycles
            breakdown = " + ".join(
                f"{recipe.reagent_group_size.get(i, 1)}x{ATOM_NAMES[next(iter(pf.inputs[i].atom_type_counts()))]}"
                for i in member_indices
            )
            note_in = (f"input throughput {N_in}: {total_count}x needed from combined "
                       f"{total_repeats} duplicates ({breakdown})")

    for i, count in recipe.reagent_counts.items():
        if i in combined_reagent_indices:
            continue  # already accounted for, pooled with the rest of its combined group above
        if count <= 0:
            continue
        repeats = recipe.reagent_group_size.get(i, 1)
        cycles = 2 * math.ceil(count / repeats)
        if cycles > N_in:
            N_in = cycles
            note_in = (f"input throughput {N_in}: {count}x needed from input {i}"
                       + (f" with {repeats} duplicates" if repeats > 1 else ""))

    all_repeating = all(any(a.type == ATOM_REPEAT for a in mol.atoms) for mol in pf.outputs)
    drop_term = 0 if all_repeating else 1
    L_out, spine_note_out = _input_latency(pf, recipe, states)  # already includes its own grab=1/+1
    L_in, spine_note_in = _cadence_latency(pf, recipe, states)
    total_out = N_out + L_out + drop_term
    total_in = N_in + L_in + drop_term
    output_constrained = total_out >= total_in

    if output_constrained:
        total, note, L, latency_parts = total_out, note_out, L_out, [spine_note_out]
    else:
        total, note, L, latency_parts = total_in, note_in, L_in, [spine_note_in]
    if not all_repeating:
        latency_parts.append("drop=1")  # last drop isn't pipelined away unless every output repeats
        L += 1
    label = "output latency" if output_constrained else "input latency"
    latency_note = f"{label} {L}: " + " + ".join(latency_parts)

    return total, f"{note}, {latency_note}"


def cycles_lower_bound(pf: PuzzleFile, recipe: RecipeResult, states: StateGraph,
                        workers: Optional[int] = None, batch_size: int = 8) -> tuple[int, str]:
    """cycles_lower_bound_single(pf), plus — for a repeating output — the same
    computation with the repeat marker(s) mirrored onto the opposite end
    (see puzzle_parser.alternate_repeat_puzzle): a real solution isn't
    required to build the polymer in the direction the puzzle file happens
    to depict, so try both and keep whichever gives the smaller (still
    sound) bound. `workers`/`batch_size` only affect the mirrored-repeat
    side's own reachable_states call (schematic_parallel) — `states` for
    the primary orientation is passed in already computed."""
    c_lo, c_note = cycles_lower_bound_single(pf, recipe, states)

    alt_pf = alternate_repeat_puzzle(pf)
    if alt_pf is not None:
        try:
            try:
                alt_recipe = solve_recipe_combined(alt_pf)
            except NotImplementedError:
                alt_recipe = solve_recipe(alt_pf)
            alt_states = reachable_states(alt_pf, alt_recipe, workers=workers, batch_size=batch_size)
            alt_c_lo, alt_c_note = cycles_lower_bound_single(alt_pf, alt_recipe, alt_states)
        except AssertionError:
            alt_c_lo = None
        if alt_c_lo is not None and alt_c_lo < c_lo:
            c_lo, c_note = alt_c_lo, alt_c_note + " (mirrored repeat)"

    return c_lo, c_note


# ── Cycle bounds at fixed costs ───────────────────────────────────────


def _min_arm_cost(k: int, N: int) -> Optional[int]:
    """Minimum gold in arms to deliver >= k reagent grabs within N cycles,
    combining three purchasable per-stream rate tiers:
      - 20g: 1 grab per 4 cycles
      - 30g: 1 grab per 3 cycles
      - 40g: 1 grab per 2 cycles — two 20g arms cooperating/alternating on
        *one* stream, the fastest possible for a single stream. Not the
        same as two *independent* 20g arms (whose summed rate can
        undershoot N//2 under integer rounding — e.g. N=6: 2*(6//4)=2 but
        6//2=3), so this is a genuinely distinct purchasable tier.
    Multiple independent streams (different reagent positions) can each be
    bought at whichever tier is cheapest for their share of the aggregate
    demand k, rates simply adding. None if N < 2, where no tier delivers
    anything at all.

    Empirically validated against the Opus Magnum leaderboard archive
    (12345ieee/om-leaderboard-archive, filtered to overlap-glitch
    submissions): this cost never exceeded any of the 47 campaign puzzles'
    true cheapest-at-fastest-legitimate-cycles record."""
    if k <= 0:
        return 0
    rate20, rate30, rate40 = N // 4, N // 3, N // 2

    best = None
    max_n40 = (k // rate40 + 1) if rate40 > 0 else 0
    for n40 in range(max_n40 + 1):
        after40 = k - n40 * rate40
        max_n30 = (after40 // rate30 + 1) if rate30 > 0 and after40 > 0 else 0
        for n30 in range(max_n30 + 1):
            remaining = after40 - n30 * rate30
            if remaining <= 0:
                n20 = 0
            elif rate20 > 0:
                n20 = -(-remaining // rate20)  # ceil division
            else:
                continue  # 20g arms deliver nothing at this N; can't close the gap
            cost = n40 * 40 + n30 * 30 + n20 * 20
            if best is None or cost < best:
                best = cost
    return best


def cycles_lower_bound_for_budget(pf: PuzzleFile, recipe: RecipeResult, states: StateGraph,
                                   arm_budget: int) -> int:
    """Minimum cycles for any solution whose arm spend is within
    `arm_budget` gold, delivering as many reagent grabs as `recipe` needs
    (see _min_arm_cost). Not a sound unconditional floor like
    cycles_lower_bound — only valid conditioned on a cost budget already
    derived soundly. Used solely to tighten solve_analysis.py's SUM interval
    (arm_budget there comes from a SUM-record cost ceiling minus
    cost_lower_bound's non-arm floor), never as a substitute for the
    standalone cycles (c) bound.

    cycles_lower_bound is N (throughput) + L (latency/spine) — arm_budget
    only constrains N, not L (the longest sequential chain doesn't shorten
    just because arms are scarcer), so this must add the same L
    cycles_lower_bound_single would, not return budget-constrained N alone
    — otherwise it compares N_budget against the unrelated N+L and could
    produce a tightened value unsafe to max against.

    _min_arm_cost(k, N) is non-increasing in N (more cycles never makes a
    given throughput harder to afford), so the smallest N with
    cost <= arm_budget is found by doubling to bracket, then binary
    search."""
    k = sum(count for count in recipe.reagent_counts.values() if count > 0)

    L, spine_note = _cadence_latency(pf, recipe, states)
    all_repeating = all(any(a.type == ATOM_REPEAT for a in mol.atoms) for mol in pf.outputs)
    if not all_repeating:
        L += 1  # last drop isn't pipelined away unless every output repeats

    if k <= 0 or arm_budget < 20:
        return L

    def affordable(n):
        cost = _min_arm_cost(k, n)
        return cost is not None and cost <= arm_budget

    lo = 1
    while not affordable(lo):
        lo *= 2
    hi = lo
    lo = 1 if lo == 1 else lo // 2 + 1
    while lo < hi:
        mid = (lo + hi) // 2
        if affordable(mid):
            hi = mid
        else:
            lo = mid + 1
    return lo + L


# ── Area/Cost bounds via rules ────────────────────────────────────────


def _bond_signatures_by_type(mols) -> dict:
    """atom_type -> list of 6-slot direction lists, one per instance of that type across `mols`
    — slot i is None if unbonded in _HEX_DIRS[i]'s direction, else (bond_type, neighbor_atom_type).
    Comparing two such lists needs a rotation search, for any of the 6 rotations in the hex grid."""
    result: dict = {}
    for mol in mols:
        pos_to_type = {(a.u, a.v): a.type for a in mol.atoms}
        slots = {(a.u, a.v): [None] * len(_HEX_DIRS) for a in mol.atoms}
        for b in mol.bonds:
            fpos, tpos = (b.from_u, b.from_v), (b.to_u, b.to_v)
            ftype, ttype = pos_to_type.get(fpos), pos_to_type.get(tpos)
            if ftype is None or ttype is None:
                continue
            fdir = (tpos[0] - fpos[0], tpos[1] - fpos[1])
            tdir = (fpos[0] - tpos[0], fpos[1] - tpos[1])
            if fdir in _HEX_DIRS:
                slots[fpos][_HEX_DIRS.index(fdir)] = (b.type, ttype)
            if tdir in _HEX_DIRS:
                slots[tpos][_HEX_DIRS.index(tdir)] = (b.type, ftype)
        for pos, atype in pos_to_type.items():
            result.setdefault(atype, []).append(slots[pos])
    return result


def _position_bond_signatures(mol) -> tuple:
    """(pos_to_type, pos_to_slots) for one molecule — the per-position
    sibling of _bond_signatures_by_type, which flattens across every
    molecule in a list by atom type and throws the position away. The
    whole-molecule coverage search (_molecule_coverage_reachable) needs to
    know exactly where a candidate placement's atoms and bonds land, not
    just that some matching signature exists somewhere."""
    pos_to_type = {(a.u, a.v): a.type for a in mol.atoms}
    slots = {(a.u, a.v): [None] * len(_HEX_DIRS) for a in mol.atoms}
    for b in mol.bonds:
        fpos, tpos = (b.from_u, b.from_v), (b.to_u, b.to_v)
        ftype, ttype = pos_to_type.get(fpos), pos_to_type.get(tpos)
        if ftype is None or ttype is None:
            continue
        fdir = (tpos[0] - fpos[0], tpos[1] - fpos[1])
        tdir = (fpos[0] - tpos[0], fpos[1] - tpos[1])
        if fdir in _HEX_DIRS:
            slots[fpos][_HEX_DIRS.index(fdir)] = (b.type, ttype)
        if tdir in _HEX_DIRS:
            slots[tpos][_HEX_DIRS.index(tdir)] = (b.type, ftype)
    return pos_to_type, slots


def _molecule_placements_covering(in_sig: tuple, anchor_out_pos: tuple,
                                   out_pos_to_type: dict, out_slots: dict,
                                   uncovered: frozenset,
                                   needs_calc: bool, needs_dup: bool,
                                   has_rejection: bool, has_projection: bool):
    """Yield one frozenset of output (u, v) positions per valid rigid
    placement of a whole input molecule that covers `anchor_out_pos`.
    A placement is one of the input molecule's own atoms mapped onto
    anchor_out_pos, combined with one of 6 rotations.
    Valid means every input atom lands on a currently-`uncovered` output
    position whose actual atom type is _reachable_via_transform from the
    input atom's own type (same has_space rule _bond_signature_compatible
    already applies), and every bond the input molecule has is reproduced
    with the exact same bond type between the correspondingly mapped
    positions."""
    in_pos_to_type, in_slots = in_sig
    for anchor_in_pos in in_pos_to_type:
        for k in range(6):
            rot_anchor = _rotate60(anchor_in_pos[0], anchor_in_pos[1], 0, 0, k)
            du = anchor_out_pos[0] - rot_anchor[0]
            dv = anchor_out_pos[1] - rot_anchor[1]
            mapping = {}
            ok = True
            for ip, itype in in_pos_to_type.items():
                r = _rotate60(ip[0], ip[1], 0, 0, k)
                op = (r[0] + du, r[1] + dv)
                if op not in uncovered:
                    ok = False
                    break
                has_space = None in in_slots[ip] or None in out_slots[op]
                if not _reachable_via_transform(
                    itype, out_pos_to_type[op], needs_calc, needs_dup,
                    has_rejection, has_projection, has_space,
                ):
                    ok = False
                    break
                mapping[ip] = op
            if not ok:
                continue
            for ip, op in mapping.items():
                for d, slot in enumerate(in_slots[ip]):
                    if slot is None:
                        continue
                    btype = slot[0]
                    neighbor_ip = (ip[0] + _HEX_DIRS[d][0], ip[1] + _HEX_DIRS[d][1])
                    neighbor_op = mapping[neighbor_ip]
                    real_dir = (neighbor_op[0] - op[0], neighbor_op[1] - op[1])
                    out_slot = out_slots[op][_HEX_DIRS.index(real_dir)]
                    if out_slot is None or out_slot[0] != btype:
                        ok = False
                        break
                if not ok:
                    break
            if ok:
                yield frozenset(mapping.values())


def _molecule_coverage_reachable(out_mol, input_sigs: list, synthesizable_single,
                                  needs_calc: bool, needs_dup: bool,
                                  has_rejection: bool, has_projection: bool,
                                  budget: float = float("inf")) -> bool:
    """True if out_mol's full atom set can be exactly partitioned into
    whole, unmodified copies of pf.inputs molecules. Positions whose atom type
    is reachable as a free atom are excluded from the coverage requirement.

    Backtracking search over which whole input piece covers each
    still-uncovered position, always picking the same canonical
    (lexicographically smallest) uncovered position as the next anchor —
    this makes the search a DAG over frozenset(remaining) states (two
    different orders of independent placements can converge on the same
    remaining set), so memoizing on that frozenset is sound. Terminates
    without a cycle guard since a placement always covers >=1 atom, so
    `remaining` strictly shrinks every recursive call.

    `budget` bounds total states explored; on exhaustion, unresolved states
    are treated as reachable (optimistic) rather than failed."""
    out_pos_to_type, out_slots = _position_bond_signatures(out_mol)
    all_positions = frozenset(
        pos for pos, atype in out_pos_to_type.items()
        if not synthesizable_single(atype)
    )
    memo: dict = {}
    state_count = 0

    def search(remaining: frozenset) -> bool:
        nonlocal state_count
        if not remaining:
            return True
        if remaining in memo:
            return memo[remaining]
        state_count += 1
        if state_count > budget:
            return True
        anchor = min(remaining)
        result = False
        for in_mol, in_sig in input_sigs:
            if len(in_sig[0]) > len(remaining):
                continue
            for covered in _molecule_placements_covering(
                in_sig, anchor, out_pos_to_type, out_slots, remaining,
                needs_calc, needs_dup, has_rejection, has_projection,
            ):
                if search(remaining - covered):
                    result = True
                    break
            if result:
                break
        memo[remaining] = result
        return result

    return search(all_positions)


def _needs_debond_coverage(pf: PuzzleFile, synthesizable_single, needs_calc: bool, needs_dup: bool,
                            has_rejection: bool, has_projection: bool) -> bool:
    """True if any output molecule with more than one atom can't be tiled
    out of whole, un-debonded input pieces, so an Unbonder is actually required."""
    input_sigs = [(mol, _position_bond_signatures(mol)) for mol in pf.inputs]
    return any(
        not _molecule_coverage_reachable(mol, input_sigs, synthesizable_single, needs_calc, needs_dup, has_rejection, has_projection)
        for mol in pf.outputs
        if len(mol.atoms) > 1
    )


def _reachable_via_transform(earlier: int, later: int, needs_calc: bool, needs_dup: bool,
                              has_rejection: bool, has_projection: bool, has_space: bool) -> bool:
    """True if a bonded neighbor of type `earlier` could still be sitting
    in that same bonded spot, now showing as `later`, because a
    single-atom transform reaction turned it there in place without ever touching the bond
    — calcification (elemental -> salt), duplication (salt -> elemental),
    rejection (higher metal -> lower), or projection (lower metal -> higher).
    Duplication/rejection/projection additionally require `has_space` —
    at least one open (None) slot in one of the two full signatures being
    compared — since those glyphs need the atom to still have a free hex
    to be walked to/from by an arm; calcification has no such requirement."""
    if earlier == later:
        return True
    if needs_calc and earlier in ELEMENTALS and later == ATOM_SALT:
        return True
    if not has_space:
        return False
    if needs_dup and earlier == ATOM_SALT and later in ELEMENTALS:
        return True
    if needs_calc and needs_dup and earlier in ELEMENTALS and later in ELEMENTALS:
        return True
    if earlier in METAL_CHAIN and later in METAL_CHAIN:
        ei, li = METAL_CHAIN.index(earlier), METAL_CHAIN.index(later)
        if has_rejection and li <= ei:
            return True
        if has_projection and li >= ei:
            return True
    return False


def _transform_source_types(atype: int, needs_calc: bool, needs_dup: bool,
                             has_rejection: bool, has_projection: bool) -> set:
    """Types a bonded input atom could have been before a single-atom
    transform reaction turned it into `atype` in place, without ever
    touching its bonds — the reverse of _reachable_via_transform. Always
    includes `atype` itself. Defers checking for space to later."""
    types = {atype}
    if needs_calc and atype == ATOM_SALT:
        types |= ELEMENTALS
    if needs_dup and atype in ELEMENTALS:
        types.add(ATOM_SALT)
    if needs_calc and needs_dup and atype in ELEMENTALS:
        types |= ELEMENTALS
    if atype in METAL_CHAIN:
        li = METAL_CHAIN.index(atype)
        if has_rejection:
            types.update(METAL_CHAIN[li:])
        if has_projection:
            types.update(METAL_CHAIN[:li + 1])
    return types


def _bond_signature_compatible(reference: list, candidate: list,
                                normal_only: bool = False, triplex_only: bool = False,
                                reference_is_earlier: bool = True,
                                needs_calc: bool = False, needs_dup: bool = False,
                                has_rejection: bool = False, has_projection: bool = False) -> bool:
    """True if some rotation of `reference` has every non-None slot
    matching `candidate` at that same (rotated) slot — i.e. `reference`'s
    bonds are all still present in `candidate`, just possibly turned to a
    different one of the 6 hex orientations. With normal_only (or
    triplex_only), any reference slot whose bond isn't a normal (or
    triplex) bond is treated as a don't-care (None) — used to tell
    whether an incompatibility is specifically about a missing bond of
    that kind, since Bonder and Bonder-Prisma are separate glyphs.

    A slot "matches" if the bond type is identical and either the
    neighbor atom type is identical too, or a single-atom transform
    reaction the puzzle needs could turn one neighbor type into the
    other in place — see _reachable_via_transform. reference_is_earlier
    says which side of the pair is chronologically first (needs_debond
    compares raw-input-first against the final output; needs_bond/
    needs_star_tracks compare a target shape second against what the raw
    input already offers first, i.e. reference is the *later* side)."""
    n = len(reference)
    has_space = None in reference or None in candidate

    def ref_slot(i):
        s = reference[i]
        if s is not None:
            if normal_only and not (s[0] & BOND_NORMAL):
                return None
            if triplex_only and not (s[0] & BOND_ANY_TRIPLEX):
                return None
        return s

    def slots_match(r, c):
        if r is None:
            return True
        if c is None or r[0] != c[0]:
            return False
        earlier, later = (r[1], c[1]) if reference_is_earlier else (c[1], r[1])
        return _reachable_via_transform(earlier, later, needs_calc, needs_dup, has_rejection, has_projection, has_space)

    return any(
        all(slots_match(ref_slot(i), candidate[(i + shift) % n]) for i in range(n))
        for shift in range(n)
    )


def _needs_bond_action(output_bond_sigs: dict, input_bond_sigs: dict, *,
                        pf: PuzzleFile, relevant_part: int,
                        reference_is_earlier: bool,
                        normal_only: bool = False, triplex_only: bool = False,
                        needs_calc: bool, needs_dup: bool,
                        has_rejection: bool, has_projection: bool,
                        synthesizable_single=None,
                        single_atom_input_types: Optional[set] = None,
                        metal_glyph_up: Optional[str] = None, metal_glyph_down: Optional[str] = None,
                        single_salt: bool = False, single_quicksilver: bool = False,
                        needs_proliferation: bool = False, needs_animismus: bool = False,
                        needs_dispersion: bool = False, needs_unification: bool = False) -> tuple:
    """Shared core of needs_debond/needs_bonder/needs_bonder_prisma: for
    every output atom's bond signature, is there a transform-reachable
    bonded input precedent compatible with it via _bond_signature_compatible?
    reference_is_earlier picks which side is the input side.
    `synthesizable_single` is required to verify if single atom outputs
    are reachable, as that case can use non-bonded transforms.

    When `relevant_part` isn't available, a fallback search tries
    sets of (calcification, duplication, rejection, projection,
    animismus, dispersion, unification, proliferation), smallest first,
    elevated to their raw part availability."""
    def relevant(sig):
        # Consider only molecules with relevant bonds
        return any(
            s is not None
            and (not normal_only or (s[0] & BOND_NORMAL))
            and (not triplex_only or (s[0] & BOND_ANY_TRIPLEX))
            for s in sig
        )

    def compatible_with(atype, out_sig, c, d, r, p):
        srcs = _transform_source_types(atype, c, d, r, p)
        return any(
            _bond_signature_compatible(
                *((in_sig, out_sig) if reference_is_earlier else (out_sig, in_sig)),
                normal_only=normal_only, triplex_only=triplex_only,
                reference_is_earlier=reference_is_earlier,
                needs_calc=c, needs_dup=d, has_rejection=r, has_projection=p,
            )
            for src in srcs
            for in_sig in input_bond_sigs.get(src, ())
            if _reachable_via_transform(src, atype, c, d, r, p, None in in_sig)
        )

    extra_needed = set()
    can_fallback = not (pf.parts_available & relevant_part)
    allow_single_synth = reference_is_earlier and bool(pf.parts_available & (PART_BONDER | PART_BONDER_PRISMA))
    fallback_candidates = []
    if can_fallback:
        fallback_candidates = [
            (name, avail) for name, avail, already in (
                # bonded
                ("calcification", bool(pf.parts_available & PART_CALCIFICATION), needs_calc),
                ("duplication", bool(pf.parts_available & PART_DUPLICATION), needs_dup),
                ("rejection", bool(pf.parts_available & PART_REJECTION), has_rejection),
                ("projection", bool(pf.parts_available & PART_PROJECTION), has_projection),
                # others
                ("purification", bool(pf.parts_available & PART_PURIFICATION), has_projection),
                ("division", bool(pf.parts_available & PART_DIVISION), has_rejection),
                ("animismus", bool(pf.parts_available & PART_ANIMISMUS), needs_animismus),
                ("dispersion", bool(pf.parts_available & PART_DISPERSION), needs_dispersion),
                ("unification", bool(pf.parts_available & PART_DISPERSION), needs_unification),
                ("proliferation", bool(pf.parts_available & PART_PROLIFERATION), needs_proliferation),
            )
            if avail and not already
        ]

    def matches_with(atype, out_sig, names):
        c = needs_calc or "calcification" in names
        d = needs_dup or "duplication" in names
        r = has_rejection or "rejection" in names
        p = has_projection or "projection" in names
        if compatible_with(atype, out_sig, c, d, r, p):
            return True
        if allow_single_synth and single_atom_input_types is not None:
            dv = (metal_glyph_down is not None and "division" in metal_glyph_down) or "division" in names
            pu = (metal_glyph_up is not None and "purification" in metal_glyph_up) or "purification" in names
            mg_up = "projection-or-purification" if p and pu else "projection" if p else "purification" if pu else None
            mg_down = "rejection-or-division" if r and dv else "rejection" if r else "division" if dv else None
            elevated_salt = single_salt or (c and bool(single_atom_input_types & ELEMENTALS))
            elevated_quicksilver = single_quicksilver or (
                r and any(m in single_atom_input_types for m in METAL_CHAIN[1:])
            )
            an = needs_animismus or "animismus" in names
            di = needs_dispersion or "dispersion" in names
            un = needs_unification or "unification" in names
            pr = needs_proliferation or "proliferation" in names
            if _synthesizable_single(atype, single_atom_input_types, mg_up, mg_down,
                                      elevated_salt, elevated_quicksilver, pr, d, di, an, un):
                return True
        return False

    def compatible(atype, out_sig, srcs):
        if matches_with(atype, out_sig, set()):
            return True
        if not can_fallback:
            return False
        full_names = {n for n, _ in fallback_candidates}
        if not matches_with(atype, out_sig, full_names):
            return False
        required = {
            n for n in full_names
            if not matches_with(atype, out_sig, full_names - {n})
        }
        extra_needed.update(required)
        return True

    def needs_action(atype, out_sig, srcs):
        # cant be made from bond-less inputs
        if all(s is None for s in out_sig):
            return synthesizable_single is not None and not synthesizable_single(atype)
        # cant be made by removing pre-existing bounds
        if reference_is_earlier:
            if not (relevant(out_sig) and any(src in input_bond_sigs for src in srcs)):
                return False
            return not compatible(atype, out_sig, srcs)
        if not relevant(out_sig):
            return False
        # and cant use pre-existing bounds
        return not compatible(atype, out_sig, srcs)

    result = any(
        needs_action(atype, out_sig, srcs)
        for atype, out_sigs in output_bond_sigs.items()
        for out_sig in out_sigs
        for srcs in [_transform_source_types(atype, needs_calc, needs_dup, has_rejection, has_projection)]
    )
    return result, (extra_needed if not result else set())


def _division_closure(m: int) -> set:
    """All metals reachable from m via zero or more divide_ firings."""
    seen = set()
    stack = [m]
    while stack:
        for nxt in DIVISION_OUTPUTS.get(stack.pop(), ()):
            if nxt not in seen:
                seen.add(nxt)
                stack.append(nxt)
    return seen


_DIVISION_REACHABLE = {m: _division_closure(m) for m in METAL_CHAIN}


def _division_reachable_from(m: int, available: set, have_up_glyph: bool) -> bool:
    """True if metal `m` is reachable via Division from something in
    `available`, allowing the divided-from metal to first be climbed up
    via Projection/Purification (if have_up_glyph) past its own rank —
    mirroring the lead-bootstrap idea already used for Rejection's
    quicksilver byproduct (see _resolve_single_quicksilver). Division's
    own reachability (_DIVISION_REACHABLE) only covers dividing a metal
    already at that rank; the climb is a separate, earlier step (e.g.
    P256-Lodestone: copper purifies up to silver, then silver divides to
    tin+iron — copper's own division alone only reaches tin/lead)."""
    for h in available:
        if h not in METAL_CHAIN:
            continue
        climbable = METAL_CHAIN[METAL_CHAIN.index(h):] if have_up_glyph else (h,)
        if any(m in _DIVISION_REACHABLE.get(c, ()) for c in climbable):
            return True
    return False


def _resolve_single_quicksilver(single_atom_input_types: set, can_reject: bool,
                                 metal_glyph_up: Optional[str]) -> bool:
    """True if a free (single-atom) Quicksilver is reachable."""
    return (
        ATOM_QUICKSILVER in single_atom_input_types
        or (can_reject and any(m in single_atom_input_types for m in METAL_CHAIN[1:]))
        or (can_reject and metal_glyph_up is not None and METAL_CHAIN[0] in single_atom_input_types)
    )


def _synthesizable_single(x: int, single_atom_input_types: set,
                           metal_glyph_up: Optional[str], metal_glyph_down: Optional[str],
                           single_salt: bool, single_quicksilver: bool, has_proliferation: bool,
                           needs_baron_duplication: bool, needs_dispersion: bool,
                           needs_animismus: bool, needs_unification: bool) -> bool:
    """True if a free single atom of type `x` is reachable directly from
    the puzzle's single-atom raw inputs, without ever passing through a bonded intermediate."""
    if x in single_atom_input_types:
        return True
    if x in METAL_CHAIN:
        i = METAL_CHAIN.index(x)
        if metal_glyph_down is not None and any(h in single_atom_input_types for h in METAL_CHAIN[i + 1:]):
            return True
        if metal_glyph_up is not None and any(lo in single_atom_input_types for lo in METAL_CHAIN[:i]):
            return True
        return has_proliferation and single_quicksilver
    if x == ATOM_SALT:
        return single_salt
    if x in ELEMENTALS:
        if needs_baron_duplication and single_salt:
            return True
        return needs_dispersion and ATOM_QUINTESSENCE in single_atom_input_types
    if x == ATOM_QUICKSILVER:
        return single_quicksilver
    if x in (ATOM_VITAE, ATOM_MORS):
        return needs_animismus and single_salt
    if x == ATOM_QUINTESSENCE:
        return (
            (needs_unification and ELEMENTALS.issubset(single_atom_input_types))
            or (needs_baron_duplication and single_salt)
        )
    return False


def _resolve_metal_chain_forcing(input_atom_types: set, output_atom_types: set,
                                  has_projection: bool, has_purification: bool,
                                  has_rejection: bool, has_division: bool,
                                  has_quicksilver: bool):
    """Which metal-chain direction(s) a solution is forced to buy, and the
    concrete/ambiguous glyph label(s) that resolves to. Walks each metal
    absent from input but present in output, checking whether it's
    reachable climbing up (Projection/Purification from a lower metal
    already in input) and/or down (Rejection from a higher metal, or
    Division from a metal whose fixed DIVISION_OUTPUTS pair reaches it) —
    forced up-only, forced down-only, either-direction-works, or (if
    neither) in need of Ravari+Proliferation instead. Also folds in the
    special case where Quicksilver itself is the missing output (only
    Rejection produces it — and if no metal above lead is present to
    reject, forces climbing lead up first as a bootstrap)."""
    have_up_glyph = has_projection or has_purification
    force_up = force_down = force_rejection = either_ok = needs_proliferation = False
    for i, m in enumerate(METAL_CHAIN):
        if m not in output_atom_types or m in input_atom_types:
            continue
        can_up = have_up_glyph and any(lower in input_atom_types for lower in METAL_CHAIN[:i])
        can_down_rej = has_rejection and any(higher in input_atom_types for higher in METAL_CHAIN[i + 1:])
        can_down_div = has_division and _division_reachable_from(m, input_atom_types, have_up_glyph)
        can_down_div_direct = has_division and _division_reachable_from(m, input_atom_types, False)
        can_down = can_down_rej or can_down_div
        if can_up and not can_down:
            force_up = True
        elif can_down and not can_up:
            force_down = True
            if can_down_rej and not can_down_div:
                force_rejection = True
            if can_down_div and not can_down_div_direct and not can_down_rej:
                force_up = True
        elif can_up and can_down:
            if can_down_rej or can_down_div_direct:
                either_ok = True
            else:
                force_up = True
        else:
            needs_proliferation = True

    if has_rejection and ATOM_QUICKSILVER in output_atom_types and ATOM_QUICKSILVER not in input_atom_types:
        force_down = True
        force_rejection = True
        if not any(m2 in input_atom_types for m2 in METAL_CHAIN[1:]):
            assert METAL_CHAIN[0] in input_atom_types
            force_up = True

    if not (force_up or either_ok):
        metal_glyph_up = None
    elif has_projection and has_purification:
        metal_glyph_up = "projection-or-purification" if has_quicksilver else "projection-or-purification-no-quicksilver"
    elif has_projection:
        metal_glyph_up = "projection"
        if not has_quicksilver:
            force_down = True
            if not force_up:
                metal_glyph_up = None
            else:
                force_rejection = True
    elif has_purification:
        metal_glyph_up = "purification"
    else:
        metal_glyph_up = None
    if not (force_down or either_ok):
        metal_glyph_down = None
    elif force_rejection:
        metal_glyph_down = "rejection"
    elif has_rejection and has_division:
        metal_glyph_down = "rejection-or-division"
    elif has_rejection:
        metal_glyph_down = "rejection"
    elif has_division:
        metal_glyph_down = "division"
    else:
        metal_glyph_down = None

    return force_up, force_down, either_ok, needs_proliferation, metal_glyph_up, metal_glyph_down


def _resolve_elemental_glyphs(pf: PuzzleFile, input_atom_types: set, output_atom_types: set):
    """The non-metal single-atom transform glyphs: calcification (salt
    from elemental), animismus (vitae/mors from salt), dispersion/
    unification (elemental<->quintessence), and the Baron+Duplication
    alternative to dispersion. Also returns the single-atom-input
    reachability sets these (and the metal-chain/bond-action glyphs) rely
    on to know whether a bonded intermediate would be needed instead of a
    free atom."""
    needs_animismus = (
        (ATOM_VITAE in output_atom_types and ATOM_VITAE not in input_atom_types)
        or (ATOM_MORS in output_atom_types and ATOM_MORS not in input_atom_types)
    )
    needs_unification = (ATOM_QUINTESSENCE in output_atom_types
                          and ATOM_QUINTESSENCE not in input_atom_types)
    needs_calcification = (
        (ATOM_SALT in output_atom_types or needs_animismus)
        and ATOM_SALT not in input_atom_types
    )
    needs_elemental = (
        bool((output_atom_types & ELEMENTALS) - input_atom_types)
        or (needs_unification and not ELEMENTALS.issubset(input_atom_types))
        or (needs_calcification and not bool(input_atom_types & ELEMENTALS))
    )
    # Dispersion is always cheaper/smaller
    needs_dispersion = needs_elemental and ATOM_QUINTESSENCE in input_atom_types
    if needs_calcification:
        assert (bool(input_atom_types & ELEMENTALS) or needs_dispersion)
    single_atom_input_types = {a.type for mol in pf.inputs if len(mol.atoms) == 1 for a in mol.atoms}
    single_salt = (
        ATOM_SALT in single_atom_input_types
        or (needs_calcification and (bool(single_atom_input_types & ELEMENTALS) or needs_dispersion))
    )
    single_quintessence = ATOM_QUINTESSENCE in single_atom_input_types
    single_all_elementals = ELEMENTALS.issubset(single_atom_input_types)
    needs_baron_duplication = needs_elemental and ATOM_QUINTESSENCE not in input_atom_types
    if needs_baron_duplication:
        assert bool(pf.parts_available & PART_BARON)
    return (needs_calcification, needs_animismus, needs_dispersion, needs_unification,
            needs_baron_duplication, single_atom_input_types, single_salt,
            single_quintessence, single_all_elementals)


def _resolve_bond_actions(pf: PuzzleFile, input_atom_types: set,
                           input_bond_sigs: dict, output_bond_sigs: dict,
                           have_up_glyph: bool, has_rejection: bool, has_division: bool,
                           metal_glyph_up: Optional[str], metal_glyph_down: Optional[str],
                           force_up: bool, force_down: bool,
                           needs_calcification: bool, needs_baron_duplication: bool,
                           needs_animismus: bool, needs_dispersion: bool, needs_unification: bool,
                           single_atom_input_types: set, single_salt: bool,
                           single_quintessence: bool, single_all_elementals: bool,
                           has_quicksilver: bool, can_get_quicksilver_via_rejection: bool,
                           needs_proliferation: bool):
    """Bonder/Unbonder/Bonder-Prisma requirements: per-atom-instance,
    geometry-aware bond-signature comparison, plus needs_unbonder's other sources
    — animismus/dispersion/unification each needing an unbonder when their raw
    material (salt / quintessence / all four elementals respectively) only exists bonded
    — a free-metal-output that requires an intermediate that only exists bonded."""
    can_project = metal_glyph_up is not None and "projection" in metal_glyph_up
    can_purify = metal_glyph_up is not None and "purification" in metal_glyph_up
    can_reject = metal_glyph_down is not None and "rejection" in metal_glyph_down
    if needs_proliferation:
        assert has_quicksilver or can_get_quicksilver_via_rejection
        assert pf.parts_available & PART_PROLIFERATION and pf.parts_available & PART_RAVARI
    single_quicksilver = _resolve_single_quicksilver(single_atom_input_types, can_reject, metal_glyph_up)

    def synthesizable_single(x):
        return _synthesizable_single(
            x, single_atom_input_types, metal_glyph_up, metal_glyph_down,
            single_salt, single_quicksilver, needs_proliferation,
            needs_baron_duplication, needs_dispersion, needs_animismus, needs_unification,
        )

    needs_debond, debond_extra = _needs_bond_action(
        output_bond_sigs, input_bond_sigs, reference_is_earlier=True,
        needs_calc=needs_calcification, needs_dup=needs_baron_duplication,
        has_rejection=can_reject, has_projection=can_project,
        synthesizable_single=synthesizable_single,
        single_atom_input_types=single_atom_input_types,
        metal_glyph_up=metal_glyph_up, metal_glyph_down=metal_glyph_down,
        single_salt=single_salt, single_quicksilver=single_quicksilver,
        needs_proliferation=needs_proliferation, needs_animismus=needs_animismus,
        needs_dispersion=needs_dispersion, needs_unification=needs_unification,
        pf=pf, relevant_part=PART_UNBONDER,
    )
    if not needs_debond:
        # todo: catch new required glyphs to avoid assert fails
        def elevated_synthesizable_single(x):
            return _synthesizable_single(
                x, single_atom_input_types, metal_glyph_up, metal_glyph_down,
                single_salt, single_quicksilver,
                needs_proliferation or "proliferation" in debond_extra,
                needs_baron_duplication or "duplication" in debond_extra,
                needs_dispersion or "dispersion" in debond_extra,
                needs_animismus or "animismus" in debond_extra,
                needs_unification or "unification" in debond_extra,
            )
        needs_debond = _needs_debond_coverage(
            pf, elevated_synthesizable_single,
            needs_calcification or "calcification" in debond_extra,
            needs_baron_duplication or "duplication" in debond_extra,
            can_reject or "rejection" in debond_extra,
            can_project or "projection" in debond_extra,
        )
        if needs_debond:
            debond_extra = set()
    single_metal_input_types = {
        a.type for mol in pf.inputs if len(mol.atoms) == 1 and mol.atoms[0].type in METAL_CHAIN
        for a in mol.atoms
    }
    single_metal_output_types = {
        a.type for mol in pf.outputs if len(mol.atoms) == 1 and mol.atoms[0].type in METAL_CHAIN
        for a in mol.atoms
    }
    metal_output_types = {a.type for mol in pf.outputs for a in mol.atoms if a.type in METAL_CHAIN}

    def metal_reachable(i, m, available):
        return (
            (have_up_glyph and any(lower in available for lower in METAL_CHAIN[:i]))
            or (has_rejection and any(higher in available for higher in METAL_CHAIN[i + 1:]))
            or (has_division and _division_reachable_from(m, available, have_up_glyph))
        )

    needs_metal_single_debond = any(
        m in single_metal_output_types
        and m not in single_metal_input_types
        and metal_reachable(i, m, input_atom_types)
        and not metal_reachable(i, m, single_metal_input_types)
        for i, m in enumerate(METAL_CHAIN)
    )
    would_need_via_projection = can_project and not single_quicksilver
    would_need_via_purification = can_purify and any(
        m in metal_output_types
        and m not in input_atom_types
        and not any(lower in single_metal_input_types for lower in METAL_CHAIN[:i])
        for i, m in enumerate(METAL_CHAIN)
    )
    needs_metal_up_debond = force_up and not force_down and (
        (would_need_via_projection and would_need_via_purification)
        if (can_project and can_purify) else
        (would_need_via_projection or would_need_via_purification)
    )
    would_need_via_division = metal_glyph_down == "division" and any(
        m in metal_output_types
        and m not in input_atom_types
        and not any(higher in single_metal_input_types for higher in METAL_CHAIN[i + 1:])
        for i, m in enumerate(METAL_CHAIN)
    )
    needs_metal_down_debond = force_down and not force_up and would_need_via_division
    needs_unbonder = (
        (needs_animismus and not single_salt)
        or (needs_dispersion and not single_quintessence)
        or (needs_unification and not (single_all_elementals or (needs_baron_duplication and single_salt)))
        or needs_metal_up_debond
        or needs_metal_down_debond
        or needs_debond
        or needs_metal_single_debond
    )
    needs_bonder, bonder_extra = _needs_bond_action(
        output_bond_sigs, input_bond_sigs, reference_is_earlier=False, normal_only=True,
        needs_calc=needs_calcification, needs_dup=needs_baron_duplication,
        has_rejection=can_reject, has_projection=can_project,
        pf=pf, relevant_part=PART_BONDER,
    )
    needs_bonder_prisma, prisma_extra = _needs_bond_action(
        output_bond_sigs, input_bond_sigs, reference_is_earlier=False, triplex_only=True,
        needs_calc=needs_calcification, needs_dup=needs_baron_duplication,
        has_rejection=can_reject, has_projection=can_project,
        pf=pf, relevant_part=PART_BONDER_PRISMA,
    )
    extra_needed = debond_extra | bonder_extra | prisma_extra
    assert not needs_unbonder or pf.parts_available & PART_UNBONDER
    assert not needs_bonder or pf.parts_available & PART_BONDER
    assert not needs_bonder_prisma or pf.parts_available & PART_BONDER_PRISMA
    return needs_bonder, needs_unbonder, needs_bonder_prisma, extra_needed


def _has_star_shape(bond_sigs: dict) -> bool:
    """True if some atom instance in `bond_sigs` is fully surrounded (all 6
    hex directions bonded) — the star-bond case, see P025-Water-Purifier."""
    return any(
        all(slot is not None for slot in sig)
        for sigs in bond_sigs.values()
        for sig in sigs
    )


def _has_tetra_shape(mols) -> bool:
    """True if some atom instance, in a molecule of exactly 4 atoms, has
    exactly 3 bonds spaced 120° apart (every other of the 6 hex
    directions) — the tetra-bond case, e.g. P024-Alcohol-Separation's water
    atom bonded to salt/salt/fire. Restricted to a bare 4-atom
    hub-and-3-leaves molecule: once other atoms are attached to the leaves
    (e.g. a ring — see P031b/P207's 7-atom wheel, whose "hub" atom has the
    same 3-bonds-at-120° signature but isn't this shape), a normal arm can
    approach without the extra reach this shape otherwise forces."""
    for mol in mols:
        if len(mol.atoms) != 4:
            continue
        for sigs in _bond_signatures_by_type([mol]).values():
            for sig in sigs:
                filled = frozenset(i for i, slot in enumerate(sig) if slot is not None)
                if filled in ({0, 2, 4}, {1, 3, 5}):
                    return True
    return False


def _tetra_hub_forced_by_reagent(pf: PuzzleFile, needed: dict) -> bool:
    """True if some output atom has exactly 3 bonds at 120° (the tetra
    signature, see _has_tetra_shape) where two of those three bonds
    already exist, pre-formed, in a single input reagent instance of the
    hub's own atom type — e.g. P015-Stamina-Potion's water hub, whose
    bonds to its two salts already exist in the "water bonded to two
    salts" input piece. Since there's no unbonder here to ever split that
    piece apart, those two bonds are unavoidably already present, and the
    only choice left is when to add the third (fresh) bond. If the fresh
    neighbor's atom type never appears bonded in *any* input reagent, nor
    is reachable by an in-place transform (calcification/duplication/
    rejection/projection — see _transform_source_types) from a type that
    does — no bonded piece can ever provide it, so it can only arrive as a
    bare single atom — then whichever moment that fresh bond is added, it
    always turns a pre-existing 2-bond piece plus one bare atom into a
    bare 4-atom tetra, regardless of build order. Unlike _has_tetra_shape,
    this isn't restricted to a standalone 4-atom molecule — the hub can
    sit inside a larger finished output (e.g. P015's 7-atom molecule),
    since the forcing argument is about the single bond action, not the
    shape's permanence.

    A hub can have more than one input reagent providing a different
    pre-bonded piece of its three edges (e.g. one reagent gives A-hub-B
    with C only reachable bare, another gives B-hub-C with A also
    reachable bonded elsewhere) — or a reagent providing just *one* edge
    (e.g. hub-A alone), leaving two fresh edges rather than one. Each such
    reagent is one *candidate* build order for this hub; a candidate is
    forcing only if *every* one of its fresh (not-yet-provided) edges is
    bare-only — with two fresh edges, whichever of the two is added last
    still completes a bare tetra, so both must be inescapable, not just
    one. The hub is only actually forced if *every* candidate across every
    reagent is forcing. If even one candidate has a fresh edge that could
    itself arrive already bonded to something else, that edge can simply
    be added last, bringing extra baggage along so the completing bond
    isn't bare — that candidate escapes the tetra, and since a solver can
    just choose that build order, the hub isn't forced at all, regardless
    of how forcing the other candidates look."""
    bonded_input_types = {a.type for mol in pf.inputs if len(mol.atoms) > 1 for a in mol.atoms}
    needs_calc = needed["calcification"]
    needs_dup = needed["baron_duplication"]
    has_rejection = needed["metal_glyph_down"] is not None and "rejection" in needed["metal_glyph_down"]
    has_projection = needed["metal_glyph_up"] is not None and "projection" in needed["metal_glyph_up"]
    input_sigs = []
    for in_mol in pf.inputs:
        in_pos_to_type, in_slots = _position_bond_signatures(in_mol)
        for in_pos, in_type in in_pos_to_type.items():
            input_sigs.append((in_type, in_slots[in_pos]))

    def leaf_escapes(leaf_type: int) -> bool:
        return any(
            s in bonded_input_types
            for s in _transform_source_types(leaf_type, needs_calc, needs_dup, has_rejection, has_projection)
        )

    for out_mol in pf.outputs:
        out_pos_to_type, out_slots = _position_bond_signatures(out_mol)
        for pos, hub_type in out_pos_to_type.items():
            sig = out_slots[pos]
            filled = frozenset(i for i, slot in enumerate(sig) if slot is not None)
            if filled not in ({0, 2, 4}, {1, 3, 5}):
                continue
            found_forcing = False
            found_escape = False
            for provided_size in (1, 2):
                for provided in itertools.combinations(filled, provided_size):
                    partial = [slot if i in provided else None for i, slot in enumerate(sig)]
                    reagent_provides = any(
                        in_type == hub_type
                        and sum(1 for s in in_sig if s is not None) == provided_size
                        and _bond_signature_compatible(partial, in_sig)
                        for in_type, in_sig in input_sigs
                    )
                    if not reagent_provides:
                        continue
                    fresh = filled - set(provided)
                    if any(leaf_escapes(sig[f][1]) for f in fresh):
                        found_escape = True
                    else:
                        found_forcing = True
            if found_forcing and not found_escape:
                return True
    return False


def _resolve_star_bond(pf: PuzzleFile, input_bond_sigs: dict, output_bond_sigs: dict,
                        needs_bonder: bool, needs_bonder_prisma: bool,
                        needs_unbonder: bool) -> tuple[bool, bool]:
    """An extra +2 track floor for a fully-surrounded (6-bonded) atom (see
    _has_star_shape) on either side of the recipe, type-agnostically (a
    ring whose center atom changes identity via a metal-chain transform,
    e.g. P022's iron->copper, still counts):

      - forward: the star is in the *output*, built entirely from
        single-atom inputs — the arm assembling it has to reach around the
        star's own bonds — but only when that isn't already accounted for
        by a bonder-prisma or unbonder requirement elsewhere.
      - reverse: the star is in the *input*, torn down to single-atom
        outputs — the arm disassembling it needs the same reach — but only
        when no bonder of any kind (plain or prisma) is already required
        elsewhere (an unbonder is expected here; it's what does the
        tearing-down and doesn't cancel this).

    "Built/torn down to single atoms" also holds for an isolated-chamber
    production puzzle whose narrowest usable conduit has capacity 1:
    nothing bonded can cross at all, so the star has to be built (or torn
    down) out of atoms that arrive/leave one at a time through the conduit,
    regardless of whether the puzzle's own reagents happen to be bonded.
    Returned as (forward, reverse) rather than a single bool because an
    isolated production puzzle needing both at once pays the +2 floor
    twice — once per chamber, since track in one chamber can't serve the
    other (see _tracks_access)."""
    built_from_singles = all(len(mol.atoms) == 1 for mol in pf.inputs)
    torn_down_to_singles = all(len(mol.atoms) == 1 for mol in pf.outputs)
    conduit_forces_singles = (
        pf.is_production and pf.is_isolated
        and pf.conduit_capacities and max(pf.conduit_capacities) == 1
    )
    forward = (
        _has_star_shape(output_bond_sigs)
        and (built_from_singles or conduit_forces_singles)
        and not needs_bonder_prisma
        and not needs_unbonder
    )
    reverse = (
        _has_star_shape(input_bond_sigs)
        and (torn_down_to_singles or conduit_forces_singles)
        and not needs_bonder
        and not needs_bonder_prisma
    )
    return forward, reverse


def _resolve_tetra_bond(pf: PuzzleFile, needed: dict, needs_bonder: bool, needs_bonder_prisma: bool,
                         needs_unbonder: bool) -> tuple[bool, bool]:
    """Bare 4-atom hub-and-3-leaves shape (see _has_tetra_shape), forward
    and reverse — mirrors _resolve_star_bond's two directions and
    exclusions exactly, but unlike star-bond, *neither* direction adds a
    track floor directly. See _tracks_access, where each instead
    escalates its own bond-action's access requirement from 1 to 2
    (reaching all 3 branches takes more than a single approach), and the
    ordinary access/track formula converts that into whatever extra track
    it actually costs — which can be nothing at all, if the puzzle already
    had enough access headroom (see P021-Courage-Potion, forward-tetra's
    exact mirror of P024-Alcohol-Separation's reverse case)."""
    built_from_singles = all(len(mol.atoms) == 1 for mol in pf.inputs)
    torn_down_to_singles = all(len(mol.atoms) == 1 for mol in pf.outputs)
    conduit_forces_singles = (
        pf.is_production and pf.is_isolated
        and pf.conduit_capacities and max(pf.conduit_capacities) == 1
    )
    forward = (
        (
            (_has_tetra_shape(pf.outputs) and (built_from_singles or conduit_forces_singles))
            or _tetra_hub_forced_by_reagent(pf, needed)
        )
        and not needs_bonder_prisma
        and not needs_unbonder
    )
    reverse = (
        _has_tetra_shape(pf.inputs)
        and (torn_down_to_singles or conduit_forces_singles)
        and not needs_bonder
        and not needs_bonder_prisma
    )
    return forward, reverse


def _needs_second_arm(pf: PuzzleFile, recipe: RecipeResult, needed: dict) -> bool:
    """Whether the production layout can't be worked by a single arm in a
    single chamber: either inputs/outputs are architecturally split
    (is_isolated), or the working area (plus space taken up by any conduit
    anchored in the biggest chamber, which the arm can't use) overflows the
    biggest chamber."""
    if not pf.is_production:
        return False
    if pf.is_isolated:
        return True
    assert len(pf.conduit_capacity_per_chamber) == len(pf.cabinet_sizes)
    biggest_chamber_conduit_capacity = pf.conduit_capacity_per_chamber[
        pf.cabinet_sizes.index(max(pf.cabinet_sizes))]
    return (
        area_lower_bound(pf, recipe, needed)[0] + biggest_chamber_conduit_capacity
        > max(pf.cabinet_sizes)
    )


def _needs_piston_arm(pf: PuzzleFile, recipe: RecipeResult, needed: dict, tracks: int) -> bool:
    """Whether the chamber has room for the puzzle's atoms/glyphs/conduits
    alone, but not for that plus the track the arm needs to reach
    everything — track is laid down as physical hexes, but (unlike area
    itself) only ever needs 1 extra hex beyond the arm's own base position,
    regardless of how many track segments get bought (see cost_lower_bound).
    A piston arm covers that access without laying down any track hexes,
    avoiding a full second arm."""
    if not pf.is_production or pf.is_isolated:
        return False
    assert len(pf.conduit_capacity_per_chamber) == len(pf.cabinet_sizes)
    biggest_chamber_conduit_capacity = pf.conduit_capacity_per_chamber[
        pf.cabinet_sizes.index(max(pf.cabinet_sizes))]
    base_area, note = area_lower_bound(pf, recipe, needed)
    return base_area - (1 if 'access' in note else 0) + biggest_chamber_conduit_capacity + tracks > max(pf.cabinet_sizes)


def _needed_parts(pf: PuzzleFile, recipe: RecipeResult) -> dict:
    """
    Which extra parts a solution is provably forced to buy, from atom-type
    and bond-type presence/absence between inputs and outputs, plus how
    much Track that forces (needed["tracks"], needed["access"] — see
    cost_lower_bound's docstring for the exact access/track rules).
    Shared by cost_lower_bound and area_lower_bound.
    """
    input_atom_types: set[int] = set()
    for mol in pf.inputs:
        input_atom_types.update(a.type for a in mol.atoms)
    output_atom_types: set[int] = set()
    for mol in pf.outputs:
        output_atom_types.update(a.type for a in mol.atoms)

    has_projection = bool(pf.parts_available & PART_PROJECTION)
    has_purification = bool(pf.parts_available & PART_PURIFICATION)
    has_rejection = bool(pf.parts_available & PART_REJECTION)
    has_division = bool(pf.parts_available & PART_DIVISION)
    have_up_glyph = has_projection or has_purification
    has_quicksilver = ATOM_QUICKSILVER in input_atom_types
    can_get_quicksilver_via_rejection = has_rejection and any(
        m2 in input_atom_types for m2 in METAL_CHAIN[1:]
    )
    # metals
    force_up, force_down, either_ok, needs_proliferation, metal_glyph_up, metal_glyph_down = (
        _resolve_metal_chain_forcing(
            input_atom_types, output_atom_types, has_projection, has_purification,
            has_rejection, has_division, has_quicksilver,
        )
    )
    # other atoms
    (needs_calcification, needs_animismus, needs_dispersion, needs_unification,
     needs_baron_duplication, single_atom_input_types, single_salt,
     single_quintessence, single_all_elementals) = _resolve_elemental_glyphs(
        pf, input_atom_types, output_atom_types,
    )
    # bonds
    input_bond_sigs = _bond_signatures_by_type(pf.inputs)
    output_bond_sigs = _bond_signatures_by_type(pf.outputs)
    needs_bonder, needs_unbonder, needs_bonder_prisma, extra_needed = _resolve_bond_actions(
        pf, input_atom_types, input_bond_sigs, output_bond_sigs,
        have_up_glyph, has_rejection, has_division, metal_glyph_up, metal_glyph_down, force_up, force_down,
        needs_calcification, needs_baron_duplication, needs_animismus, needs_dispersion, needs_unification,
        single_atom_input_types, single_salt, single_quintessence, single_all_elementals,
        has_quicksilver, can_get_quicksilver_via_rejection, needs_proliferation,
    )
    # production
    freespace_needs_unbonder = needs_unbonder
    freespace_needs_bonder = needs_bonder
    if pf.is_production and pf.is_isolated and pf.conduit_capacities:
        conduit_capacity = max(pf.conduit_capacities)
        needed_input_idxs = necessary_inputs(pf, recipe)
        if needed_input_idxs and conduit_capacity < max(len(pf.inputs[i].atoms) for i in needed_input_idxs):
            assert pf.parts_available & PART_UNBONDER
            needs_unbonder = True
        if pf.outputs and conduit_capacity < max(mol.atom_count() for mol in pf.outputs):
            assert pf.parts_available & PART_BONDER
            needs_bonder = True

    if "projection" in extra_needed and metal_glyph_up is None:
        metal_glyph_up = "projection"
        force_up = True
    if "rejection" in extra_needed and metal_glyph_down is None:
        metal_glyph_down = "rejection"
        force_down = True

    needed = {
        "bonder": needs_bonder,
        "unbonder": needs_unbonder,
        "bonder_prisma": needs_bonder_prisma,
        "calcification": needs_calcification or "calcification" in extra_needed,
        "baron_duplication": needs_baron_duplication,
        "bare_duplication": "duplication" in extra_needed,
        "star_bond_forward": False,
        "star_bond_reverse": False,
        "tetra_bond_forward": False,
        "tetra_bond_reverse": False,
        "metal_up_mandatory": force_up,
        "metal_down_mandatory": force_down,
        "metal_either": either_ok,
        "metal_glyph_up": metal_glyph_up,  # None | "projection" | "purification" | "projection-or-purification" | "projection-or-purification-no-quicksilver"
        "metal_glyph_down": metal_glyph_down,  # None | "rejection" | "division" | "rejection-or-division"
        "ravari_proliferation": needs_proliferation and has_quicksilver,
        "ravari_proliferation_reject": needs_proliferation and not has_quicksilver,
        "bare_proliferation": "proliferation" in extra_needed,
        "animismus": needs_animismus or "animismus" in extra_needed,
        "dispersion": needs_dispersion or "dispersion" in extra_needed,
        "unification": needs_unification or "unification" in extra_needed,
        "waste": False,
    }

    if not needed["bonder"] or pf.is_production:
        base_parts = 0
        if needs_calcification:
            base_parts |= PART_CALCIFICATION
        if needs_baron_duplication:
            base_parts |= PART_DUPLICATION
            if pf.parts_available & PART_BARON:
                base_parts |= PART_BARON
        if needs_animismus:
            base_parts |= PART_ANIMISMUS
        if needs_dispersion or needs_unification:
            base_parts |= PART_DISPERSION
        if needs_proliferation:
            base_parts |= PART_PROLIFERATION
            if pf.parts_available & PART_RAVARI:
                base_parts |= PART_RAVARI
        if metal_glyph_up == "projection":
            up_options = [PART_PROJECTION] if has_projection else [0]
        elif metal_glyph_up == "purification":
            up_options = [PART_PURIFICATION] if has_purification else [0]
        elif metal_glyph_up is not None:
            up_options = [p for p, has in
                          ((PART_PROJECTION, has_projection), (PART_PURIFICATION, has_purification)) if has]
        else:
            up_options = [0]

        if metal_glyph_down == "rejection":
            down_options = [PART_REJECTION] if has_rejection else [0]
        elif metal_glyph_down == "division":
            down_options = [PART_DIVISION] if has_division else [0]
        elif metal_glyph_down is not None:
            down_options = [p for p, has in
                            ((PART_REJECTION, has_rejection), (PART_DIVISION, has_division)) if has]
        else:
            down_options = [0]

        allowed_waste_types = {ATOM_FIRE} if needs_bonder_prisma else set()
        if needs_bonder_prisma and needs_calcification:
            allowed_waste_types |= {ATOM_SALT}
        if needs_bonder_prisma and needs_calcification and needs_baron_duplication:
            allowed_waste_types |= ELEMENTALS

        def waste_type_avoidable() -> bool:
            for up_opt in up_options:
                for down_opt in down_options:
                    restricted_pf = replace(pf, parts_available=base_parts | up_opt | down_opt)
                    try:
                        limited_recipe = solve_recipe_min_waste(restricted_pf)
                    except ValueError:
                        continue
                    if all(t in allowed_waste_types for t in limited_recipe.waste):
                        return True
            return False

        if not pf.is_production:
            needed["waste"] = not waste_type_avoidable()
        else:
            assert pf.cabinet_sizes
            needs_second_arm = _needs_second_arm(pf, recipe, needed)
            input_atoms = sum(pf.inputs[i].atom_count() for i in necessary_inputs(pf, recipe))
            smallest_output = min(
                6 * (mol.atom_count() - 1) if any(a.type == ATOM_REPEAT for a in mol.atoms)
                else mol.atom_count()
                for mol in pf.outputs
            )
            sizes = sorted(pf.cabinet_sizes, reverse=True)
            if needs_second_arm:
                assert len(sizes) >= 2
                free_space = sizes[0] + sizes[1] - input_atoms - smallest_output - 2
            else:
                free_space = sizes[0] - input_atoms - smallest_output - 1
            extra_chambers = len(sizes) - (2 if needs_second_arm else 1)
            if extra_chambers > 0:
                assert pf.conduit_capacities
                free_space += max(pf.conduit_capacities) * extra_chambers

            def acceptable(waste: Optional[dict]) -> bool:
                if waste is None:
                    return False
                return sum(waste.values()) <= free_space

            def waste_for(parts: int) -> Optional[dict]:
                try:
                    return solve_recipe_min_waste(replace(pf, parts_available=parts)).waste
                except ValueError:
                    return None

            def waste_for_excluding(reaction_name: str) -> Optional[dict]:

                try:
                    return solve_recipe_min_waste(pf, excluded_reactions=frozenset({reaction_name})).waste
                except ValueError:
                    return None

            already_proj = metal_glyph_up is not None and "projection" == metal_glyph_up
            already_pur = metal_glyph_up is not None and "purification" == metal_glyph_up
            already_rej = metal_glyph_down is not None and "rejection" == metal_glyph_down
            already_div = metal_glyph_down is not None and "division" == metal_glyph_down
            waste_candidates = [
                (n, p) for n, p, already in (
                    ("calcification", PART_CALCIFICATION, needs_calcification),
                    ("duplication", PART_DUPLICATION, needs_baron_duplication),
                    ("projection", PART_PROJECTION, already_proj),
                    ("purification", PART_PURIFICATION, already_pur),
                    ("rejection", PART_REJECTION, already_rej),
                    ("division", PART_DIVISION, already_div),
                    ("animismus", PART_ANIMISMUS, needs_animismus),
                    ("dispersion", PART_DISPERSION, needs_dispersion),
                    ("unification", PART_DISPERSION, needs_unification),
                    ("proliferation", PART_PROLIFERATION, needs_proliferation),
                )
                if (pf.parts_available & p) and not already
            ]

            already_fine = any(
                acceptable(waste_for(base_parts | up_opt | down_opt))
                for up_opt in up_options for down_opt in down_options
            )
            if already_fine:
                needed["waste"] = not waste_type_avoidable()
            elif pf.parts_available & PART_DISPOSAL:
                # tracks_access prefers disposal anyway
                needed["waste"] = True
            else:
                # Force a waste-less solution
                assert acceptable(waste_for(pf.parts_available))
                best_required = {
                    name for name, p in waste_candidates
                    if not acceptable(
                        waste_for_excluding(name) if name in ("dispersion", "unification")
                        else waste_for(pf.parts_available & ~p)
                    )
                }
                if "calcification" in best_required:
                    needed["calcification"] = True
                if "duplication" in best_required:
                    needed["bare_duplication"] = True
                proj = "projection" in best_required
                pur = "purification" in best_required
                assert not (proj and pur)
                assert not (already_proj and pur)
                assert not (already_pur and proj)
                if proj or pur:
                    needed["metal_up_mandatory"] = True
                if proj:
                    needed["metal_glyph_up"] = "projection"
                if pur:
                    needed["metal_glyph_up"] = "purification"
                rej = "rejection" in best_required
                div = "division" in best_required
                assert not (rej and div)
                assert not (already_rej and div)
                assert not (already_div and rej)
                if rej or div:
                    needed["metal_down_mandatory"] = True
                if rej:
                    needed["metal_glyph_down"] = "rejection"
                if div:
                    needed["metal_glyph_down"] = "division"
                if "animismus" in best_required:
                    needed["animismus"] = True
                if "dispersion" in best_required:
                    needed["dispersion"] = True
                if "unification" in best_required:
                    needed["unification"] = True
                if "proliferation" in best_required:
                    needed["bare_proliferation"] = True
                needed["waste"] = False

    bonder_signal = freespace_needs_bonder or (needed["waste"] and not (pf.parts_available & PART_DISPOSAL))
    needed["star_bond_forward"], needed["star_bond_reverse"] = _resolve_star_bond(
        pf, input_bond_sigs, output_bond_sigs,
        bonder_signal, needs_bonder_prisma, freespace_needs_unbonder,
    )
    needed["tetra_bond_forward"], needed["tetra_bond_reverse"] = _resolve_tetra_bond(
        pf, needed, bonder_signal, needs_bonder_prisma, freespace_needs_unbonder,
    )

    return needed


def _resolve_metal_glyphs(needed: dict, use_metal_up: bool, use_metal_down: bool) -> tuple[Optional[str], Optional[str]]:
    """Collapse _needed_parts's ambiguous metal_glyph_up/down labels
    ("...-or-...", "...-no-quicksilver") into the concrete glyph each
    direction resolves to, given which direction(s) are actually being bought"""
    metal_glyph_up = needed["metal_glyph_up"] if use_metal_up else None
    metal_glyph_down = needed["metal_glyph_down"] if use_metal_down else None
    if metal_glyph_up == "projection-or-purification-no-quicksilver":
        quicksilver_free = metal_glyph_down in ("rejection", "rejection-or-division")
        metal_glyph_up = "projection-or-purification" if quicksilver_free else "purification"
    if metal_glyph_up == "projection-or-purification":
        metal_glyph_up = "projection"
    if metal_glyph_down == "rejection-or-division":
        metal_glyph_down = "rejection"
    return metal_glyph_up, metal_glyph_down


def _tracks_access(pf: PuzzleFile, recipe: RecipeResult, needed: dict,
                    use_metal_up: bool, use_metal_down: bool) -> tuple[int, int, int, Optional[str], Optional[str], str, bool]:
    """Access-point/Track accounting for cost_lower_bound, parameterized by
    which metal-glyph direction(s) this call actually buys since
    purification/division each carry their own extra track/access cost
    beyond projection/rejection's."""
    metal_glyph_up, metal_glyph_down = _resolve_metal_glyphs(needed, use_metal_up, use_metal_down)
    needs_extra_rejection = needed["ravari_proliferation_reject"] and metal_glyph_down != "rejection"
    has_proliferation = (
        needed["ravari_proliferation"] or needed["ravari_proliferation_reject"] or needed["bare_proliferation"]
    )

    tracks = 0
    if needed["dispersion"]:
        tracks = max(tracks, 4)
    if needed["unification"]:
        tracks = max(tracks, 3)
    if metal_glyph_up == "purification":
        tracks = max(tracks, 3)
    if metal_glyph_down == "division":
        tracks = max(tracks, 2)
    if needed["animismus"]:
        tracks = max(tracks, 4)
    if needed["star_bond_forward"] and needed["star_bond_reverse"] and pf.is_production and pf.is_isolated:
        tracks = max(tracks, 4)
    elif needed["star_bond_forward"] or needed["star_bond_reverse"]:
        tracks = max(tracks, 2)

    needs_bonder = needed["bonder"]
    needs_waste_disposal = False
    if needed["waste"]:
        if pf.parts_available & PART_DISPOSAL:
            needs_waste_disposal = True
        else:
            assert pf.parts_available & PART_BONDER
            needs_bonder = True

    has_input_bonds = any(mol.bonds for mol in pf.inputs)
    input_access = len({
        i for i in necessary_inputs(pf, recipe)
        if not any(a.type == ATOM_REPEAT for a in pf.inputs[i].atoms)
    })
    output_access = sum(1 for mol in pf.outputs if not any(a.type == ATOM_REPEAT for a in mol.atoms))
    access = input_access + output_access
    if not needs_bonder and not needed["bonder_prisma"] and not has_input_bonds:
        # full access
        access_points = {
            "unbonder": 2, "calcification": 1, "baron_duplication": 1,
            "metal_glyph_up": 3 if metal_glyph_up == "purification" else 2,
            "metal_glyph_down": 3 if metal_glyph_down == "division" else 2,
            "proliferation": 2, "extra_rejection": 2,
            "dispersion": 4, "unification": 3, "animismus": 4,
            "waste_disposal": 1,
        }
        if has_proliferation:
            tracks = max(tracks, 4)
    else:
        # zero/partial access
        access_points = {"metal_glyph_up": 3 if metal_glyph_up == "purification" else 1,
                         "metal_glyph_down": 3 if metal_glyph_down == "division" else 1,
                         "proliferation": 2, "extra_rejection": 1,
                         "dispersion": 4, "animismus": 4, "unification": 3,
                         "waste_disposal": 1}
        multi_atom_output_types = {a.type for mol in pf.outputs if len(mol.atoms) > 1
                                    for a in mol.atoms} - {ATOM_REPEAT}
        multi_atom_input_types = {a.type for mol in pf.inputs if len(mol.atoms) > 1 for a in mol.atoms}
        needs_project = metal_glyph_up == "projection"
        needs_reject = metal_glyph_down == "rejection"
        has_single_atom_needing_bond = any(
            not any(s in multi_atom_input_types for s in
                    _transform_source_types(hidx_type, needed["calcification"], needed["baron_duplication"],
                                            needs_reject, needs_project))
            for hidx_type in multi_atom_output_types
        )
        has_two_single_atoms_needing_bond = any(
            all(
                not any(s in multi_atom_input_types for s in
                        _transform_source_types(t, needed["calcification"], needed["baron_duplication"],
                                                needs_reject, needs_project))
                for t in mol_types
            )
            for mol in pf.outputs if len(mol.atoms) > 1
            for mol_types in [{a.type for a in mol.atoms} - {ATOM_REPEAT}]
        )
        single_atom_output_types = {a.type for mol in pf.outputs if len(mol.atoms) == 1 for a in mol.atoms}
        single_atom_input_types = {a.type for mol in pf.inputs if len(mol.atoms) == 1 for a in mol.atoms}
        single_salt = (
                ATOM_SALT in single_atom_input_types
                or (needed["calcification"] and bool(single_atom_input_types & ELEMENTALS))
        )
        single_quicksilver = _resolve_single_quicksilver(
            single_atom_input_types, metal_glyph_down == "rejection", metal_glyph_up)

        def synthesizable_single(x):
            return _synthesizable_single(
                x, single_atom_input_types, metal_glyph_up, metal_glyph_down,
                single_salt, single_quicksilver, has_proliferation,
                needed["baron_duplication"], needed["dispersion"], needed["animismus"], needed["unification"],
            )

        has_single_atom_needing_debond = any(not synthesizable_single(x) for x in single_atom_output_types)
        has_two_single_atoms_needing_debond = any(
            all(
                not any(t in _transform_source_types(o, needed["calcification"], needed["baron_duplication"],
                                                     needs_reject, needs_project)
                        for o in multi_atom_output_types)
                for t in mol_types
            )
            for mol in pf.inputs if len(mol.atoms) > 1
            for mol_types in [{a.type for a in mol.atoms} - {ATOM_REPEAT}]
        ) or (
            not needs_bonder and not needed["bonder_prisma"]
            and all(len(mol.atoms) <= 2 for mol in pf.inputs)
        )
        # bonder/debonder
        if needs_bonder and has_single_atom_needing_bond:
            access_points["bonder"] = 2 if (has_two_single_atoms_needing_bond or needed["tetra_bond_forward"]) else 1
        if needed["bonder_prisma"] and has_single_atom_needing_bond and not needs_bonder:
            access_points["bonder_prisma"] = 2 if has_two_single_atoms_needing_bond else 1
        if needed["unbonder"] and has_single_atom_needing_debond:
            access_points["unbonder"] = 2 if (has_two_single_atoms_needing_debond or needed["tetra_bond_reverse"]) else 1
    access_parts = [f"inputs={input_access}", f"outputs={output_access}"]
    for key, points in access_points.items():
        if key == "metal_glyph_up":
            present = metal_glyph_up is not None
        elif key == "metal_glyph_down":
            present = metal_glyph_down is not None
        elif key == "proliferation":
            present = has_proliferation
        elif key == "extra_rejection":
            present = needs_extra_rejection
        elif key == "waste_disposal":
            present = needs_waste_disposal
        elif key == "bonder":
            present = needs_bonder
        else:
            present = bool(needed[key])
        if present:
            access += points
            access_parts.append(f"{key}={points}")
    access_note = " + ".join(access_parts)

    baseline_tracks = tracks
    free_access = 12 if (pf.is_production and pf.is_isolated) else 6
    if free_access < access:
        tracks = max(tracks, 1)
    while max(0, tracks - 1) * 6 < access - free_access:
            tracks += 1

    return access, tracks, baseline_tracks, metal_glyph_up, metal_glyph_down, access_note, needs_bonder


def cost_lower_bound(pf: PuzzleFile, recipe: RecipeResult) -> tuple[int, str]:
    """
    Minimum cost based on required parts, plus track if this puzzle's glyphs outnumber what a bare arm can reach.

    Every solution needs:
      - at least 1 arm (20g)
      - a bonder (10g) if some specific normal bond-type (unordered atom-type
        pair) appears in some output but not in any input
      - an unbonder (10g) if some specific bond-type (normal or triplex)
        appears in some input but not in any output
      - a triplex bonder (20g) if some specific triplex bond-type appears in
        some output but not in any input
      - a calcification glyph (10g) if salt appears in any output (or animisums is needed) but no salt (only elementals) in any input
      - a berlow wheel (30g) + duplication glyph (20g) = 50g total if any outputs contains (air/earth/fire/water) that is not in any input, and no quintessence in any input
      - i): a projection OR purification glyph (20g) if any metal output is not in any input but a lower-rank metal is
      - OR ii): a rejection OR division glyph (20 g) if any metal output is not in any input but a higher-rank metal is
      - a rejection glyph (20g) if quicksilver in any output but not in any input
      - ravari wheel (30g) + proliferation glyph (40g) = 70g total if any metal output is not in any input cant be reached via i)/ii) but a quicksilver (or other non-lead metal; via rejection) is
      - an animismus glyph (20g) if vitae or mors appear in any output but not in any inputs
        - 4 additional tracks (20g) to access the glyph
      - a dispersion glyph (20g) if any outputs contains (air/earth/fire/water) that is not in any input, and the berlow wheel is disabled
        - 4 additional tracks (20g) to access the glyph
      - a unification glyph (20g) if any output contains quintessence but none of the inputs
        - 3 additional tracks (15g) to access the glyph
      - track (5g each) if the access points the arm provides are less than the ones needed by glyphs.
        If bonds are available glyphs can be zero-access and only inputs/outputs need to be accessed.

    refs:
    https://biggieblog.com/optimizing-cost-in-opus-magnum/
    """
    needed = _needed_parts(pf, recipe)

    use_metal_up = needed["metal_up_mandatory"]
    use_metal_down = needed["metal_down_mandatory"]
    if needed["metal_either"] and not use_metal_up and not use_metal_down:
        access_up, tracks_up, baseline_up, glyph_up_up, glyph_down_up, note_up, bonder_up = _tracks_access(pf, recipe, needed, True, False)
        access_down, tracks_down, baseline_down, glyph_up_down, glyph_down_down, note_down, bonder_down = _tracks_access(pf, recipe, needed, False, True)
        if tracks_down < tracks_up:
            use_metal_down = True
            access, tracks, baseline_tracks, access_note, needs_bonder = access_down, tracks_down, baseline_down, note_down, bonder_down
            metal_glyph_up, metal_glyph_down = glyph_up_down, glyph_down_down
        else:
            use_metal_up = True
            access, tracks, baseline_tracks, access_note, needs_bonder = access_up, tracks_up, baseline_up, note_up, bonder_up
            metal_glyph_up, metal_glyph_down = glyph_up_up, glyph_down_up
    else:
        access, tracks, baseline_tracks, metal_glyph_up, metal_glyph_down, access_note, needs_bonder = _tracks_access(
            pf, recipe, needed, use_metal_up, use_metal_down)

    needs_second_arm = _needs_second_arm(pf, recipe, needed)
    uses_piston = False
    if needs_second_arm:
        g_lo = 40
        reasons = ["2×arm=40g"]
    elif _needs_piston_arm(pf, recipe, needed, tracks):
        uses_piston = True
        g_lo = 40
        reasons = ["1×piston-arm=40g"]
        if "access" in area_lower_bound(pf, recipe, needed)[1]:
            g_lo += 10
            reasons.append("2×track=10g")
    else:
        g_lo = 20
        reasons = ["1×arm=20g"]

    if needs_bonder:
        g_lo += 10
        reasons.append("1×bonder=10g")
    if needed["unbonder"]:
        g_lo += 10
        reasons.append("1×unbonder=10g")
    if needed["bonder_prisma"]:
        g_lo += 20
        reasons.append("1×bonder-prisma=20g")
    if needed["calcification"]:
        g_lo += 10
        reasons.append("1×calcification=10g")
    if needed["baron_duplication"]:
        g_lo += 50  # baron=30g + glyph-duplication=20g
        reasons.append("1×baron=30g + 1×duplication=20g")
    elif needed["bare_duplication"]:
        g_lo += 20  # glyph-duplication alone, no baron wheel
        reasons.append("1×duplication=20g")
    if use_metal_up:
        g_lo += 20  # projection=20g, purification=20g — same either way
        reasons.append(f"1×{metal_glyph_up}=20g")
    if use_metal_down:
        g_lo += 20  # rejection=20g, division=20g — same either way
        reasons.append(f"1×{metal_glyph_down}=20g")
    if needed["ravari_proliferation"] or needed["ravari_proliferation_reject"]:
        g_lo += 70  # ravari=30g + proliferation=40g
        reasons.append("1×ravari=30g + 1×proliferation=40g")
        if needed["ravari_proliferation_reject"] and not (use_metal_down and metal_glyph_down == "rejection"):
            g_lo += 20
            reasons.append("1×rejection=20g")
    elif needed["bare_proliferation"]:
        g_lo += 40  # proliferation glyph alone, no ravari wheel
        reasons.append("1×proliferation=40g")
    if needed["animismus"]:
        g_lo += 20
        reasons.append("1×animismus=20g")
    if needed["dispersion"]:
        g_lo += 20
        reasons.append("1×dispersion=20g")
    if needed["unification"]:
        g_lo += 20
        reasons.append("1×unification=20g")

    note = " + ".join(reasons)
    if tracks and not uses_piston:
        g_lo += tracks * 5
        if tracks > baseline_tracks:
            reasons.append(f"{tracks}×track={5*tracks}g (access={access}: {access_note})")
        else:
            reasons.append(f"{tracks}×track={5*tracks}g")
        note = " + ".join(reasons)

    return g_lo, note


def area_lower_bound(pf: PuzzleFile, recipe: RecipeResult, needed: Optional[dict] = None) -> tuple[int, str]:
    """
    Minimum area based on required parts plus one hex per required input/output atom.

    berlow wheel on tracks can expose the underlying hexes:
      - 0 track: 7+0, 2 track: 4+6, 3 track: 3+10
    """
    if needed is None:
        needed = _needed_parts(pf, recipe)
    use_metal_up = needed["metal_up_mandatory"]
    use_metal_down = needed["metal_down_mandatory"]
    metal_glyph_up, metal_glyph_down = _resolve_metal_glyphs(needed, True, True)
    up_area = 3 if metal_glyph_up == "purification" else 2
    down_area = 3 if metal_glyph_down == "division" else 2

    if needed["metal_either"] and not use_metal_up and not use_metal_down:
        if down_area < up_area:
            use_metal_down = True
        else:
            use_metal_up = True

    if pf.is_production and pf.is_isolated:
        a_lo = 2  # two arm bases, one per chamber
        reasons = ["2×arm=2"]
    else:
        a_lo = 1  # arm base
        reasons = ["1×arm=1"]

    if needed["bonder"]:
        a_lo += 2
        reasons.append("1×bonder=2")
    if needed["unbonder"]:
        a_lo += 2
        reasons.append("1×unbonder=2")
    if needed["bonder_prisma"]:
        a_lo += 3
        reasons.append("1×bonder-prisma=3")
    if needed["calcification"]:
        a_lo += 1
        reasons.append("1×calcification=1")
    if use_metal_up:
        a_lo += up_area
        reasons.append(f"1×{metal_glyph_up}={up_area}")
    if use_metal_down:
        a_lo += down_area
        reasons.append(f"1×{metal_glyph_down}={down_area}")
    if needed["animismus"]:
        a_lo += 4
        reasons.append("1×animismus=4")
    if needed["dispersion"]:
        a_lo += 5
        reasons.append("1×dispersion=5")
    if needed["unification"]:
        a_lo += 5
        reasons.append("1×unification=5")

    if (needed["animismus"] or needed["dispersion"] or needed["unification"]
            or (use_metal_up and metal_glyph_up == "purification")
            or needed["ravari_proliferation"] or needed["ravari_proliferation_reject"]):
        a_lo += 1
        reasons.append("access=1")

    input_atoms = sum(pf.inputs[i].atom_count() for i in necessary_inputs(pf, recipe))
    output_atoms = sum(
        6 * (mol.atom_count() - 1) if any(a.type == ATOM_REPEAT for a in mol.atoms)
        else mol.atom_count()
        for mol in pf.outputs
    )
    a_lo += input_atoms + output_atoms

    # Track use based on area count
    if needed["baron_duplication"]:
        a_lo += 2  # duplication
        # baron wheel base=1 + tracks / covered hexes
        if a_lo > 9:
            a_lo = max(13, a_lo + 3)
        elif a_lo > 3:
            a_lo = max(10, a_lo + 4)
        else:
            a_lo += 7
        reasons.append("1×baron=1 + 1×duplication=2")
    elif needed["bare_duplication"]:
        a_lo += 2  # duplication glyph alone, no baron wheel
        reasons.append("1×duplication=2")
    if needed["ravari_proliferation"] or needed["ravari_proliferation_reject"]:
        a_lo += 6  # todo: proliferation has 6 but only 2-3 can be under ravari
        if needed["ravari_proliferation_reject"] and not (use_metal_down and metal_glyph_down == "rejection"):
            a_lo += 2
            reasons.append("1×rejection=2")
        if a_lo > 9:
            a_lo = max(13, a_lo + 3)
        elif a_lo > 3:
            a_lo = max(10, a_lo + 4)
        else:
            a_lo += 7
        reasons.append("1×ravari=1 + 1×proliferation=2")
    elif needed["bare_proliferation"]:
        a_lo += 6  # proliferation glyph alone, no ravari wheel
        reasons.append("1×proliferation=6")

    reasons.append(f"input atoms={input_atoms}")
    reasons.append(f"output atoms={output_atoms}")

    return a_lo, " + ".join(reasons)


def instructions_lower_bound(pf: PuzzleFile) -> tuple[int, str]:
    """
    All non-trivial puzzles need grab, action, drop.
    4 is sufficient for all free space puzzles.

    refs:
    https://biggieblog.com/optimizing-instructions-in-opus-magnum/
    """
    return 3, ''
