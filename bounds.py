"""
bounds.py — analytic lower bounds on the metrics of an optimal solution.

These are provable floors (cost, cycles, area, instructions) derived from the
puzzle's inputs/outputs/available parts alone — no search, no simulation.
See PLAN.md §7.4/§7.5: a metric is *proven optimal* when one of these bounds
equals the best known record.
"""

import itertools
import math
import re
from collections import Counter, deque
from typing import Optional
from weakref import WeakKeyDictionary

from puzzle_parser import (
    PuzzleFile, ATOM_NAMES,
    ATOM_SALT, ATOM_VITAE, ATOM_MORS, ATOM_REPEAT, ATOM_QUINTESSENCE, ATOM_QUICKSILVER,
    ELEMENTALS,
    PART_BARON, PART_PROJECTION, PART_PURIFICATION,
    alternate_repeat_puzzle,
)
from stoichiometry import (
    RecipeResult, METAL_CHAIN, necessary_inputs, describe_molecule,
    solve_recipe, solve_recipe_combined, solve_recipe_cheap,
)
from schematic import StateGraph, molecule_signature, _is_drop_and_create
from schematic_parallel import reachable_states

_cadence_latency_memo: "WeakKeyDictionary[StateGraph, tuple]" = WeakKeyDictionary()


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

    num_blocks = max(
        math.ceil((recipe.reagent_counts[i] // (pf.products_needed() / len(paths))) / recipe.reagent_group_size.get(i, 1))
        for i in reagent_sig.values()
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
    built for one specific recipe, and callers — notably solve.py's
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
        unit_count = count // products_needed
        period = math.lcm(period, repeats // math.gcd(unit_count, repeats))

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


def _bond_type_counts(mols):
    """(normal_counts, triplex_counts): atom-type-pair (frozenset) -> bond
    count, across all molecules, split by bond kind. A bond between two atoms
    of the same type collapses to a size-1 frozenset key; count still tracks
    multiplicity."""
    normal = Counter()
    triplex = Counter()
    for mol in mols:
        pos_to_type = {(a.u, a.v): a.type for a in mol.atoms}
        for b in mol.bonds:
            t1 = pos_to_type.get((b.from_u, b.from_v))
            t2 = pos_to_type.get((b.to_u, b.to_v))
            if t1 is None or t2 is None:
                continue
            key = frozenset((t1, t2))
            if b.is_normal:
                normal[key] += 1
            if b.is_triplex:
                triplex[key] += 1
    return normal, triplex


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

    N = products_needed
    note = f"output throughput: {products_needed} products, 1/cycle"

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
        if cycles > N:
            N = cycles
            breakdown = " + ".join(
                f"{recipe.reagent_group_size.get(i, 1)}x{ATOM_NAMES[next(iter(pf.inputs[i].atom_type_counts()))]}"
                for i in member_indices
            )
            note = (f"input throughput {N}: {total_count}x needed from combined "
                    f"{total_repeats} duplicates ({breakdown})")

    for i, count in recipe.reagent_counts.items():
        if i in combined_reagent_indices:
            continue  # already accounted for, pooled with the rest of its combined group above
        if count <= 0:
            continue
        repeats = recipe.reagent_group_size.get(i, 1)
        cycles = 2 * math.ceil(count / repeats)
        if cycles > N:
            N = cycles
            note = (f"input throughput {N}: {count}x needed from input {i}"
                    + (f" with {repeats} duplicates" if repeats > 1 else ""))

    L, spine_note = _cadence_latency(pf, recipe, states)
    latency_parts = [spine_note]
    all_repeating = all(any(a.type == ATOM_REPEAT for a in mol.atoms) for mol in pf.outputs)
    if not all_repeating:
        latency_parts.append("drop=1")  # last drop isn't pipelined away unless every output repeats
        L += 1
    latency_note = f"input latency {L}: " + " + ".join(latency_parts)

    return N + L, f"{note}, {latency_note}"


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
    derived soundly. Used solely to tighten solve.py's SUM interval
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


def _needed_parts(pf: PuzzleFile, recipe: RecipeResult) -> dict:
    """
    Which extra parts a solution is provably forced to buy, from atom-type
    and bond-type presence/absence between inputs and outputs, plus how
    much Track that forces (needed["tracks"], needed["access"] — see
    cost_lower_bound's docstring for the exact access/track rules). Shared
    by cost_lower_bound and area_lower_bound.
    """
    # Collect atom types and bond types across inputs and outputs
    input_atom_types: set[int] = set()
    for mol in pf.inputs:
        input_atom_types.update(a.type for a in mol.atoms)

    output_atom_types: set[int] = set()
    for mol in pf.outputs:
        output_atom_types.update(a.type for a in mol.atoms)

    input_normal_bonds, input_triplex_bonds = _bond_type_counts(pf.inputs)
    output_normal_bonds, output_triplex_bonds = _bond_type_counts(pf.outputs)
    input_bonds = set(input_normal_bonds) | set(input_triplex_bonds)
    output_bonds = set(output_normal_bonds) | set(output_triplex_bonds)

    def _new(need_keys, have_keys):
        """True if some bond-type in `need_keys` doesn't appear at all in `have_keys`."""
        return bool(need_keys - have_keys)

    needs_elemental = bool((output_atom_types & ELEMENTALS) - input_atom_types)
    needs_metal = any(
        m in output_atom_types and m not in input_atom_types
        and any(lower in input_atom_types for lower in METAL_CHAIN[:i])
        for i, m in enumerate(METAL_CHAIN)
    )
    has_projection = bool(pf.parts_available & PART_PROJECTION)
    has_purification = bool(pf.parts_available & PART_PURIFICATION)
    has_quicksilver = ATOM_QUICKSILVER in input_atom_types
    if not needs_metal:
        metal_glyph = None
    elif has_projection and has_purification:
        metal_glyph = "purification" if not has_quicksilver else "projection-or-purification"
    elif has_projection:
        metal_glyph = "projection"
    elif has_purification:
        metal_glyph = "purification"
    else:
        metal_glyph = None  # neither available — out of scope (see rejection/division)

    needed = {
        "bonder": _new(set(output_normal_bonds), set(input_normal_bonds)),
        "unbonder": _new(input_bonds, output_bonds),
        "bonder_prisma": _new(set(output_triplex_bonds), set(input_triplex_bonds)),
        "calcification": (ATOM_SALT in output_atom_types and ATOM_SALT not in input_atom_types
                           and bool(input_atom_types & ELEMENTALS)),
        "baron_duplication": (
            needs_elemental
            and ATOM_QUINTESSENCE not in input_atom_types
            and bool(pf.parts_available & PART_BARON)
        ),
        "metal_glyph": metal_glyph,  # None | "projection" | "purification" | "either"
        "animismus": (
            (ATOM_VITAE in output_atom_types and ATOM_VITAE not in input_atom_types)
            or (ATOM_MORS in output_atom_types and ATOM_MORS not in input_atom_types)
        ),
        "dispersion": needs_elemental and not bool(pf.parts_available & PART_BARON),
        "unification": (ATOM_QUINTESSENCE in output_atom_types
                         and ATOM_QUINTESSENCE not in input_atom_types),
    }

    # Access/Track: animismus/dispersion/unification/purification each have their own minimum track count
    tracks = 0
    if needed["dispersion"]:
        tracks = max(tracks, 2)
    if needed["unification"]:
        tracks = max(tracks, 3)
    if needed["metal_glyph"] == "purification":
        tracks = max(tracks, 3)
    if needed["animismus"]:
        tracks = max(tracks, 4)

    has_input_bonds = any(mol.bonds for mol in pf.inputs)
    access = len({
        i for i, count in recipe.reagent_counts.items()
        if count > 0 and not any(a.type == ATOM_REPEAT for a in pf.inputs[i].atoms)
    }) + sum(1 for mol in pf.outputs if not any(a.type == ATOM_REPEAT for a in mol.atoms))
    if not needed["bonder"] and not needed["bonder_prisma"] and not has_input_bonds:
        # full access
        access_points = {
            "unbonder": 2, "calcification": 1, "baron_duplication": 1, "metal_glyph": 2,
            "dispersion": 4, "unification": 3, "animismus": 4,
        }
    else:
        # zero/partial access
        access_points = {"metal_glyph": 1, "dispersion": 4, "animismus": 4, "unification": 3}
        if needed["bonder"] and not has_input_bonds:
            access_points["bonder"] = 2
        if needed["bonder_prisma"] and not has_input_bonds:
            access_points["bonder_prisma"] = 2 if not needed["bonder"] else 1
        multi_atom_output_types = {a.type for mol in pf.outputs if len(mol.atoms) > 1
                                    for a in mol.atoms} - {ATOM_REPEAT}
        multi_atom_input_types = {a.type for mol in pf.inputs if len(mol.atoms) > 1 for a in mol.atoms}
        bondable_single_types = (multi_atom_output_types - multi_atom_input_types) | {
            METAL_CHAIN[idx]
            for idx in range(len(METAL_CHAIN))
            for hidx in range(idx + 1, len(METAL_CHAIN))
            if METAL_CHAIN[hidx] in multi_atom_output_types
            and not any(METAL_CHAIN[j] in multi_atom_input_types for j in range(idx, hidx + 1))
        }
        has_single_atom_input = any(
            len(pf.inputs[i].atoms) == 1 and pf.inputs[i].atoms[0].type in bondable_single_types
            for i, count in recipe.reagent_counts.items() if count > 0
        )
        if has_single_atom_input and (needed["bonder"] or needed["bonder_prisma"]):
            access += 1
    if needed["metal_glyph"] == "purification":
        access_points["metal_glyph"] = 3
    for key, points in access_points.items():
        value = needed[key]
        present = (value is not None) if key == "metal_glyph" else bool(value)
        if present:
            access += points

    baseline_tracks = tracks
    if tracks:
        while tracks * 6 < access:
            tracks += 1
    elif access > 6:
        tracks = 1
        while tracks * 6 < access:
            tracks += 1

    needed["access"] = access
    needed["tracks"] = tracks
    needed["baseline_tracks"] = baseline_tracks
    return needed


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
      - a calcification glyph (10g) if salt appears in any output but no salt (only elementals) in any input
      - a berlow wheel (30g) + duplication glyph (20g) = 50g total if any outputs contains (air/earth/fire/water) that is not in any input, and no quintessence in any input
      - a projection OR purification glyph (20g) if any metal output is not in any input but a lower-rank metal is
      - an animismus glyph (20g) if vitae or mors appear in any output but not in any inputs
        - 4 additional tracks (20g) to access the glyph
      - a dispersion glyph (20g) if any outputs contains (air/earth/fire/water) that is not in any input, and the berlow wheel is disabled
        - 4 additional tracks (20g) to access the glyph
      - a unification glyph (20g) if any output contains quintessence but none of the inputs
        - 3 additional tracks (15g) to access the glyph
      - track (5g each) if the access points the arm provides are less than the ones needed by glyphs.
        If bonds are available glyphs can be zero-access and only inputs/outputs need to be accessed.
      # ignored for now:
      - DLC:
          - a rejection glyph (20 g) if ...
          - a division glyph (20 g) if ...
          - ravari wheel (30g) + proliferation glyph (20g) = 50g total if ...

    refs:
    https://biggieblog.com/optimizing-cost-in-opus-magnum/
    """
    needed = _needed_parts(pf, recipe)

    g_lo = 20
    reasons = ["1×arm=20g"]

    if needed["bonder"]:
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
        reasons.append("1×baron=30g + 1×glyph-duplication=20g")
    if needed["metal_glyph"] is not None:
        g_lo += 20  # projection=20g, purification=20g — same either way
        reasons.append(f"1×{needed['metal_glyph']}=20g")
    if needed["animismus"]:
        g_lo += 20
        reasons.append("1×animismus=20g")
    if needed["dispersion"]:
        g_lo += 20
        reasons.append("1×dispersion=20g")
    if needed["unification"]:
        g_lo += 20
        reasons.append("1×unification=20g")

    if needed["tracks"]:
        g_lo += needed["tracks"] * 5
        if needed["tracks"] > needed["baseline_tracks"]:
            reasons.append(f"{needed['tracks']}×track=5g (access={needed['access']})")
        else:
            reasons.append(f"{needed['tracks']}×track=5g")

    return g_lo, " + ".join(reasons)


def area_lower_bound(pf: PuzzleFile, recipe: RecipeResult) -> tuple[int, str]:
    """
    Minimum area based on required parts plus one hex per required input/output atom.

    berlow wheel on tracks can expose the underlying hexes:
      - 0 track: 7+0, 2 track: 4+6, 3 track: 3+10
    """
    needed = _needed_parts(pf, recipe)

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
    metal_glyph = needed["metal_glyph"]
    if metal_glyph is not None:
        area = 3 if metal_glyph == "purification" else 2
        a_lo += area
        reasons.append(f"1×{metal_glyph}={area}")
    if needed["animismus"]:
        a_lo += 4
        reasons.append("1×animismus=4")
    if needed["dispersion"]:
        a_lo += 5
        reasons.append("1×dispersion=5")
    if needed["unification"]:
        a_lo += 5
        reasons.append("1×unification=5")

    if needed["animismus"] or needed["dispersion"] or needed["unification"] or needed["metal_glyph"] == "purification":
        a_lo += 1
        reasons.append("access=1")

    input_atoms = sum(pf.inputs[i].atom_count() for i in necessary_inputs(pf, recipe))
    products_needed = pf.products_needed()
    output_atoms = sum(
        products_needed * (mol.atom_count() - 1) if any(a.type == ATOM_REPEAT for a in mol.atoms)
        else mol.atom_count()
        for mol in pf.outputs
    )
    a_lo += input_atoms + output_atoms

    # Track use based on area count
    if needed["baron_duplication"]:
        a_lo += 2  # glyph-duplication footprint=2
        # baron wheel base=1 + tracks / covered hexes
        if a_lo > 9:
            a_lo = max(13, a_lo + 3)
        elif a_lo > 3:
            a_lo = max(10, a_lo + 4)
        else:
            a_lo += 7
        reasons.append("1×baron=1 + 1×glyph-duplication=2")

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
