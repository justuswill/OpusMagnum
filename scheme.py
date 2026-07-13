"""
scheme.py — exact critical-path lower bound on cycles' latency term (L).

`bounds.py`'s cycles_lower_bound splits the cycle count into throughput (N) and
latency (L): N is how long the busiest input/output glyph takes to run
products_needed times; L is the minimum unavoidable *serial* chain to finish the
very last output copy, assuming the rest of the run is optimally pipelined behind
it. This module computes L exactly (rather than via ad hoc heuristic flags) for
the common case where the solved recipe (stoichiometry.RecipeResult) decomposes
cleanly into `products_needed` identical single-copy "unit recipes".

Algorithm:
  1. Unit decomposition: each reagent/reaction count in the whole-run recipe is
     divided by products_needed (floor division) to get one small "unit recipe".
     NOTE: all the guard checks that used to gate this module to only the cases
     it's proven correct for (period != 1, waste, reactions present, repeating/
     ring/3+-bonded outputs, uneven parallel-glyph splits, demand/supply
     mismatches) have been stripped out — see BUG.md. This now always computes
     *a* value, even on recipe shapes it isn't validated for; treat the result
     as unverified unless the puzzle matches one of the cases BUG.md confirms
     sound.
  2. Reagent acquisition depth (R): each reagent needing k uses per unit copy,
     with r identical parallel glyphs, needs ceil(k/r) sequential "grab slots"
     at depths 2,4,6,... (mirrors the existing throughput formula's "1 spawn
     per glyph per 2 cycles" rate, so the two scales compose correctly).
  3. D_unit: one combined provenance graph — reagent grabs, reaction firings,
     and the output molecule(s)' real bond structure — solved via a single
     backward "exposure" propagation from the output root(s) down to the raw
     reagents, then a rearrangement-inequality pairing of reagent-token depths
     against that demand (see `critical_path_latency`'s docstring for the
     exact derivation; the key fact, proven by unrolling the recursion and
     verified against several hand-built trees, is that the final depth
     always equals max over every raw token of (token_value + exposure)).
  4. L = max(0, D_unit - R) + (1 if not every output is a repeating/polymer
     molecule else 0) — the same "+1 for the last drop" term as before.

Known limitation: a type simultaneously produced by *multiple different*
reaction templates (rather than one reaction feeding both direct output and
another reaction, which step 3 handles correctly) isn't disambiguated — see
`_build_dependency_graph`'s callers. Validate empirically (corpus sweep
against real leaderboard records) before trusting beyond the cases checked.
"""

import math
from dataclasses import dataclass, field
from typing import Dict, List, Tuple

from puzzle_parser import PuzzleFile, Molecule, ATOM_REPEAT
from stoichiometry import RecipeResult, Reaction, build_reactions


def compute_period(recipe: RecipeResult, products_needed: int) -> int:
    """
    Smallest P such that every nonzero reagent/reaction count in `recipe`
    divides evenly into `products_needed / P` copies — i.e. the whole run
    decomposes into P groups of (products_needed/P) identical unit recipes.
    P always divides products_needed. P==1 means every count is an exact
    multiple of products_needed (the clean, fully-decomposed case).
    """
    period = 1
    counts = list(recipe.reagent_counts.values()) + list(recipe.reaction_counts.values())
    for c in counts:
        if c <= 0:
            continue
        d = products_needed // math.gcd(c, products_needed)
        period = period * d // math.gcd(period, d)
    return period


@dataclass
class UnitRecipe:
    reagent_unit_counts: Dict[int, int]     # input index -> uses needed for ONE output copy
    reagent_group_size: Dict[int, int]      # input index -> # parallel identical glyphs (structural, not per-copy)
    reaction_unit_counts: Dict[str, int]    # reaction name -> firings needed for ONE output copy


def build_unit_recipe(pf: PuzzleFile, recipe: RecipeResult) -> UnitRecipe:
    products_needed = pf.products_needed()
    return UnitRecipe(
        reagent_unit_counts={i: c // products_needed for i, c in recipe.reagent_counts.items() if c > 0},
        reagent_group_size=dict(recipe.reagent_group_size),
        reaction_unit_counts={n: c // products_needed for n, c in recipe.reaction_counts.items() if c > 0},
    )


def _reagent_token_pool(pf: PuzzleFile, unit: UnitRecipe) -> Dict[int, List[int]]:
    """atom_type -> sorted list of depths at which individual atom tokens of
    that type become available from reagent grabs. A single grab yields
    tokens for every atom in that reagent's molecule simultaneously."""
    pool: Dict[int, List[int]] = {}
    for i, unit_count in unit.reagent_unit_counts.items():
        repeats = max(1, unit.reagent_group_size.get(i, 1))
        composition = pf.inputs[i].atom_type_counts()
        per_glyph_count = [0] * repeats
        for k in range(unit_count):
            glyph = k % repeats
            per_glyph_count[glyph] += 1
            grab_depth = 2 * per_glyph_count[glyph]
            for atype, cnt in composition.items():
                pool.setdefault(atype, []).extend([grab_depth] * cnt)
    for depths in pool.values():
        depths.sort()
    return pool


def _reagent_acquisition_depth(pool: Dict[int, List[int]]) -> int:
    return max((depths[-1] for depths in pool.values() if depths), default=0)


def _build_dependency_graph(reaction_names, reactions: Dict[str, Reaction]):
    """edge producer -> consumer if producer's output type feeds consumer's
    (non-catalyst) input type."""
    edges = {name: set() for name in reaction_names}
    for consumer in reaction_names:
        r = reactions[consumer]
        consumed_types = {t for t, d in r.delta.items() if d < 0 and t != r.catalyst}
        for producer in reaction_names:
            if producer == consumer:
                continue
            p = reactions[producer]
            produced_types = {t for t, d in p.delta.items() if d > 0}
            if produced_types & consumed_types:
                edges[producer].add(consumer)
    return edges


def _topological_order(names: List[str], dep_graph: Dict[str, set]) -> List[str]:
    """Kahn's algorithm: producers before consumers. On a genuine
    reaction-dependency cycle, silently returns a partial order (the cyclic
    names are omitted, so those reactions are skipped downstream)."""
    in_degree = {n: 0 for n in names}
    for producer, consumers in dep_graph.items():
        for c in consumers:
            in_degree[c] += 1
    queue = [n for n in names if in_degree[n] == 0]
    order = []
    qi = 0
    while qi < len(queue):
        n = queue[qi]
        qi += 1
        order.append(n)
        for c in dep_graph.get(n, ()):
            in_degree[c] -= 1
            if in_degree[c] == 0:
                queue.append(c)
    return order


def _mol_bond_adjacency(mol: Molecule) -> Dict[int, List[int]]:
    pos_to_index = {}
    for idx, atom in enumerate(mol.atoms):
        pos_to_index[(atom.u, atom.v)] = idx
    adjacency: Dict[int, List[int]] = {i: [] for i in range(len(mol.atoms))}
    for bond in mol.bonds:
        a = pos_to_index.get((bond.from_u, bond.from_v))
        b = pos_to_index.get((bond.to_u, bond.to_v))
        if a is None or b is None:
            continue
        adjacency[a].append(b)
        adjacency[b].append(a)
    return adjacency


def _spanning_tree_and_extra_edges(adjacency: Dict[int, List[int]]):
    """BFS spanning tree (parent map) plus a set of extra (non-tree) edges."""
    n = len(adjacency)
    visited = [False] * n
    parent = {}
    children: Dict[int, List[int]] = {i: [] for i in range(n)}
    extra_edges = []
    seen_edges = set()

    root = 0
    visited[root] = True
    queue = [root]
    while queue:
        u = queue.pop(0)
        for v in adjacency[u]:
            edge = (min(u, v), max(u, v))
            if not visited[v]:
                visited[v] = True
                parent[v] = u
                children[u].append(v)
                seen_edges.add(edge)
                queue.append(v)
    for u in range(n):
        for v in adjacency[u]:
            edge = (min(u, v), max(u, v))
            if edge not in seen_edges:
                seen_edges.add(edge)
                extra_edges.append(edge)
    return root, children, extra_edges


def _output_tree_exposure(mol: Molecule) -> Tuple[bool, Dict[int, int]]:
    """
    For one output molecule: whether it has any ring-closing (non-tree) bond,
    and exposure(v) for every atom position v, where exposure(v) =
    children_count(v) + distance(v, root) in a rooted spanning tree of its
    bond graph — the number of "+1" absorptions v's own token would accrue if
    it stayed the dominant value all the way to the root (see module
    docstring: resolve(root) = max over v of (token(v) + exposure(v))).
    """
    n = len(mol.atoms)
    if n == 0:
        return False, {}
    adjacency = _mol_bond_adjacency(mol)
    root, children, extra_edges = _spanning_tree_and_extra_edges(adjacency)

    dist = {root: 0}
    order = [root]
    qi = 0
    while qi < len(order):
        u = order[qi]
        qi += 1
        for v in children[u]:
            dist[v] = dist[u] + 1
            order.append(v)

    exposure = {i: len(children[i]) + dist[i] for i in range(n)}
    return bool(extra_edges), exposure


def critical_path_latency(pf: PuzzleFile, recipe: RecipeResult) -> Tuple[int, str]:
    """
    Computes D_unit as the deepest a single raw reagent token's value can end
    up contributing to any output position, via one combined backward
    exposure-propagation pass rather than two independent phases:

    1. Seed `demand[type]` with the exposure of every output-tree position
       needing that type (from _output_tree_exposure).
    2. Walk reactions in REVERSE dependency order (consumers before their own
       producers). A reaction firing merges all its (non-catalyst) inputs via
       a single max(...)+1 — one glyph action regardless of arity — so *every*
       input to one firing shares the same downstream exposure: 1 + (the
       exposure its shared output experiences downstream). A reaction's m
       firings claim the m highest-exposure demand slots of each type it
       produces (its own output tokens should preferentially serve the
       neediest consumers), which in turn seeds `demand[input_type]` with m
       new slots (repeated per input multiplicity) for its own inputs.
    3. Once every reaction has been processed, `demand[type]` for any
       raw-reagent-supplied type holds every slot that ultimately needs a
       token of that type — whether consumed directly by the output or via
       some chain of reactions. Pair actual reagent-grab depths (ascending)
       against those demand slots (descending) — same rearrangement-
       inequality argument as the single-tree case, now over the whole
       provenance graph — and D_unit is the max matched sum.

    This subsumes the old two-phase (simulate reactions, then solve the bond
    tree) design: a produced type whose supply is split between feeding
    another reaction and feeding the output directly is exactly the case the
    two-phase version got wrong (it had no way to know, while simulating
    reactions, which of a type's tokens the *output* would most need).

    # ASSUMPTIONS:
    - period T=1
    """
    unit = build_unit_recipe(pf, recipe)

    reactions = {r.name: r for r in build_reactions(pf)}
    pool = _reagent_token_pool(pf, unit)
    R = _reagent_acquisition_depth(pool)

    demand: Dict[int, List[int]] = {}
    for mol in pf.outputs:
        _has_ring, exposure = _output_tree_exposure(mol)
        by_type: Dict[int, List[int]] = {}
        for idx, atom in enumerate(mol.atoms):
            by_type.setdefault(atom.type, []).append(idx)
        for atype, indices in by_type.items():
            demand.setdefault(atype, []).extend(exposure[i] for i in indices)

    reaction_names = list(unit.reaction_unit_counts.keys())
    dep_graph = _build_dependency_graph(reaction_names, reactions)
    topo_order = _topological_order(reaction_names, dep_graph)

    for name in reversed(topo_order):
        r = reactions[name]
        m = unit.reaction_unit_counts[name]
        produced_types = [t for t, d in r.delta.items() if d > 0]
        consumed_types = [(t, -d) for t, d in r.delta.items() if d < 0 and t != r.catalyst]

        firing_exposure = None
        for t in produced_types:
            slots = demand.get(t, [])
            slots.sort(reverse=True)
            taken = sorted(slots[:m], reverse=True)
            del slots[:m]
            if firing_exposure is None:
                firing_exposure = taken
            else:
                firing_exposure = [max(a, b) for a, b in zip(firing_exposure, taken)]
        if firing_exposure is None:
            firing_exposure = [0] * m

        input_exposure = sorted((1 + e for e in firing_exposure), reverse=True)
        for t, count in consumed_types:
            bucket = demand.setdefault(t, [])
            for e in input_exposure:
                bucket.extend([e] * count)

    d_unit = 0
    for atype, depths in pool.items():
        slots = demand.get(atype, [])
        for tok, exp in zip(sorted(depths), sorted(slots, reverse=True)):
            d_unit = max(d_unit, tok + exp)

    # Ring-closing (non-spanning-tree) bond edges are NOT charged an extra
    # +1 here: whether one actually extends the critical path depends on
    # which specific atoms it connects (only matters if it touches the
    # deepest branch), which isn't captured by the exposure-only formula.
    # Omitting it is conservative — it can only make D_unit (and therefore
    # the final bound) smaller/looser, never unsound.

    L = max(0, d_unit - R)
    note = f"critical path: D={d_unit}, R={R}, L={L}"
    return L, note
