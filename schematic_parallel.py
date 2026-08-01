"""
schematic_parallel.py — multi-process variant of schematic.reachable_states.

Parallelizes the backward BFS across CPU cores: neighbors()'s per-state
combinatorial enumeration (schematic.py's expensive part — some states take
fractions of a second, others several seconds) is embarrassingly parallel
across independent states, while the merge into the shared StateGraph
(dedup via graph._index, edge insertion) stays strictly single-writer in
the main process. Each state is expanded exactly once, its result list
processed in the same fixed order regardless of which worker produced it,
so parallel and serial graphs always match on node set and edge-pair set
(move *labels* can differ on same-priority ties — see the differential CLI
check for why that's cosmetic, not a correctness gap).

Uses continuous submission, not round/level barriers: a worker finishing
early immediately picks up whatever's newly discoverable instead of
waiting on the rest of a "round." Measured against a level-synchronous
version of this same design: it plateaus hard past ~4-8 workers (20
workers gave no improvement over 8), since a round can't advance until its
slowest member finishes and per-state cost varies wildly (microseconds to
seconds) — most of the round idles on one straggler. Continuous submission
removes that stall: just a standing pool of in-flight futures, refilled
from the main process the instant any one completes.

Deliberately duplicates ~25 lines of schematic.reachable_states' own
tracked_types-computation and match_indices/finalize logic rather than
extracting a shared helper, to keep this file fully additive and
schematic.py untouched.

Batches several newly-discovered states into one worker task (`batch_size`,
see reachable_states) instead of one task per state, amortizing fixed
per-task overhead (pickling, executor bookkeeping, the wait()/result()
round trip) — worthwhile once per-state cost is cheap enough that this
fixed overhead dominates. In-flight batches stay capped at one per worker
and refill the instant any one completes, preserving the same
continuous-submission property at the batch level.

Usage:
  python schematic_parallel.py --puzzle P040 --workers 8
  python schematic_parallel.py --puzzle P040 --workers 8 --batch-size 16
  python schematic_parallel.py --puzzle P040 --check   # diff against schematic.reachable_states
"""

import argparse
import os
import time
from collections import deque
from concurrent.futures import FIRST_COMPLETED, ProcessPoolExecutor, wait
from typing import Dict, List, Optional, Tuple

from puzzle_parser import PuzzleFile
from stoichiometry import RecipeResult, Reaction, build_reactions
from schematic import (
    StateGraph, State, initial_state, neighbors, _state_signature,
    _matches_raw_reagents, _is_drop_and_create, _forward_fallback,
)

# ── worker-process globals, set once via the pool's initializer ───────────
_W_REACTIONS: Dict[str, Reaction] = None
_W_PARTS_AVAILABLE: int = None
_W_TRACKED_TYPES: frozenset = None


def _init_worker(reactions: Dict[str, Reaction], parts_available: int, tracked_types: frozenset) -> None:
    global _W_REACTIONS, _W_PARTS_AVAILABLE, _W_TRACKED_TYPES
    _W_REACTIONS, _W_PARTS_AVAILABLE, _W_TRACKED_TYPES = reactions, parts_available, tracked_types


def _expand_with(state: State, reactions: Dict[str, Reaction], parts_available: int,
                  tracked_types: frozenset) -> List[Tuple[State, str, tuple]]:
    """neighbors() plus each result's canonical dedup key, computed once
    here rather than again in the main process's merge loop — offloads
    _state_signature's cost (color refinement, the search's single hottest
    computation) onto whichever process already has the result in hand."""
    return [(nxt, move, _state_signature(nxt, tracked_types))
            for nxt, move in neighbors(state, reactions, parts_available, tracked_types)]


def _expand_batch(states: List[State]):
    """Pool entrypoint — expands several states in one task, amortizing
    per-task overhead (pickling the call, executor bookkeeping, the
    wait()/result() round trip) across `len(states)` states instead of
    paying it once per state. Reads the same worker-process globals
    _init_worker set as _expand. Returns one results-list per input state,
    in the same order — the caller zips it back against the (idx, state)
    pairs it submitted."""
    return [_expand_with(s, _W_REACTIONS, _W_PARTS_AVAILABLE, _W_TRACKED_TYPES) for s in states]


def reachable_states(pf: PuzzleFile, recipe: RecipeResult, workers: Optional[int] = None,
                      batch_size: int = 8) -> StateGraph:
    """Drop-in parallel equivalent of schematic.reachable_states — same
    signature (plus `workers`/`batch_size`), same return type, same graph
    content. `workers=1` runs a plain in-process BFS (no pool at all), not a
    separate implementation of the merge logic — same `merge()` closure
    either way, just fed synchronously instead of via futures. `batch_size`
    caps how many newly-discovered states ride along in one worker task —
    see the in-flight-batches loop below for why this doesn't reintroduce
    the round-barrier straggler problem continuous submission was built to
    avoid."""
    resolved = workers if workers is not None else (os.process_cpu_count() or 1)

    # Duplicated from schematic.reachable_states by design (no changes to
    # schematic.py) — keep in sync if that logic ever changes there.
    reactions = {r.name: r for r in build_reactions(pf)}
    reactions.update(recipe.extra_reactions)
    tracked_types = frozenset(
        atype
        for name, r in reactions.items()
        if _is_drop_and_create(name) and recipe.reaction_counts.get(name, 0) > 0
        for atype, d in r.delta.items()
        if d > 0
    )

    start = initial_state(pf, recipe)
    graph = StateGraph()
    start_idx = graph.add_state(start, _state_signature(start, tracked_types))

    def merge(current_idx: int, results: List[Tuple[State, str, tuple]]) -> List[Tuple[int, State]]:
        new_states = []
        for nxt, move, key in results:
            nxt_idx = graph._index.get(key)
            if nxt_idx is None:
                nxt_idx = graph.add_state(nxt, key)
                new_states.append((nxt_idx, nxt))
            graph.add_edge(current_idx, nxt_idx, move)
        return new_states

    if resolved <= 1:
        queue = deque([(start_idx, start)])
        while queue:
            idx, state = queue.popleft()
            queue.extend(merge(idx, _expand_with(state, reactions, pf.parts_available, tracked_types)))
    else:
        pending: deque = deque([(start_idx, start)])
        futures: Dict = {}

        def _fill():
            while pending and len(futures) < resolved:
                chunk = [pending.popleft() for _ in range(min(batch_size, len(pending)))]
                idxs = [idx for idx, _s in chunk]
                states = [s for _idx, s in chunk]
                futures[pool.submit(_expand_batch, states)] = idxs

        with ProcessPoolExecutor(max_workers=resolved, initializer=_init_worker,
                                  initargs=(reactions, pf.parts_available, tracked_types)) as pool:
            _fill()
            while futures:
                done, _pending_futs = wait(futures, return_when=FIRST_COMPLETED)
                for fut in done:
                    idxs = futures.pop(fut)
                    for idx, results in zip(idxs, fut.result()):
                        pending.extend(merge(idx, results))
                _fill()

    # Duplicated tail from schematic.reachable_states (match_indices/assert).
    products_needed = pf.products_needed()
    match_indices = [i for i, s in enumerate(graph.states)
                      if _matches_raw_reagents(s, pf, recipe, products_needed, exact=True)]
    if not match_indices:
        match_indices = [i for i, s in enumerate(graph.states)
                          if _matches_raw_reagents(s, pf, recipe, products_needed)]
    if match_indices:
        graph.input_state_indices = match_indices
        graph.input_state_idx = match_indices[0]
    else:
        merged = _forward_fallback(pf, recipe, graph, reactions, pf.parts_available,
                                    tracked_types, products_needed)
        assert merged, (
            f"{pf.name!r}: no reachable state matches the actual raw reagents — "
            "the un-react/unbond move rules failed to find a valid full "
            "decomposition path that solve_recipe already proved exists"
        )
    return graph


def _fingerprint(graph: StateGraph):
    """Canonical (discovery-order-independent) content fingerprint for
    differential comparison: node keys as a set, edges as
    {(from_key, to_key): move}."""
    idx_to_key = {idx: key for key, idx in graph._index.items()}
    edges = {
        (idx_to_key[a], idx_to_key[b]): graph.edge_move[(a, b)]
        for a, lst in graph.edges.items() for b in lst
    }
    return set(idx_to_key.values()), edges


if __name__ == "__main__":
    import schematic
    from solve import find_puzzle_dir, load_puzzle
    from stoichiometry import solve_recipe, solve_recipe_combined

    parser = argparse.ArgumentParser(description="Parallel backward-BFS state-space search.")
    parser.add_argument("--puzzle", required=True, metavar="ID", help="Puzzle ID, e.g. P040")
    parser.add_argument("--workers", type=int, default=None,
                         help="Worker process count (default: os.process_cpu_count())")
    parser.add_argument("--batch-size", type=int, default=8,
                         help="States per worker task (default: 8)")
    parser.add_argument("--serial", action="store_true",
                         help="Also run schematic.reachable_states for an A/B timing comparison")
    parser.add_argument("--check", action="store_true",
                         help="Run both serial and parallel and diff their content (implies --serial)")
    args = parser.parse_args()

    folder, collection = find_puzzle_dir(args.puzzle)
    pf, puzzle_file = load_puzzle(folder, args.puzzle)
    try:
        recipe = solve_recipe_combined(pf)
    except NotImplementedError:
        recipe = solve_recipe(pf)

    t0 = time.perf_counter()
    graph = reachable_states(pf, recipe, workers=args.workers, batch_size=args.batch_size)
    parallel_time = time.perf_counter() - t0
    resolved_workers = args.workers if args.workers is not None else (os.process_cpu_count() or 1)
    n_edges = sum(len(v) for v in graph.edges.values())
    print(f"puzzle={args.puzzle} workers={resolved_workers} batch_size={args.batch_size} "
          f"states={len(graph.states)} edges={n_edges} time={parallel_time:.2f}s")

    if args.serial or args.check:
        t0 = time.perf_counter()
        serial_graph = schematic.reachable_states(pf, recipe)
        serial_time = time.perf_counter() - t0
        speedup = serial_time / parallel_time if parallel_time > 0 else float("inf")
        efficiency = speedup / resolved_workers if resolved_workers > 0 else 0
        print(f"serial: time={serial_time:.2f}s  speedup={speedup:.2f}x  "
              f"parallel_efficiency={efficiency:.2%}")

        if args.check:
            p_nodes, p_edges = _fingerprint(graph)
            s_nodes, s_edges = _fingerprint(serial_graph)
            # The correctness bar is the *pair* set (from_key, to_key), not
            # exact move-label text: when a target is reachable via more
            # than one same-priority move, add_edge's "first one wins" tie
            # keeps whichever the neighbor list happened to offer first —
            # and that ordering can differ across worker processes (e.g.
            # PYTHONHASHSEED varies per process), same as it can differ
            # across unrelated code-path changes within one process. This
            # never affects any downstream numeric bound (bounds.py only
            # uses edge_move for a human-readable note string) — verified
            # via full solve.py output diffs during this module's
            # development. Exact label match is reported as an FYI below,
            # not part of PASS/FAIL.
            p_pairs, s_pairs = set(p_edges), set(s_edges)
            pairs_ok = p_pairs == s_pairs
            labels_ok = p_edges == s_edges
            nodes_ok = p_nodes == s_nodes
            print(f"nodes match: {nodes_ok}  edge pairs match: {pairs_ok}  "
                  f"(exact move-label text also matches: {labels_ok})")
            if not nodes_ok:
                print(f"  serial-only nodes: {len(s_nodes - p_nodes)}  "
                      f"parallel-only nodes: {len(p_nodes - s_nodes)}")
            if not pairs_ok:
                print(f"  serial-only edge pairs: {len(s_pairs - p_pairs)}  "
                      f"parallel-only edge pairs: {len(p_pairs - s_pairs)}")
            print("PASS" if nodes_ok and pairs_ok else "FAIL")
