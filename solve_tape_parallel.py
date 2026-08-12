"""
solve_tape_parallel.py — multi-process variant of solve_tape.solve_tape.

solve_tape.solve_tape's own BFS gets its O(1)-per-candidate speed from
sim_bridge.SimState.snapshot()/try_step(): a snapshot is a raw pointer into
one process's libsearch.so heap (search_shim.c's struct search_snapshot),
meaningless in any other process — it cannot be pickled and handed to a
worker. So unlike schematic_parallel.py (whose States are plain, trivially
picklable Python objects), what crosses the process boundary here is each
node's (rotation, path) pair — small and cheap to pickle — and every worker
replays that path from its own root via reset_and_replay() (one C loop call
covering the whole path, sim_bridge.py's own module docstring) to reach the
node's state before expanding it. Losing the snapshot's O(1) restore is the
cost of parallelizing this particular search; each worker still only pays
that replay once per node (not once per candidate — see _expand_node's own
snapshot, mirroring solve_tape.solve_tape's node_snap), and the replay cost
is spread across `workers` CPU cores instead of serialized onto one.

(A serialized-snapshot variant — shipping a node's state as plain bytes,
via sim_bridge.SimState.serialize_snapshot()/restore_from_bytes(), so any
worker could restore it in O(1) instead of replaying — was tried and
measured no faster than the replay-based version here on a real P012 run
across workers=1..20: the per-candidate snapshot()+serialize()+
free_snapshot() calls needed to produce that bytes payload (candidates
per node and path depth are similar orders of magnitude at the depths
tested) cancel out the replay savings. Reverted rather than kept as
unused-but-idle complexity.)

Level-synchronous, not continuous-submission like schematic_parallel.py:
solve_tape.solve_tape's early-termination pruning (once no goal at a deeper
level could possibly beat the current best score, stop — see its own
docstring) needs to know the *complete* best score found through depth D
before it can safely decide whether to expand depth D+1 at all, which
requires a hard barrier between levels. schematic_parallel.py deliberately
avoids barriers because its per-state cost varies wildly (microseconds to
seconds) and a round stalls on its slowest member; solve_tape's per-node
cost is far more uniform (a small, bounded candidate-instruction count per
node, each a fixed-cost single-cycle step), so a level barrier here doesn't
have that straggler problem.

workers<=1 delegates straight to solve_tape.solve_tape (real snapshots, no
pool) — the level-synchronous replay-based path here is strictly more
expensive per node than that, so there's no reason to run it in-process.

Usage: no CLI of its own — solve_cost.py's --workers flag is the entry
point (mirrors solve_tape.py's own "no CLI of its own" scoping).
"""

import os
from typing import List, Optional, Tuple

from concurrent.futures import ProcessPoolExecutor
from tqdm import tqdm

from sim_bridge import SimState, Edge
# Aliased, not `import solve_tape` — this module defines its own top-level
# solve_tape() (drop-in name match with solve_tape.solve_tape, mirroring
# schematic_parallel.reachable_states' naming against schematic
# .reachable_states), which would otherwise shadow a plain module import of
# the same name the instant Python finishes executing this file.
from solve_tape import (solve_tape as _serial_solve_tape, _candidate_instructions,
                         reset_overhead, _verify_full_solution)

# ── worker-process globals, set once via the pool's initializer ───────────
_W_SIM: Optional[SimState] = None
_W_TEMPLATE_PATH: Optional[str] = None
_W_PRODUCTS_NEEDED: int = None
_W_TRACK_LOOP_LENGTH: int = None


def _init_worker(puzzle_path: str, template_path: str, max_cycles: int,
                  products_needed: int, track_loop_length: int) -> None:
    global _W_SIM, _W_TEMPLATE_PATH, _W_PRODUCTS_NEEDED, _W_TRACK_LOOP_LENGTH
    _W_SIM = SimState(puzzle_path, template_path, max_cycles=max_cycles)
    _W_TEMPLATE_PATH = template_path
    _W_PRODUCTS_NEEDED = products_needed
    _W_TRACK_LOOP_LENGTH = track_loop_length


def _expand_node(sim: SimState, rotation: int, path: List[Edge],
                  products_needed: int, track_loop_length: int, template_path: str):
    """One node's worth of solve_tape.solve_tape's per-dequeue body: replay
    to (rotation, path)'s state, try every legal candidate instruction, and
    report each non-colliding one's (new_path, key, output_dropped) plus
    this node's own best (score, new_path) if any candidate both dropped
    an output and verified (or None). Unlike solve_tape.solve_tape's
    node_snap (captured by the *parent* the instant this node was first
    discovered, reused as-is), this snapshot is freshly captured here,
    right after the replay that got `sim` to this node's state in the
    first place — the parent that discovered this node ran in a different
    process, possibly on a different worker, so there is nothing to hand
    down. Falls back to a fresh reset_and_replay() before every candidate,
    same as solve_tape.solve_tape's own fallback, if node_snap comes back
    None (flag_reset pending or a live atom overlap — see
    search_snapshot()'s docstring in search_shim.c).

    Every output-dropping candidate is verified here individually (not
    just the raw-score best of the node) — narrowing to one candidate
    before verification would silently discard a worse-scoring-but-sound
    candidate in favor of a better-scoring-but-unsound one that later
    fails verification, losing it for good (confirmed against a real
    P202 search: the sound tape was reachable but never even attempted,
    because a different candidate at the same node had a better raw
    score and won the old pre-verification narrowing)."""
    sim.reset_and_replay(path, initial_rotation=rotation)
    node_key = sim.canonical_key()
    node_snap = sim.snapshot()

    candidates: List[Edge] = []
    for arm in range(sim.number_of_arms()):
        for inst in _candidate_instructions(sim, arm, path, node_key):
            candidates.append((arm, inst))

    results = []
    best_local = None  # (score, new_path)
    for arm_index, inst in candidates:
        if node_snap is not None:
            snap_for_step = node_snap
        else:
            sim.reset_and_replay(path, initial_rotation=rotation)
            snap_for_step = None
        result = sim.try_step(snap_for_step, arm_index, inst)
        if result.collided:
            continue
        new_path = path + [(arm_index, inst)]

        if result.output_dropped:
            output_drop_cycle = len(new_path)
            tape_length = output_drop_cycle + reset_overhead(new_path, track_loop_length)
            score = (products_needed - 1) * tape_length + output_drop_cycle
            if (best_local is None or score < best_local[0]) and _verify_full_solution(
                    sim, template_path, new_path, rotation, products_needed, tape_length) is not None:
                best_local = (score, new_path)

        results.append((new_path, result.key, result.output_dropped))

    if node_snap is not None:
        sim.free_snapshot(node_snap)
    return results, best_local


def _expand_batch(items: List[Tuple[int, List[Edge]]]):
    """Pool entrypoint — expands several (rotation, path) nodes in one
    task, amortizing per-task overhead (pickling, executor bookkeeping,
    the future round trip) across `len(items)` nodes instead of one. Reads
    the worker-process globals _init_worker set. Returns one
    (rotation, results, best_local) tuple per input node, in order."""
    return [(rotation,) + _expand_node(_W_SIM, rotation, path, _W_PRODUCTS_NEEDED,
                                        _W_TRACK_LOOP_LENGTH, _W_TEMPLATE_PATH)
            for rotation, path in items]


def solve_tape(puzzle_path: str, template_path: str, products_needed: int, max_cycles: int = 100,
               progress: bool = True, workers: Optional[int] = None, batch_size: int = 512
               ) -> Tuple[int, List[Edge], int, int]:
    """Drop-in parallel equivalent of solve_tape.solve_tape — same
    signature (plus `workers`/`batch_size`), same return value
    (initial_rotation, path, nodes_expanded, track_loop_length), same
    scoring/pruning semantics (see this module's own docstring for how the
    level-synchronous structure here preserves solve_tape.solve_tape's
    early-termination bound exactly). `workers<=1` (the resolved value,
    when `workers` isn't given: os.process_cpu_count()) just calls
    solve_tape.solve_tape directly.

    `batch_size` defaults much larger than schematic_parallel.py's (8): a
    level here can reach tens of thousands of nodes (a real P012 run hit
    58,661 at depth 16), and per-node cost is small and uniform — chunking
    a level that size into small pieces means thousands of separate
    submit()/pickle/result() round trips per level, which measured out to
    erasing the entire multi-core win (confirmed empirically: batch_size=8
    gave ~0 speedup from 1 to 20 workers on a real P012 run)."""
    resolved = workers if workers is not None else (os.process_cpu_count() or 1)
    if resolved <= 1:
        return _serial_solve_tape(puzzle_path, template_path, products_needed,
                                   max_cycles=max_cycles, progress=progress)

    # Bootstrap only: single-arm/non-piston assertions, track_loop_length,
    # and the multi-source root rotations (see solve_tape.solve_tape's own
    # docstring for why every starting facing is a free root, not a real
    # action) — closed again immediately after, since node expansion
    # (including new-best verification, per _expand_node's own docstring)
    # happens entirely inside the worker pool from here on.
    sim = SimState(puzzle_path, template_path, max_cycles=max_cycles)
    n_arms = sim.number_of_arms()
    assert n_arms == 1, (
        f"{template_path} has {n_arms} arms — this search only supports single-arm "
        "layouts (it only ever moves arm 0: reset_and_replay's initial_rotation)")
    assert not ({"w", "s"} & set(sim.legal_instructions(0))), (
        f"{template_path}'s arm is a piston — reset_overhead() doesn't model piston "
        "extension (decode.c's 'X' reset restores it too), so predicted loop lengths "
        "would be wrong for a piston arm")
    track_loop_length = sim.track_loop_length(0)

    visited = set()  # canonical_key
    level: List[Tuple[int, List[Edge]]] = []  # (rotation, path) pairs at the current depth
    for rotation in range(6):
        sim.reset_and_replay([], initial_rotation=rotation)
        key = sim.canonical_key()
        if key not in visited:
            visited.add(key)
            level.append((rotation, []))
    sim.close()

    nodes_expanded = 0
    depth = 0
    best = None  # (rotation, path)
    best_score = None

    with ProcessPoolExecutor(max_workers=resolved, initializer=_init_worker,
                              initargs=(puzzle_path, template_path, max_cycles,
                                        products_needed, track_loop_length)) as pool, \
         tqdm(desc="BFS", unit=" nodes", disable=not progress) as bar:
        while level:
            if depth >= max_cycles:
                break
            # Same bound as solve_tape.solve_tape's own: the best possible
            # score at depth d is products_needed*d (output_drop_cycle=d,
            # reset_overhead=0) — evaluated once per level, using
            # best_score as of everything found through the *previous*
            # level (this level hasn't been expanded yet), exactly
            # mirroring the per-dequeue check there.
            if best_score is not None and products_needed * (depth + 1) >= best_score:
                break
            best_display = best_score if best_score is not None else f">={products_needed * (depth + 1)}"
            bar.set_postfix(depth=depth, level_size=len(level), visited=len(visited),
                             best=best_display, refresh=False)

            batches = [level[i:i + batch_size] for i in range(0, len(level), batch_size)]
            futures = [pool.submit(_expand_batch, batch) for batch in batches]

            next_level: List[Tuple[int, List[Edge]]] = []
            for fut in futures:
                for rotation, results, best_local in fut.result():
                    nodes_expanded += 1
                    bar.update(1)
                    if best_local is not None:
                        score, new_path = best_local
                        if best_score is None or score < best_score:
                            best_score = score
                            best = (rotation, new_path)
                    for new_path, key, _output_dropped in results:
                        if key in visited:
                            continue
                        visited.add(key)
                        next_level.append((rotation, new_path))

            level = next_level
            depth += 1

    sim.close()
    if best is None:
        raise RuntimeError(f"no solution found within max_cycles={max_cycles} "
                            f"(nodes_expanded={nodes_expanded})")
    return (*best, nodes_expanded, track_loop_length)
