"""
solve_tape.py — helper module: BFS search for the shortest arm-instruction
tape that gets a puzzle from its bare-layout starting state (parts placed,
no instructions) to the first successful output drop.

Nodes are physical simulation states (occupied grid cells + arm mechanism
state, via sim_bridge.SimState.canonical_key — deliberately excludes cycle
number, see search_shim.c). Edges are single-arm actions: one arm performs
one runtime instruction (grab/release/rotate/pivot/piston/track) while every
other arm idles that cycle. Every edge costs one cycle, so BFS finds the
shortest tape reaching the drop.

No idle edge, and no candidate list at all beyond legal_instructions()
minus _candidate_instructions()'s own filters (see its docstring: grab
while already grabbing or with nothing under the grabber, release/pivot
while not grabbing, and undoing the immediately preceding rotate/pivot/
track move while not grabbing are all always-wasteful no-ops or
dominated moves). Every known-optimal tape checked against so far (the
real GC-leaderboard solutions for P007/P009/P010/P011/P013) never idles —
every cycle carries a real, useful instruction — so omitting idle
entirely is an empirical simplification scoped to what's been verified,
not a proven-general rule for every puzzle this might ever run against.

The tape doesn't need to manually return the arm to its starting position
for looping to work: the on-disk solution format has a dedicated 'X'
(reset) instruction — decode.c:1158-1245 — that, at decode time, scans
back over the tape since the last reset and auto-appends the shortest
counter-sequence (release if still grabbing, then rotate back by the net
rotation reduced to [-3,3]) to return the arm to a clean state, extending
the tape's own tape_length/tape_period to cover it (decode.c:1290-1291).
write_solution_with_tape appends a single trailing 'X' rather than having
the search hunt for an explicit return path — cheaper and, since decode.c
always picks the shorter rotation direction, at least as short as anything
a search over raw rotate instructions could find.

The search is multi-source: an arm's starting facing is a free choice of
placement, not a real action, so all 6 possible starting rotations are
roots at depth 0, sharing one visited set.

Reaching a dequeued node from root is not a replay at all: every frontier
entry carries its own snapshot, captured the instant it's discovered (see
solve_tape()'s own docstring), so a dequeue restores in O(atoms on the
board) rather than O(depth). search_snapshot()'s representation is sparse
(only occupied cells, not the dense 64x64 array) specifically so this is
cheap enough to keep one alive per not-yet-expanded frontier entry — see
sim_bridge.py's and search_shim.c's module docstrings for exactly what it
captures/restores.

Scope: single-arm, non-piston layouts only (asserted at the top of
solve_tape()) — verified against P007/P009/P010/P011/P013, all matching
their known GC leaderboard cycle counts exactly.

No CLI of its own — solve_cost.py is the entry point.
"""

import os
import struct
import subprocess
import tempfile
from collections import deque
from typing import List, Optional, Tuple

from tqdm import tqdm

from solve_analysis import OMSIM_BIN
from sim_bridge import SimState, Edge

# Inverse of omsim/decode.c:8-28's decode_instruction table — translates a
# runtime tape letter (what SimState/search_shim deal in) back to the
# on-disk solution-file letter a real .solution part's instruction list
# uses. 'X' (reset) has no runtime-letter equivalent — it's an on-disk-only
# instruction interpreted entirely at decode time (see module docstring).
_RUNTIME_TO_ON_DISK = {
    "d": "R", "a": "r", "w": "E", "s": "e", "f": "G",
    "r": "g", "e": "P", "q": "p", "g": "A", "t": "a", "b": "B",
}

_ARM_PART_NAMES = {"arm1", "arm2", "arm3", "arm6"}


def _read_string(data: bytes, pos: int) -> Tuple[str, int]:
    length = 0
    shift = 0
    while True:
        b = data[pos]
        pos += 1
        length |= (b & 0x7F) << shift
        shift += 7
        if not (b & 0x80):
            break
    return data[pos:pos + length].decode("utf-8", errors="replace"), pos + length


def reset_overhead(path: List[Edge], track_loop_length: int = 0) -> int:
    """Mirrors decode.c's 'X' (reset) instruction logic (decode.c:1158-1224)
    for a plain non-piston arm, so the resulting loop period can be
    predicted without asking omsim: how many extra instructions a trailing
    'X' will auto-append after `path` — a release if the arm is still
    grabbing at the end, plus abs(net rotation, reduced to [-3,3])
    rotate-backs, plus however many track moves it takes to undo the net
    track displacement. Pistons aren't modeled (solve_tape() asserts none
    are present) — decode.c's reset also restores piston extension, which
    this doesn't account for.

    `track_loop_length` (SimState.track_loop_length(arm_index), 0 if the
    arm isn't on a track or its track doesn't loop) lets the track term
    replicate decode.c's actual shortcut search (decode.c:1194-1224):
    on a loop, continuing forward the rest of the way around can be
    shorter than reversing the full net displacement, so the true cost is
    min(net steps one way, loop_length - net steps) — plain modular
    arithmetic once the loop's length is known. Without a known loop
    length (track_loop_length=0, including all non-looping tracks) this
    falls back to the straight-line case, abs(net displacement), which is
    exact for a non-looping track. One remaining gap: decode.c also
    tie-breaks based on how many times the *original* path itself crossed
    back over the start position mid-tape (decode.c:1183-1187,
    track_looping_steps) — a tape that loops around the track more than
    once before resetting — which this doesn't replicate."""
    rotation = 0
    grabbing = False
    track_steps = 0
    for arm_index, inst in path:
        if arm_index is None:
            continue
        if inst == "a":
            rotation += 1
        elif inst == "d":
            rotation -= 1
        elif inst == "f":
            grabbing = True
        elif inst == "r":
            grabbing = False
        elif inst == "g":
            track_steps += 1
        elif inst == "t":
            track_steps -= 1
    while rotation > 3:
        rotation -= 6
    while rotation < -3:
        rotation += 6
    if track_loop_length and track_steps:
        net = abs(track_steps) % track_loop_length
        track_overhead = min(net, track_loop_length - net)
    else:
        track_overhead = abs(track_steps)
    return (1 if grabbing else 0) + abs(rotation) + track_overhead


# Inverse direction for each pair of moves that can directly undo one
# another: 'a'/'d' (rotate CCW/CW), 'q'/'e' (pivot CCW/CW), 'g'/'t' (track
# +/- direction). None of the three pairs cancel across pairs (each touches
# a different mechanism field — direction_u/v/arm_rotation, pivot_parity,
# position respectively).
_OPPOSITE = {"a": "d", "d": "a", "q": "e", "e": "q", "g": "t", "t": "g"}


def _currently_grabbed(path: List[Edge]) -> bool:
    for arm_index, inst in reversed(path):
        if arm_index is None:
            continue
        if inst == "f":
            return True
        if inst == "r":
            return False
    return False


def _last_real_instruction(path: List[Edge]) -> Optional[str]:
    for arm_index, inst in reversed(path):
        if arm_index is not None:
            return inst
    return None


# canonical_key()'s own byte layout (search_canonical_key in
# search_shim.c): n occupied-cell records (position.u:i32, position.v:i32,
# atom:u64 — 16 bytes each, sorted by position) followed by one
# arm-mechanism record per arm (position.u/v, direction_u.u/v,
# direction_v.u/v, arm_rotation, type — eight i32/u32 fields, 32 bytes;
# pivot_parity excluded, see search_canonical_key's own comment), all
# little-endian. n isn't stored explicitly, but since solve_tape() is
# single-arm-only (asserted at the top) the record count is recoverable
# from the buffer's total length alone.
_ARM_RECORD_SIZE = 32
_ATOM_RECORD_SIZE = 16
_VALID = 0x1
_REMOVED = 1 << 27


def _grabbable_atom_present(key: bytes) -> bool:
    """Whether arm 0's grabber (one grabber, at the arm's own facing
    direction — mechanism_relative_position(m, 1, 0, 1) reduces to
    position + direction_u for a plain, non-multi-grabber arm; see sim.c's
    u_offset_for_direction(0)/mechanism_relative_position) currently sits
    on a valid, non-removed atom — decoded directly from a canonical_key()
    already computed for this node (returned by try_step()/threaded
    through the frontier) rather than a second round-trip through ctypes
    for state we've already paid to compute once per node."""
    arm_offset = len(key) - _ARM_RECORD_SIZE
    n_atoms = arm_offset // _ATOM_RECORD_SIZE
    pos_u, pos_v, dir_u_u, dir_u_v = struct.unpack_from("<iiii", key, arm_offset)
    grab_u, grab_v = pos_u + dir_u_u, pos_v + dir_u_v
    for i in range(n_atoms):
        u, v, atom = struct.unpack_from("<iiQ", key, i * _ATOM_RECORD_SIZE)
        if u == grab_u and v == grab_v:
            return bool(atom & _VALID) and not (atom & _REMOVED)
    return False


def _candidate_instructions(sim: SimState, arm: int, path: List[Edge], key: bytes) -> List[str]:
    """legal_instructions(arm), minus moves that are always wasteful given
    what's already happened on this path:
      - grab while already grabbing (sim.c: a no-op)
      - grab while nothing valid, non-removed sits under the grabber (sim.c:
        a no-op — the 'f' case in perform_arm_instructions only sets
        GRABBING_LOW_BIT and increments the grab count when get_atom()
        finds something there; see _grabbable_atom_present())
      - release while not grabbing (sim.c: a no-op)
      - pivot while not grabbing: pivot only moves a grabbed atom around a
        different grabber (multi-arm mechanisms); with nothing grabbed it
        only flips pivot_parity, no physical effect
      - the exact opposite of the immediately preceding rotate/pivot/track
        move, but ONLY while nothing is grabbed. With nothing held, a bare
        rotate/track step truly has no side effect to undo — reversing it
        right back is always strictly worse than never having moved. But
        with something grabbed, each step carries it through a real
        intermediate position (possibly near a glyph/bonder that reacts on
        contact), so "undo the last step" is not the same as "as if it
        never happened" — e.g. rotate CW x4 then CCW x2 (net CW x2) is a
        real, sometimes-necessary sequence this must not prune just
        because its 5th and 6th instructions are exact opposites. Confirmed
        against a real P009 leaderboard solution that does exactly this,
        which the search couldn't find before this was scoped to
        not-grabbed.
    """
    grabbed = _currently_grabbed(path)
    blocked = _OPPOSITE.get(_last_real_instruction(path)) if not grabbed else None
    out = []
    for inst in sim.legal_instructions(arm):
        if inst == "f" and grabbed:
            continue
        if inst == "f" and not grabbed and not _grabbable_atom_present(key):
            continue
        if inst == "r" and not grabbed:
            continue
        if inst in ("q", "e") and not grabbed:
            continue
        if inst == blocked:
            continue
        out.append(inst)
    return out


def solve_tape(puzzle_path: str, template_path: str, products_needed: int, max_cycles: int = 100,
               progress: bool = True) -> Tuple[int, List[Edge], int, int]:
    """Multi-source BFS for the instruction tape — among all tapes reaching
    the first output drop (searching all 6 possible starting rotations for
    arm 0 at once, each a root at depth 0 — see module docstring) — that
    minimizes the expected cycle count for the puzzle's full
    `products_needed` repetitions: (products_needed-1)*tape_length +
    output_drop_cycle, where tape_length = output_drop_cycle +
    reset_overhead(path) (the auto-generated trailing 'X' — see module
    docstring). This isn't just the shortest path to the drop: a path that
    takes one cycle longer but leaves the arm facing its starting rotation
    with nothing grabbed (reset_overhead=0) can beat a shorter path that
    ends mid-rotation or still grabbing (reset_overhead up to 4).

    BFS doesn't stop at the first goal found — it keeps searching, tracking
    the best-scoring goal seen, and prunes once no goal at a deeper level
    could possibly beat it (the best possible score at depth d is
    products_needed*d, i.e. output_drop_cycle=d with reset_overhead=0).

    `visited` is hashes only (canonical_key bytes — a permanent dedup
    record every discovered state needs forever, cheap regardless of how
    it's stored). Each *frontier* entry, though, carries its own snapshot
    alongside its path — captured once, the instant a state is discovered
    (right after the try_step() that produced it, so the live state is
    already exactly there — no extra call needed to get it), and reused to
    restore directly to that state when the entry is later dequeued.
    That's what actually eliminates the O(depth) reset_and_replay() this
    function used to pay on every single dequeue: a snapshot lets restore
    happen in O(atoms actually on the board) instead. A snapshot is freed
    the moment its node has been dequeued and all of its own candidates
    tried — after that nothing needs it again, so peak memory is bounded
    by frontier size (which is what search_snapshot()'s sparse
    representation — only occupied cells, not the dense 64x64 array — is
    for: cheap enough per-entry that keeping one alive per not-yet-
    expanded frontier entry is practical, unlike a dense snapshot would
    have been)."""
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
    # each entry: (initial_rotation, path from that root, path's own
    # canonical_key, a snapshot of that exact state — None only in the
    # rare case snapshot() itself refused (see its docstring), in which
    # case dequeuing this entry falls back to a full replay)
    frontier = deque()
    for rotation in range(6):
        sim.reset_and_replay([], initial_rotation=rotation)
        key = sim.canonical_key()
        if key not in visited:
            visited.add(key)
            frontier.append((rotation, [], key, sim.snapshot()))

    nodes_expanded = 0
    current_depth = -1
    best = None  # (rotation, path)
    best_score = None

    with tqdm(desc="BFS", unit=" nodes", disable=not progress) as bar:
        while frontier:
            rotation, path, node_key, node_snap = frontier.popleft()
            if len(path) != current_depth:
                current_depth = len(path)
                if best_score is not None and products_needed * (current_depth + 1) >= best_score:
                    if node_snap is not None:
                        sim.free_snapshot(node_snap)
                    break
                best_display = best_score if best_score is not None else f">={products_needed * (current_depth + 1)}"
                bar.set_postfix(depth=current_depth, frontier=len(frontier) + 1,
                                 visited=len(visited), best=best_display, refresh=False)

            if len(path) >= max_cycles:
                if node_snap is not None:
                    sim.free_snapshot(node_snap)
                continue

            nodes_expanded += 1
            bar.update(1)

            candidates: List[Edge] = []
            for arm in range(n_arms):
                for inst in _candidate_instructions(sim, arm, path, node_key):
                    candidates.append((arm, inst))

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
                    if (best_score is None or score < best_score) and _verify_full_solution(
                            sim, template_path, new_path, rotation, products_needed, tape_length) is not None:
                        best = (rotation, new_path)
                        best_score = score

                if result.key in visited:
                    continue
                visited.add(result.key)
                # Live state is exactly this new child's state right now —
                # try_step() just produced it and nothing since has
                # touched it — so this snapshot is free to capture here,
                # no extra restore needed first.
                frontier.append((rotation, new_path, result.key, sim.snapshot()))

            if node_snap is not None:
                sim.free_snapshot(node_snap)

    # Snapshots are owned by their frontier entry, not by `sim` — pruning
    # can `break` out above with entries still pending, each holding a
    # snapshot sim.close() below has no knowledge of and won't free.
    for _rotation, _path, _key, leftover_snap in frontier:
        if leftover_snap is not None:
            sim.free_snapshot(leftover_snap)

    sim.close()
    if best is None:
        raise RuntimeError(f"no solution found within max_cycles={max_cycles} "
                            f"(nodes_expanded={nodes_expanded})")
    return (*best, nodes_expanded, track_loop_length)


def _build_solution_bytes(template_path: str, path: List[Edge],
                           rotation: Optional[int] = None) -> bytes:
    """Copy template_path, inserting `path`'s non-idle instructions into the
    file's one arm part's (empty) instruction list, optionally also setting
    that arm's initial `rotation` field to `rotation` (0-5, absolute).
    Scoped to single-arm puzzles: raises if the template doesn't have
    exactly one arm part."""
    data = bytearray(open(template_path, "rb").read())
    p = 0
    version, = struct.unpack_from("<I", data, p); p += 4
    if version != 7:
        raise ValueError(f"unexpected solution file version {version} in {template_path}")
    _puzzle, p = _read_string(data, p)
    _name, p = _read_string(data, p)
    solved, = struct.unpack_from("<I", data, p); p += 4
    if solved:
        p += 4 * 8

    number_of_parts, = struct.unpack_from("<I", data, p); p += 4
    arm_ninstr_offset = None
    arm_old_n_instr = None
    arm_rotation_offset = None
    for _ in range(number_of_parts):
        name, p = _read_string(data, p)
        p += 1  # presence flag
        p += 4 + 4  # position
        p += 4  # size
        rotation_offset = p
        p += 4  # rotation
        p += 4  # which_input_or_output
        n_instr_offset = p
        n_instr, = struct.unpack_from("<I", data, p); p += 4
        p += n_instr * 5
        if name == "track":
            n_hexes, = struct.unpack_from("<I", data, p); p += 4
            p += n_hexes * 8
        p += 4  # arm_number
        if name == "pipe":
            p += 4
            n_hexes, = struct.unpack_from("<I", data, p); p += 4
            p += n_hexes * 8
        if name in _ARM_PART_NAMES:
            if arm_ninstr_offset is not None:
                raise ValueError(f"{template_path} has more than one arm part — "
                                  "write_solution_with_tape only supports single-arm templates")
            arm_ninstr_offset = n_instr_offset
            arm_old_n_instr = n_instr
            arm_rotation_offset = rotation_offset

    if arm_ninstr_offset is None:
        raise ValueError(f"{template_path} has no arm part")

    if rotation is not None:
        struct.pack_into("<i", data, arm_rotation_offset, rotation % 6)

    # Replace whatever instructions the arm already had (e.g. from a
    # previous solve_tape.py run that wrote its result back into this same
    # template) rather than requiring an empty tape — the search always
    # starts from a blank tape regardless (search_shim.c's search_create
    # unconditionally discards any decoded instructions), so on-disk
    # leftovers here are stale by construction and safe to overwrite.
    instructions = [(i, inst) for i, (arm_index, inst) in enumerate(path) if arm_index is not None]
    # A trailing 'X' (reset) auto-generates the shortest return-to-start
    # sequence at decode time (decode.c:1158-1245, see module docstring)
    # and extends the file's own tape_length to cover it — no need to
    # search for or manually write those instructions.
    instructions.append((len(path), "X"))
    new_instr_bytes = b"".join(
        struct.pack("<i", cycle) + _RUNTIME_TO_ON_DISK.get(inst, inst).encode("ascii")
        for cycle, inst in instructions
    )
    old_instr_start = arm_ninstr_offset + 4
    old_instr_end = old_instr_start + arm_old_n_instr * 5
    return (bytes(data[:arm_ninstr_offset]) + struct.pack("<I", len(instructions))
            + new_instr_bytes + bytes(data[old_instr_end:]))


def write_solution_with_tape(template_path: str, out_path: str, path: List[Edge],
                              rotation: Optional[int] = None) -> None:
    """Disk-writing wrapper around _build_solution_bytes — see its own
    docstring for what gets built."""
    with open(out_path, "wb") as f:
        f.write(_build_solution_bytes(template_path, path, rotation))


def _verify_full_solution(sim: SimState, template_path: str, path: List[Edge], rotation: int,
                           products_needed: int, tape_length: int) -> Optional[int]:
    """Confirms the auto-generated 'X' reset doesn't collide (see
    SimState.verify_full_solution) before a candidate is accepted."""
    solution_bytes = _build_solution_bytes(template_path, path, rotation)
    cycle_limit = tape_length * products_needed + 1000
    return sim.verify_full_solution(solution_bytes, products_needed, cycle_limit)


def verify_product_cycles(puzzle_path: str, solution_path: str, n: int = 1) -> int:
    """Independently confirms `solution_path` actually reaches its Nth
    output via omsim's own "product N cycles" query (verifier.c's
    "product " handling — runs the real simulator until N outputs complete,
    or collision/cycle-limit, whichever first). Returns the cycle count
    omsim reports, raising if omsim reports anything else (collision, no
    completion). n=1 is the first-output check; passing pf.products_needed()
    checks the full puzzle-victory cycle count, which for a tape whose arm
    returns to its starting state each loop should equal
    (n-1)*tape_length + output_drop_cycle (each repeat after the first is a
    clean tape_length-cycle loop) — see solve_tape()'s module docstring."""
    if not os.path.isfile(OMSIM_BIN):
        raise FileNotFoundError(f"omsim not found at {OMSIM_BIN}; run `make` in omsim/")
    cmd = [OMSIM_BIN, "-p", puzzle_path, "-m", f"product {n} cycles", solution_path]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(f"omsim verification failed:\n{result.stdout.strip()}\n{result.stderr.strip()}")
    line = result.stdout.strip()
    _key, _, value = line.partition(": ")
    return int(value.strip())
