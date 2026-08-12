"""sim_bridge.py — ctypes bindings over omsim/libsearch.so, giving Python a
Pythonic single-cycle-stepping view of a live omsim simulation (build the
shared library first: `make libsearch.so` in omsim/).

Unlike verify_with_omsim (solve_analysis.py), which only runs a whole
solution start-to-finish for metric reporting, SimState here exposes one
game cycle at a time — the primitive solve_tape.py's BFS is built on.

search_reset() (SimState.reset_and_replay()) always returns to the *one*
fixed just-after-initial_setup() state (via destroy(0, &board) +
initial_setup(), the same free-and-zero pattern this repo's other callers
already use) rather than an arbitrary snapshot, restoring each arm's
mechanism state from a flat, pointer-free struct mechanism copy taken once
at search_create() time, and setting arm 0's starting rotation — this lets
a path be replayed without re-parsing the puzzle/solution files or
re-running decode_solution() on every call.

snapshot()/try_step() give O(atoms on the board) — not O(depth) — restore
to any previously-reached state, letting solve_tape.py's BFS keep a
snapshot alongside every frontier entry instead of replaying from root on
each dequeue. search_snapshot()'s own representation is sparse (only
occupied grid cells, not the dense 64x64 array), which is what makes
holding one per not-yet-expanded frontier entry practical rather than a
memory problem — see omsim/search_shim.c's module docstring and
search_snapshot()/search_restore()'s own comments for exactly which board
fields are captured and why (board->flag_reset holds raw pointers into the
grid's own array and is asserted empty rather than captured; several
scratch lists a *colliding* run() can leave non-empty are explicitly
zeroed on restore rather than copied, since they're always empty at any
point snapshot() could have been called).

The C API exposed here is deliberately narrow: only what genuinely needs
omsim's internal state (grid/arm/board fields Python has no access to) gets
a ctypes entry point, and calls that are always made together on every one
of a BFS's hundreds of thousands of candidate tries (restore, set this
cycle's instruction, step, read back collided/output_dropped/canonical_key)
are collapsed into single combined C functions (search_reset does
reset+rotation; search_try_step does restore+instruction+step+key) rather
than issuing several separate ctypes calls for each — at that call volume,
ctypes' own per-call overhead costs more than the simulation work each
call actually does. Results that never change over a search (arm count,
legal-instruction set) are queried once and cached in Python rather than
re-queried through ctypes on every use.
"""

import ctypes
import os
from typing import Dict, List, NamedTuple, Optional, Tuple

_LIB_PATH = os.path.join(os.path.dirname(__file__), "omsim", "libsearch.so")

try:
    _lib = ctypes.CDLL(_LIB_PATH)
except OSError as e:
    raise FileNotFoundError(
        f"{_LIB_PATH} not found or failed to load — run `make libsearch.so` in omsim/"
    ) from e

_lib.search_create.argtypes = [ctypes.c_char_p, ctypes.c_char_p, ctypes.c_uint32,
                                ctypes.POINTER(ctypes.c_char_p)]
_lib.search_create.restype = ctypes.c_void_p

_lib.search_destroy.argtypes = [ctypes.c_void_p]
_lib.search_destroy.restype = None

_lib.search_reset.argtypes = [ctypes.c_void_p, ctypes.c_uint32, ctypes.c_int32]
_lib.search_reset.restype = None

_lib.search_number_of_arms.argtypes = [ctypes.c_void_p]
_lib.search_number_of_arms.restype = ctypes.c_uint32

_lib.search_legal_instruction_mask.argtypes = [ctypes.c_void_p, ctypes.c_uint32]
_lib.search_legal_instruction_mask.restype = ctypes.c_uint32

_lib.search_track_loop_length.argtypes = [ctypes.c_void_p, ctypes.c_uint32]
_lib.search_track_loop_length.restype = ctypes.c_uint32

_lib.search_replay.argtypes = [ctypes.c_void_p, ctypes.c_char_p, ctypes.c_uint32]
_lib.search_replay.restype = ctypes.c_uint32

_lib.search_snapshot.argtypes = [ctypes.c_void_p]
_lib.search_snapshot.restype = ctypes.c_void_p

_lib.search_snapshot_free.argtypes = [ctypes.c_void_p]
_lib.search_snapshot_free.restype = None

_lib.search_collision_reason.argtypes = [ctypes.c_void_p]
_lib.search_collision_reason.restype = ctypes.c_char_p

_lib.search_canonical_key.argtypes = [ctypes.c_void_p, ctypes.POINTER(ctypes.c_uint8), ctypes.c_size_t]
_lib.search_canonical_key.restype = ctypes.c_size_t

_lib.search_try_step.argtypes = [
    ctypes.c_void_p, ctypes.c_void_p, ctypes.c_uint32, ctypes.c_char,
    ctypes.POINTER(ctypes.c_uint8), ctypes.c_size_t, ctypes.POINTER(ctypes.c_size_t),
]
_lib.search_try_step.restype = ctypes.c_uint32

_lib.search_verify_full.argtypes = [
    ctypes.c_void_p, ctypes.c_char_p, ctypes.c_uint32,
    ctypes.c_uint32, ctypes.c_uint32, ctypes.POINTER(ctypes.c_uint64),
]
_lib.search_verify_full.restype = ctypes.c_int

# Must match search_shim.c's NO_ARM sentinel exactly (arm_index for the
# explicit all-idle edge — every arm idles regardless of instruction).
_NO_ARM = 0xFFFFFFFF

# Runtime instruction letters and the shim's bit position for each — must
# match search_shim.c's BIT_* defines exactly.
_LETTER_BITS = [
    ("f", 1 << 0), ("r", 1 << 1), ("a", 1 << 2), ("d", 1 << 3),
    ("q", 1 << 4), ("e", 1 << 5), ("w", 1 << 6), ("s", 1 << 7),
    ("g", 1 << 8), ("t", 1 << 9),
]

_CANONICAL_KEY_BUF_SIZE = 4096

# One edge in a BFS path: (arm_index, instruction), or (None, ' ') for the
# explicit all-idle edge.
Edge = Tuple[Optional[int], str]


class StepResult(NamedTuple):
    collided: bool
    output_dropped: bool
    key: bytes


class SimState:
    def __init__(self, puzzle_path: str, bare_solution_path: str, max_cycles: int = 200):
        self.puzzle_path = puzzle_path
        self.bare_solution_path = bare_solution_path
        self.max_cycles = max_cycles
        self._handle = None
        self._create()
        # Both static for the whole search (arm count and each arm's legal
        # instruction set — piston-ness and track membership — never change
        # once decoded), so query once here rather than re-querying through
        # ctypes on every node/candidate.
        self._n_arms = _lib.search_number_of_arms(self._handle)
        self._legal_instructions: Dict[int, List[str]] = {}
        # Allocated once and reused for every try_step()/canonical_key()
        # call rather than a fresh ctypes array + c_size_t per call (this
        # runs hundreds of thousands of times per search): profiling
        # (callgrind against a real P012 run) showed ctypes array
        # allocation/zeroing alone was a significant fraction of total
        # instructions, on top of the even larger cost fixed alongside
        # this — see try_step()'s own comment on the bytes() conversion.
        self._key_buf = (ctypes.c_uint8 * _CANONICAL_KEY_BUF_SIZE)()
        self._key_length = ctypes.c_size_t()

    def _create(self) -> None:
        error = ctypes.c_char_p()
        handle = _lib.search_create(
            self.puzzle_path.encode(), self.bare_solution_path.encode(),
            self.max_cycles, ctypes.byref(error))
        if not handle:
            msg = error.value.decode() if error.value else "unknown error"
            raise ValueError(f"search_create failed: {msg}")
        self._handle = handle

    def close(self) -> None:
        if self._handle is not None:
            _lib.search_destroy(self._handle)
            self._handle = None

    def __del__(self):
        self.close()

    def number_of_arms(self) -> int:
        return self._n_arms

    def track_loop_length(self, arm_index: int) -> int:
        """0 if arm_index isn't on a track or its track doesn't loop back
        to its starting hex; otherwise the loop's length in hexes. Static
        for the whole search (track layout never changes) — call once and
        reuse rather than querying per candidate."""
        return _lib.search_track_loop_length(self._handle, arm_index)

    def legal_instructions(self, arm_index: int) -> List[str]:
        cached = self._legal_instructions.get(arm_index)
        if cached is None:
            mask = _lib.search_legal_instruction_mask(self._handle, arm_index)
            cached = [letter for letter, bit in _LETTER_BITS if mask & bit]
            self._legal_instructions[arm_index] = cached
        return cached

    def try_step(self, snapshot, arm_index: Optional[int], inst: str) -> StepResult:
        """Restore to `snapshot` (pass None to skip — when the live state is
        already exactly where this candidate should start from, e.g. the
        first candidate tried right after reset_and_replay()+snapshot()),
        then apply one cycle with `arm_index` performing `inst` (arm_index
        None for the explicit all-idle edge — every other arm idles
        regardless), and read back the resulting canonical key — one ctypes
        call (search_try_step) covering what used to be up to 7 separate
        ones. On collision, the C side skips computing a key at all (never
        read by any caller here — every call site does `if result.collided:
        continue` before ever touching .key) and reports key_length=0 for
        that reason rather than a buffer overflow; key is simply empty
        bytes in that case, not an error.

        Reuses self._key_buf/self._key_length (allocated once in __init__)
        rather than a fresh ctypes array + c_size_t per call, and converts
        via bytes(self._key_buf)[:n] rather than bytes(self._key_buf[:n])
        — slicing a ctypes Array goes through Python's generic sequence-
        slice protocol (an isinstance check + Py_ssize_t conversion per
        *individual byte*), while bytes() on the whole array uses the
        buffer protocol (one memcpy). Confirmed via callgrind against a
        real P012 run: that one-character difference (slice-then-bytes vs
        bytes-then-slice) accounted for roughly 45% of the entire
        process's instruction count — object_recursive_isinstance alone
        was the single hottest function in the whole program, ahead of
        every actual simulation function combined. The result is copied
        out into a real, independent bytes object before this function
        returns, so reusing the same buffer on the next call is safe."""
        idx = _NO_ARM if arm_index is None else arm_index
        flags = _lib.search_try_step(
            self._handle, snapshot, idx, inst.encode(),
            self._key_buf, _CANONICAL_KEY_BUF_SIZE, ctypes.byref(self._key_length))
        collided = bool(flags & 1)
        key_length = self._key_length.value
        if not collided and key_length == 0:
            raise BufferError("canonical key exceeded scratch buffer size")
        return StepResult(
            collided=collided,
            output_dropped=bool(flags & 2),
            key=bytes(self._key_buf)[:key_length])

    def snapshot(self):
        """Captures occupied grid cells + arm state (sparse, not a dense
        grid copy — see search_shim.c) for fast restore later (see
        search_shim.c's module docstring for the flag_reset safety
        argument). Returns an opaque handle, or None if it's currently
        unsafe to snapshot (a pending flag-reset entry is outstanding) —
        callers must have a fallback (e.g. replay) for that case, not
        treat it as any other kind of failure. Caller owns the handle and
        must eventually pass it to free_snapshot()."""
        snap = _lib.search_snapshot(self._handle)
        return snap if snap else None

    def free_snapshot(self, snapshot) -> None:
        _lib.search_snapshot_free(snapshot)

    def canonical_key(self) -> bytes:
        n = _lib.search_canonical_key(self._handle, self._key_buf, _CANONICAL_KEY_BUF_SIZE)
        if n == 0:
            raise BufferError("canonical key exceeded scratch buffer size")
        return bytes(self._key_buf)[:n]

    def reset_and_replay(self, path: List[Edge], initial_rotation: int) -> None:
        """Reset the live simulation back to its just-after-initial_setup()
        state and set arm 0's starting rotation (search_reset() — no
        re-parsing, no re-decoding, one ctypes call for both), then replay
        `path` (root to target, one edge per cycle) via a single
        search_replay() call. Single-arm only: `path`'s idle entries
        already carry ' ' as their instruction, so joining every entry's
        instruction character in order is exactly arm 0's tape — see
        search_replay's docstring for why this collapses what used to be
        ~4 ctypes calls per replayed cycle into one call for the whole
        path."""
        _lib.search_reset(self._handle, 0, initial_rotation)
        if not path:
            return
        tape = "".join(inst for _arm_index, inst in path).encode("ascii")
        executed = _lib.search_replay(self._handle, tape, len(tape))
        if executed != len(tape):
            reason_bytes = _lib.search_collision_reason(self._handle)
            reason = reason_bytes.decode() if reason_bytes else None
            raise RuntimeError(
                f"replay hit an unexpected collision at cycle {executed}: "
                f"{reason!r} — path was already validated when discovered")

    def verify_full_solution(self, solution_bytes: bytes, products_needed: int,
                              cycle_limit: int) -> Optional[int]:
        """Runs `solution_bytes` (including its 'X' reset) in-process via
        search_verify_full, no disk I/O. Returns the cycle count on
        success, None on collision/failure (never raises)."""
        out_cycles = ctypes.c_uint64(0)
        ok = _lib.search_verify_full(
            self._handle, solution_bytes, len(solution_bytes),
            products_needed, cycle_limit, ctypes.byref(out_cycles))
        return out_cycles.value if ok else None
