#!/usr/bin/env python3
"""
solve_cost.py — compute cost_lower_bound for a puzzle and cross-check its
reasoning against a hand-built reference solution in templates/.

cost_lower_bound returns, alongside the gold figure, a human-readable
`cost_note` string listing exactly which parts it charged for (e.g.
"1×arm=20g + 1×bonder=10g + 1×calcification=10g"). This script parses a
template .solution file's own part list (via a minimal reimplementation of
omsim/parse.c's parse_solution_byte_string — just enough to enumerate part
names, no game simulation) and checks that the two sets match exactly:
every part the note charges for is actually used in the template, and the
template doesn't use any chargeable part the note stayed silent on. A
mismatch means the bound's reasoning and the reference build have drifted
apart — either the bound is wrong, or the template isn't the minimal build
it's meant to be.

It then drives solve_tape.py (a helper module, not meant to be run
directly) to BFS-search for an actual instruction tape reaching the
puzzle's first output, write it into the template in place, and cross-check
it via omsim's own CLI.

Usage:
  python solve_cost.py --puzzle P007
"""

import argparse
import glob
import os
import struct

from bounds import cost_lower_bound
from solve_analysis import find_puzzle_dir, load_puzzle, parse_solutions, verify_with_omsim
from stoichiometry import solve_recipe, solve_recipe_combined
import solve_tape
import solve_tape_parallel

TEMPLATES_DIR = os.path.join(os.path.dirname(__file__), "templates")

# cost_lower_bound's note names each charged glyph by a short token
# ("bonder", "calcification", ...); this maps each to the real in-game
# part name a .solution file records (mirrors the equivalent table in the
# gap-audit scratch tooling built earlier this session).
NOTE_TOKEN_TO_PART_NAME = {
    "bonder-prisma": "bonder-prisma",
    "bonder": "bonder",
    "unbonder": "unbonder",
    "calcification": "glyph-calcification",
    "baron": "baron",
    "duplication": "glyph-duplication",
    "projection": "glyph-projection",
    "purification": "glyph-purification",
    "rejection": "glyph-rejection",
    "division": "glyph-division",
    "unification": "glyph-unification",
    "dispersion": "glyph-dispersion",
    "proliferation": "glyph-proliferation",
    "ravari": "ravari",
    "animismus": "glyph-life-and-death",
}
# Any of these satisfies a note's "arm" charge — cost_lower_bound doesn't
# distinguish which arm variant, just that one (or two, for isolated
# production) exists.
ARM_PART_NAMES = {"arm1", "arm2", "arm3", "arm6"}
# Structurally required regardless of what the bound charges for — not
# part of cost_lower_bound's reasoning at all, so never flagged either way.
STRUCTURAL_PART_NAMES = {"input", "out-std", "out-rep", "pipe"}


def _read_string(data: bytes, pos: int):
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


def parse_solution_part_names(path: str) -> list:
    """Just the part names used in a .solution file — a minimal parallel
    to omsim/parse.c's parse_solution_byte_string, skipping over
    instruction/track-hex/conduit-hex payloads without interpreting them."""
    data = open(path, "rb").read()
    p = 0
    version, = struct.unpack_from("<I", data, p); p += 4
    if version != 7:
        raise ValueError(f"unexpected solution file version {version} in {path}")
    _puzzle, p = _read_string(data, p)
    _name, p = _read_string(data, p)
    solved, = struct.unpack_from("<I", data, p); p += 4
    if solved:
        p += 4 * 8  # zero, cycles, one, cost, two, area, three, instructions

    number_of_parts, = struct.unpack_from("<I", data, p); p += 4
    names = []
    for _ in range(number_of_parts):
        name, p = _read_string(data, p)
        names.append(name)
        p += 1  # presence-flag byte (always 1)
        p += 4 + 4  # position[0], position[1]
        p += 4  # size
        p += 4  # rotation
        p += 4  # which_input_or_output
        n_instr, = struct.unpack_from("<I", data, p); p += 4
        p += n_instr * 5  # each instruction: int32 index + 1 byte
        if name == "track":
            n_hexes, = struct.unpack_from("<I", data, p); p += 4
            p += n_hexes * 8  # each hex: int32, int32
        p += 4  # arm_number
        if name == "pipe":
            p += 4  # conduit_id
            n_hexes, = struct.unpack_from("<I", data, p); p += 4
            p += n_hexes * 8
    return names


def new_template_path(pid: str) -> str:
    """Where a template derived from a fallback source (e.g. a fetched GC
    leaderboard file) gets written under TEMPLATES_DIR, so it becomes a
    real, reusable template from then on."""
    return os.path.join(TEMPLATES_DIR, f"{pid}_CG.solution")


def find_template(pid: str, folder: str) -> str:
    """templates/ is named by puzzle slug (e.g. "stabilized-water-6.solution"),
    the same convention used elsewhere for personal-solution files — not by
    pid, so match against the puzzle folder's own display-name suffix.
    Also checks {pid}_CG.solution, the naming new_template_path uses for
    templates bootstrapped from a fallback source."""
    slug = os.path.basename(folder).split("-", 1)[1].lower()
    matches = sorted(glob.glob(os.path.join(TEMPLATES_DIR, f"{slug}-*.solution")))
    if matches:
        return matches[0]
    pid_named = new_template_path(pid)
    if os.path.isfile(pid_named):
        return pid_named
    raise FileNotFoundError(f"no template for '{pid}' ({slug}) under {TEMPLATES_DIR}")


def expected_parts_from_note(note: str) -> set:
    expected = {part for token, part in NOTE_TOKEN_TO_PART_NAME.items() if token in note}
    expected |= ARM_PART_NAMES  # every note charges >=1 arm unconditionally
    if "track" in note:
        expected.add("track")
    return expected


def check_template_against_note(template_path: str, note: str):
    actual_parts = set(parse_solution_part_names(template_path)) - STRUCTURAL_PART_NAMES
    expected_parts = expected_parts_from_note(note)

    # "arm" is a group (arm1/arm2/arm3/arm6) — satisfied if any one shows
    # up, so it's checked separately rather than via plain set difference.
    has_any_arm = bool(actual_parts & ARM_PART_NAMES)
    missing = (expected_parts - ARM_PART_NAMES) - actual_parts
    if not has_any_arm:
        missing = missing | {"arm"}
    extra = actual_parts - expected_parts

    return expected_parts, actual_parts, missing, extra


def main(args):
    folder, _collection = find_puzzle_dir(args.puzzle)
    pf, puzzle_path = load_puzzle(folder, args.puzzle)
    try:
        recipe = solve_recipe_combined(pf)
    except NotImplementedError:
        recipe = solve_recipe(pf)

    g_lo, note = cost_lower_bound(pf, recipe)
    print(f"{args.puzzle} {pf.name}: cost lower bound = {g_lo}g")
    print(f"  note: {note}")

    scores, files = parse_solutions(folder, args.puzzle)
    if "GC" in scores:
        print(f"  GC record: {scores['GC']['c']} cycles")

    write_path = None
    try:
        template_path = find_template(args.puzzle, folder)
    except FileNotFoundError:
        if "GC" not in files:
            raise
        template_path = os.path.join(folder, files["GC"])
        write_path = new_template_path(args.puzzle)
        print(f"  no template found — bootstrapping one from the local GC record "
              f"({os.path.relpath(template_path)})")

    if write_path is None:
        write_path = template_path
        expected_parts, actual_parts, missing, extra = check_template_against_note(template_path, note)
        if missing:
            print(f"  MISSING from template (note charges for these, template doesn't use them): {sorted(missing)}")
        if extra:
            print(f"  EXTRA in template (used, but note never charged for them): {sorted(extra)}")

    products_needed = pf.products_needed()
    print(f"\nSearching for an instruction tape ({template_path})")
    initial_rotation, path, nodes_expanded, track_loop_length = solve_tape_parallel.solve_tape(
        puzzle_path, template_path, products_needed, max_cycles=args.max_cycles,
        workers=args.workers, batch_size=args.batch_size)

    output_drop_cycle = len(path)
    tape_length = output_drop_cycle + solve_tape.reset_overhead(path, track_loop_length)
    expected_cycles = (products_needed - 1) * tape_length + output_drop_cycle
    print(f"Found tape: {expected_cycles} cycles for {products_needed} outputs, "
          f"{tape_length} cycles/loop, {output_drop_cycle} cycles to first output.")

    solve_tape.write_solution_with_tape(template_path, write_path, path, rotation=initial_rotation)

    metrics = verify_with_omsim(puzzle_path, write_path)
    assert solve_tape.verify_product_cycles(puzzle_path, write_path, n=1) == output_drop_cycle
    assert metrics["cycles"] == expected_cycles

    print(f"Wrote {os.path.relpath(write_path)}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Compute cost_lower_bound and cross-check it against a template solution's actual parts.")
    parser.add_argument("--puzzle", default="P202", help="Puzzle ID")
    parser.add_argument("--max-cycles", type=int, default=16384,
                         help="BFS depth cap for the instruction-tape search (default: 100)")
    parser.add_argument("--workers", type=int, default=os.process_cpu_count(),
                         help="Worker process count for the instruction-tape BFS "
                              "(default: cpu core count usable by this process; 1 disables the pool)")
    parser.add_argument("--batch-size", type=int, default=512,
                         help="Nodes per worker task in the parallel BFS (default: 512)")
    args = parser.parse_args()
    main(args)
