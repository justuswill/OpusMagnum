#!/usr/bin/env python3
"""
solve.py — compute lower/upper bounds for the metrics of the optimal SUM solution.

Lower bounds come from analytic proofs in bounds.py (throughput, minimum
part cost, molecule footprint).  Upper bounds come from the best SUM record on
the leaderboard (stored locally).  The omsim simulator verifies that the SUM
solution actually achieves the metrics it claims.

Filename format: {pid}_{CAT}_{cost}g-{cycles}c-{area}a-{instructions}i.solution
Production cats have a _P suffix; their SUM metric is g+c+i (not g+c+a).

Usage:
  python solve.py --puzzle P007
  python solve.py --puzzle w1611998067
"""

import argparse
import os
import re
import subprocess

from puzzle_parser import parse_puzzle, ATOM_NAMES
from bounds import cycles_lower_bound, cost_lower_bound, area_lower_bound
from stoichiometry import solve_recipe, describe_molecule

PUZZLES_DIR = os.path.join(os.path.dirname(__file__), "puzzles")
OMSIM_BIN   = os.path.join(os.path.dirname(__file__), "omsim", "omsim")

_FNAME_RE = re.compile(
    r"^(?P<pid>[^_]+)_(?P<cat>[^_]+(?:_P)?)_"
    r"(?P<g>\d+)g-(?P<c>\d+)c-(?P<a>\d+|na)a-(?P<i>\d+)i\.solution$"
)


def find_puzzle_dir(pid):
    for collection in os.listdir(PUZZLES_DIR):
        coll_path = os.path.join(PUZZLES_DIR, collection)
        if not os.path.isdir(coll_path):
            continue
        for entry in os.listdir(coll_path):
            if entry == pid or entry.startswith(pid + "-"):
                return os.path.join(coll_path, entry), collection
    raise FileNotFoundError(f"no local folder for '{pid}' under {PUZZLES_DIR}")


def parse_solutions(folder, pid):
    scores = {}
    files = {}
    for fname in os.listdir(folder):
        m = _FNAME_RE.match(fname)
        if not m or m.group("pid") != pid:
            continue
        cat = m.group("cat")
        scores[cat] = {
            "g": int(m.group("g")),
            "c": int(m.group("c")),
            "a": None if m.group("a") == "na" else int(m.group("a")),
            "i": int(m.group("i")),
        }
        files[cat] = fname
    if not scores:
        raise FileNotFoundError(f"no solution files in {folder}")
    return scores, files


def load_puzzle(folder, pid):
    path = os.path.join(folder, f"{pid}.puzzle")
    if not os.path.isfile(path):
        raise FileNotFoundError(f"puzzle file not found: {path}")
    return parse_puzzle(path), path


def format_bound(lo, hi):
    if lo == hi:
        return f"{lo}  (exact)"
    return f"[{lo}, {hi}]"


def verify_with_omsim(puzzle_file, solution_file):
    if not os.path.isfile(OMSIM_BIN):
        raise FileNotFoundError(f"omsim not found at {OMSIM_BIN}; run `make` in omsim/")
    metrics = ["cost", "cycles", "area", "instructions"]
    cmd = ([OMSIM_BIN, "-p", puzzle_file]
           + [arg for m in metrics for arg in ("-m", m)]
           + [solution_file])
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(f"omsim failed:\n{result.stderr.strip()}")
    out = {}
    for line in result.stdout.splitlines():
        if ": " in line:
            k, v = line.split(": ", 1)
            out[k.strip()] = int(v.strip())
    return out


_OMSIM_KEY_TO_SCORE_KEY = {"cost": "g", "cycles": "c", "area": "a", "instructions": "i"}


def verify_all_solutions(folder, puzzle_file, scores, files):
    """Verify every found solution with omsim, overriding any score fields that
    the filename got wrong with the omsim-computed value. Silent unless a
    mismatch is found and overridden."""
    for cat, fname in files.items():
        actual = verify_with_omsim(puzzle_file, os.path.join(folder, fname))
        for omsim_key, score_key in _OMSIM_KEY_TO_SCORE_KEY.items():
            if omsim_key not in actual:
                continue
            exp = scores[cat][score_key]
            act = actual[omsim_key]
            if exp is not None and act != exp:
                print(f"  {fname}: {score_key}={exp} was wrong, overridden with omsim value {act}")
                scores[cat][score_key] = act


def print_recipe(pf, result):
    print("Recipe (stoichiometry):")
    if result.status != "Optimal":
        print(f"  {result.status} — no atom-balanced recipe found")
        return
    for i, mol in enumerate(pf.inputs):
        count = result.reagent_counts.get(i, 0)
        if count:
            group_size = result.reagent_group_size.get(i, 1)
            suffix = f"[x{group_size}]" if group_size > 1 else ""
            print(f"  {count:>4}x  input {i}{suffix}: {describe_molecule(mol)}")
    products_needed = pf.products_needed()
    for i, mol in enumerate(pf.outputs):
        print(f"  {products_needed:>4}x  output {i}: {describe_molecule(mol)}")
    if result.reaction_counts:
        print("  Transformations:")
        for name, count in sorted(result.reaction_counts.items()):
            print(f"  {count:>4}x  {name}")
    if result.waste:
        print("  Waste (surplus atoms produced beyond what's needed):")
        for atype, count in sorted(result.waste.items()):
            print(f"  {count:>4}x  {ATOM_NAMES.get(atype, atype)}")


def main(args):
    pid = args.puzzle.strip()
    folder, collection = find_puzzle_dir(pid)
    name = os.path.basename(folder)[len(pid):].lstrip("-").replace("-", " ") or pid
    scores, solution_files = parse_solutions(folder, pid)
    pf, puzzle_file = load_puzzle(folder, pid)
    verify_all_solutions(folder, puzzle_file, scores, solution_files)

    is_prod = any(cat.endswith("_P") for cat in scores)
    if is_prod:
        sum_cat = "SUM_P"
        t_label, t_key = "instructions (i)", "i"
        sum_label = "g+c+i"
    else:
        sum_cat = "SUM"
        t_label, t_key = "area (a)", "a"
        sum_label = "g+c+a"
    if sum_cat not in scores:
        raise ValueError(f"no {sum_cat} solution file found in {folder}")

    print(f"Puzzle    : {name} ({pid})")
    print(f"Collection: {collection}")
    print(f"Type      : {'PRODUCTION' if is_prod else 'NORMAL'}")
    print()

    recipe = solve_recipe(pf)
    print_recipe(pf, recipe)
    print()

    # ── Analytic lower bounds ─────────────────────────────────────────────

    g_lo, g_note = cost_lower_bound(pf)
    c_lo, c_note = cycles_lower_bound(pf, recipe)
    if not is_prod:
        t_lo, t_note = area_lower_bound(pf, recipe)
    else:
        t_lo, t_note = 1, "trivial (≥1 instruction)"

    # ── Lower bounds + individual records table ───────────────────────────

    g_cats = ["GX_P", "GX"] if is_prod else ["GX"]
    c_cats = ["CX_P", "CX"] if is_prod else ["CX"]
    t_cats = ["IX_P"] if is_prod else ["AX"]
    cat_order = (g_cats + c_cats + t_cats + ([sum_cat]))

    # Collect the categories that actually have records, preserving order
    present_cats = [c for c in cat_order if c in scores]

    def cell(val, lo, w):
        if val is None:
            return " " * (w - 3) + "n/a"
        mark = "*" if val == lo else " "
        return f"{mark}{val:{w-1}d}"

    col_w = max(5, *(len(c) + 1 for c in present_cats))
    header = f"  {'metric':<18}  {'≥':>5}  " + "  ".join(f"{c:>{col_w}}" for c in present_cats)
    sep    = "  " + "-" * (len(header) - 2)

    def row(label, lo, vals):
        return (f"  {label:<18}  {lo:>5}  " +
                "  ".join(cell(vals[c], lo, col_w) for c in present_cats))

    print(f"  {'cost (g)':<18}>= {g_lo:<5} {g_note}")
    print(f"  {'cycles (c)':<18}>= {c_lo:<5} {c_note}")
    print(f"  {t_label:<18}>= {t_lo:<5} {t_note}")
    print()

    print("Lower bounds vs. records  (* = bound achieved):")
    print(header)
    print(sep)
    print(row("cost (g)",  g_lo, {c: scores[c]["g"]     for c in present_cats}))
    print(row("cycles (c)", c_lo, {c: scores[c]["c"]     for c in present_cats}))
    print(row(t_label,     t_lo, {c: scores[c][t_key]   for c in present_cats}))

    sums = {c: scores[c]["g"] + scores[c]["c"] + scores[c][t_key] for c in present_cats}
    sum_lo = g_lo + c_lo + t_lo
    print(sep)
    print(row(f"SUM ({sum_label})", sum_lo, sums))
    print()
    if sum_lo == sums.get(sum_cat, sum_lo + 1):
        print(f"=> {sum_cat} record is PROVEN OPTIMAL.")
    else:
        sum_hi = sums[sum_cat]
        pct = 100 * (sum_hi - sum_lo) / sum_hi
        print(f"=> Best {sum_cat} record may be improvable by up to "
              f"{sum_hi - sum_lo} ({pct:.1f}%).")

    # ── Upper bounds from best SUM record ─────────────────────────────────

    sum_hi = sums[sum_cat]
    g_hi = (sum_hi - c_lo - t_lo) // 5 * 5  # every part costs a multiple of 5g
    c_hi = sum_hi - g_lo - t_lo
    t_hi = sum_hi - g_lo - c_lo

    print()
    print(f"Bounds for optimal {sum_label} solution:")
    print(f"  cost (g)          : {format_bound(g_lo, g_hi)}")
    print(f"  cycles (c)        : {format_bound(c_lo, c_hi)}")
    print(f"  {t_label:<18}: {format_bound(t_lo, t_hi)}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Compute proven bounds for the optimal SUM solution.")
    parser.add_argument("--puzzle", default="P007", metavar="ID", help="Puzzle ID (default: P007)")
    args = parser.parse_args()
    main(args)
