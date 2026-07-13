"""
bounds.py — analytic lower bounds on the metrics of an optimal solution.

These are provable floors (cost, cycles, area, instructions) derived from the
puzzle's inputs/outputs/available parts alone — no search, no simulation.
See PLAN.md §7.4/§7.5: a metric is *proven optimal* when one of these bounds
equals the best known record.
"""

import math
from collections import Counter

from puzzle_parser import (
    PuzzleFile,
    ATOM_SALT, ATOM_VITAE, ATOM_MORS, ATOM_REPEAT, ATOM_QUINTESSENCE, ATOM_QUICKSILVER,
    ELEMENTALS,
    PART_BARON, PART_PROJECTION, PART_PURIFICATION,
)
from stoichiometry import RecipeResult, METAL_CHAIN, necessary_inputs
from scheme import critical_path_latency


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


def cycles_lower_bound(pf: PuzzleFile, recipe: RecipeResult) -> tuple[int, str]:
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

    refs
    https://biggieblog.com/battling-the-entire-world-in-opus-magnum/
    """
    products_needed = pf.products_needed()

    N = products_needed
    note = f"output throughput: {products_needed} products, 1/cycle"

    for i, count in recipe.reagent_counts.items():
        if count <= 0:
            continue
        repeats = recipe.reagent_group_size.get(i, 1)
        cycles = 2 * math.ceil(count / repeats)
        if cycles > N:
            N = cycles
            note = (f"input throughput {N}: {count}x needed from input {i}"
                    + f" with {repeats} duplicates" if repeats > 1 else "")

    l_notes = []
    L, l_note = critical_path_latency(pf, recipe)
    l_notes += [(l_note, L)]
    all_repeating = all(any(a.type == ATOM_REPEAT for a in mol.atoms) for mol in pf.outputs)
    if not all_repeating:
        l_notes.append(("drop", 1))  # last drop isn't pipelined away unless every output repeats
    L = sum(val for _, val in l_notes)
    latency_note = (f"input latency {L}: " + " + ".join(f"{name}={val}" for name, val in l_notes))

    return N + L, f"{note}, {latency_note}"


def _needed_parts(pf: PuzzleFile) -> dict:
    """
    Which extra parts a solution is provably forced to buy, from atom-type
    and bond-type presence/absence between inputs and outputs. Shared by
    cost_lower_bound and area_lower_bound (see cost_lower_bound's docstring
    for the exact conditions)
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

    # Projection/purification: a missing output metal reachable by climbing
    # the chain (lead->tin->iron->copper->silver->gold) from an available
    # input metal. Which specific glyph is actually forced depends on parts
    # availability and whether quicksilver is around to feed projection
    # (metal[N]+quicksilver -> metal[N+1]; purification is 2x metal[N] ->
    # metal[N+1], no quicksilver needed):
    #   - only one of the two available -> that one is forced
    #   - both available, no quicksilver anywhere -> projection can't fire,
    #     so purification is forced
    #   - both available and quicksilver present -> either would work, so
    #     the bound can only claim "one of them" (cost/area take the cheaper)
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

    return {
        "bonder": _new(set(output_normal_bonds), set(input_normal_bonds)),
        "unbonder": _new(input_bonds, output_bonds),
        "bonder_prisma": _new(set(output_triplex_bonds), set(input_triplex_bonds)),

        # Calcification: salt in output, only elemental-type sources available in input
        "calcification": (ATOM_SALT in output_atom_types and ATOM_SALT not in input_atom_types
                           and bool(input_atom_types & ELEMENTALS)),

        # Wheel + duplication: an output elemental (air/earth/fire/water) missing
        # from every input has no reagent-sourced catalyst atom available for
        # duplication (salt -> elemental, catalyzed by the elemental itself), so
        # the wheel's free catalyst becomes mandatory — unless quintessence is
        # available as an alternate source (dispersion, below).
        "baron_duplication": (
            needs_elemental
            and ATOM_QUINTESSENCE not in input_atom_types
            and bool(pf.parts_available & PART_BARON)
        ),

        "metal_glyph": metal_glyph,  # None | "projection" | "purification" | "either"

        # Animismus: vitae or mors in output but not in any input.
        "animismus": (
            (ATOM_VITAE in output_atom_types and ATOM_VITAE not in input_atom_types)
            or (ATOM_MORS in output_atom_types and ATOM_MORS not in input_atom_types)
        ),

        # Dispersion: a missing output elemental whose only other route
        # (duplication, above) is blocked — the wheel needed for its catalyst
        # isn't available.
        "dispersion": needs_elemental and not bool(pf.parts_available & PART_BARON),

        # Unification: quintessence in output but not in any input.
        "unification": (ATOM_QUINTESSENCE in output_atom_types
                         and ATOM_QUINTESSENCE not in input_atom_types),
    }


def cost_lower_bound(pf: PuzzleFile) -> tuple[int, str]:
    """
    Minimum cost based on required parts.

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
      - a dispersion glyph (20 g) if any outputs contains (air/earth/fire/water) that is not in any input, and the berlow wheel is disabled
      - a unification glyph (20 g) if any output contains quintessence but none of the inputs
      # ignored for now:
      - DLC:
          - a rejection glyph (20 g) if ...
          - a division glyph (2g g) if ...
          - ravari wheel (30g) + proliferation glyph (20g) = 50g total if ...
      - additional track if access points insufficient
        - full access + partial access (existing bonds or bonder) + no access (?) glyphs
        - access per arm + #track
        - channels

    refs:
    https://biggieblog.com/optimizing-cost-in-opus-magnum/
    # todo check track proofs: stamina potion, Very Dark Thread, Voltaic Coil

    These are hard minimums — a solution with any fewer parts would be invalid.
    """
    needed = _needed_parts(pf)

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
        g_lo += 40  # animismus=20g + 4×track=20g
        reasons.append("1×animismus=20g + 4×track=20g")
    if needed["dispersion"]:
        g_lo += 20
        reasons.append("1×dispersion=20g")
    if needed["unification"]:
        g_lo += 20
        reasons.append("1×unification=20g")

    return g_lo, " + ".join(reasons)


def area_lower_bound(pf: PuzzleFile, recipe: RecipeResult) -> tuple[int, str]:
    """
    Minimum area based on required parts plus one hex per required input/output atom.

    berlow wheel on tracks can expose the underlying hexes:
      - 0 track: 7+0, 2 track: 4+6, 3 track: 3+10
    """
    needed = _needed_parts(pf)

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

    input_atoms = sum(pf.inputs[i].atom_count() for i in necessary_inputs(pf, recipe))
    output_atoms = sum(mol.atom_count() for mol in pf.outputs)
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
